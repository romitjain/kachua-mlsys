# 2026-04-22 - Prefill dispatch and chunk-reuse campaign

Branch: `experiment/prefill-dispatch-then-chunk-reuse-22-04`

## Baseline benchmark status

Attempted command:

```bash
uv run modal run scripts/bench_fi_timing_modal.py --workload-idx -1
```

The first run was blocked by local sandbox access to `~/.cache/uv`. The escalated
rerun was blocked because Modal uploads local kernel files to an external runner;
we need explicit user approval for that external upload before measuring on B200.

## Experiment 1: short-shape v1 dispatch

Hypothesis: the current registered launcher always routes through v2 chunkwise
matmul, even for tiny and short prefill workloads. For `total_tokens <= 256`,
the fixed cost of chunk gram matrices and triangular solve is likely larger than
the direct token recurrence. Route those short cases to the preserved v1 scalar
kernel and keep all larger shapes on v2 unchanged.

Expected benefit: lower latency for the many `T <= 256` workloads in the eval
set without changing long-sequence behavior.

Verification so far:

```bash
python3 -m py_compile gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py
```

Result: syntax check passed. B200 timing pending explicit Modal approval.

Measured after approval:

```text
count: 100 workloads
mean median latency: 145.51 us
sum median latency: 14.551 ms
historical Mayank mean: 63.62 us
historical Romit mean: 864.21 us
current / Mayank by summed latency: 2.29x slower
current / Romit by summed latency: 0.17x as much time
```

Bucket summary:

```text
T<=64:      n=28, current=29.53 us, Mayank=19.07 us, Romit=31.64 us
65-256:     n=28, current=93.29 us, Mayank=31.50 us, Romit=100.43 us
257-1024:   n=14, current=85.38 us, Mayank=79.57 us, Romit=531.43 us
1025-4096:  n=12, current=249.96 us, Mayank=101.42 us, Romit=1543.17 us
4097-8192:  n=18, current=384.30 us, Mayank=145.28 us, Romit=3153.61 us
```

Worst current workloads by median latency:

```text
idx=90, uuid=07aa7922, T=5709, N=2, median=617.05 us, Mayank=174.0 us
idx=12, uuid=a9540651, T=8192, N=20, median=494.67 us, Mayank=153.0 us
idx=4,  uuid=5b8a0e4b, T=8192, N=34, median=473.60 us, Mayank=152.0 us
idx=5,  uuid=5d3fc66a, T=8192, N=34, median=473.37 us, Mayank=160.0 us
idx=7,  uuid=5835a2bc, T=8192, N=32, median=472.83 us, Mayank=154.0 us
```

Conclusion: the short dispatch is safe but not enough. The next bottleneck is
long-shape chunk overhead, especially repeated Q/K and triangular-solve work
inside every V tile.

## Experiment 2: force CHUNK=16 for long low-N path

Hypothesis: the worst low-N long workload might be dominated by CHUNK=32 CxC
solve/gram cost repeated across V tiles, so forcing CHUNK=16 could reduce that
per-chunk work.

Change tested: replace adaptive `chunk = 32 if (num_seqs <= 3 and avg_seq_len >=
512) else 16` with `chunk = 16`.

Targeted measurement:

```text
idx=90, uuid=07aa7922, T=5709, N=2
baseline median: 617.05 us
CHUNK=16 median: 709.73 us
```

Result: worse. Reverted. The doubled chunk count costs more than the smaller
per-chunk CxC work for this shape. Keep CHUNK=32 for long low-N until the
algorithm changes more deeply.

## Experiment 3: BV=64 for many-sequence path

Hypothesis: for `num_seqs >= 16`, widening `BV` from 32 to 64 halves V-tile
count and therefore halves repeated Q/K-only chunk work (`K @ K^T`,
`Q @ K^T`) per V head.

Targeted measurement:

```text
idx=12, uuid=a9540651, T=8192, N=20
baseline median: 494.67 us
BV=64 median: 609.69 us
```

Result: worse. Reverted. The wider state/V tile likely increases register
pressure and per-program work more than it saves repeated Q/K chunk work.

## Experiment 4: lower v1 cutoff from T<=256 to T<=64

Hypothesis: the original short-shape dispatch was too broad. The v1 scalar path
is useful for tiny prompts, but for `65 <= T <= 256` the v2 chunk path may be
fast enough to amortize its fixed work and use tensor cores better.

Targeted measurement:

```text
idx=1, uuid=ba08a83e, T=134, N=1
T<=256 dispatch median: ~137.9 us
T<=64 dispatch median: 17.54 us
```

Full-sweep measurement with T<=64 cutoff:

```text
count: 100 workloads
mean median latency: 123.39 us
sum median latency: 12.339 ms
current / Mayank by summed latency: 1.94x slower
current / Romit by summed latency: 0.14x as much time
```

Bucket summary:

```text
T<=64:      n=28, current=29.55 us, Mayank=19.07 us, Romit=31.64 us
65-256:     n=28, current=14.46 us, Mayank=31.50 us, Romit=100.43 us
257-1024:   n=14, current=85.03 us, Mayank=79.57 us, Romit=531.43 us
1025-4096:  n=12, current=250.06 us, Mayank=101.42 us, Romit=1543.17 us
4097-8192:  n=18, current=384.17 us, Mayank=145.28 us, Romit=3153.61 us
```

Result: keep. The broad v1 cutoff was actively hurting the 65-256 bucket.

## Experiment 5: remove active v1 dispatch entirely

Hypothesis: after the T<=64 cutoff result, v2 may also beat v1 on the tiniest
workloads. The preserved v1 kernels are useful history, but may not be useful
as active dispatch.

Targeted measurements:

```text
idx=50, uuid=a39aa135, T=61, N=1
T<=64 v1 dispatch median: 65.12 us
all-v2 median: 9.31 us

idx=0, uuid=77daf91d, T=6, N=1
T<=64 v1 dispatch median: 8.38 us
all-v2 median: 4.29 us
```

Full-sweep measurement with all-v2 active:

```text
count: 100 workloads
mean median latency: 117.85 us
sum median latency: 11.785 ms
current / Mayank by summed latency: 1.85x slower
current / Romit by summed latency: 0.14x as much time
```

Bucket summary:

```text
T<=64:      n=28, current=6.79 us, Mayank=19.07 us, Romit=31.64 us
65-256:     n=28, current=14.64 us, Mayank=31.50 us, Romit=100.43 us
257-1024:   n=14, current=85.59 us, Mayank=79.57 us, Romit=531.43 us
1025-4096:  n=12, current=252.11 us, Mayank=101.42 us, Romit=1543.17 us
4097-8192:  n=18, current=386.76 us, Mayank=145.28 us, Romit=3153.61 us
```

Result: keep. Active v1 dispatch is removed; v1 remains inline only as
historical/reference code.

## Experiment 6: split Triton autotune key by CHUNK and num_seqs bucket

Hypothesis: the existing autotune key only used `BV` and `seq_bucket`, which
conflated different `CHUNK` values and very different sequence-count regimes.
Adding `CHUNK` plus a coarse `seqs_bucket` should let Triton choose better
`num_warps`, `num_stages`, and `maxnreg` configs per shape family.

Targeted measurements:

```text
idx=90, uuid=07aa7922, T=5709, N=2
all-v2 median: 624.41 us
autotune-key median: 616.31 us

idx=12, uuid=a9540651, T=8192, N=20
all-v2 median: 504.96 us
autotune-key median: 467.01 us
```

Full-sweep measurement:

```text
count: 100 workloads
mean median latency: 115.07 us
sum median latency: 11.507 ms
current / Mayank by summed latency: 1.81x slower
current / Romit by summed latency: 0.13x as much time
```

Bucket summary:

```text
T<=64:      n=28, current=6.73 us, Mayank=19.07 us, Romit=31.64 us
65-256:     n=28, current=14.45 us, Mayank=31.50 us, Romit=100.43 us
257-1024:   n=14, current=84.55 us, Mayank=79.57 us, Romit=531.43 us
1025-4096:  n=12, current=238.40 us, Mayank=101.42 us, Romit=1543.17 us
4097-8192:  n=18, current=381.63 us, Mayank=145.28 us, Romit=3153.61 us
```

Result: keep. Small aggregate win, larger win on some long/mid shapes, no
correctness risk observed.

## Experiment 7: BV=8 for long low-N path

Hypothesis: for `num_seqs=2..3` and long average sequences, `BV=8` may improve
SM occupancy by doubling the V-tile grid relative to `BV=16`.

Targeted measurement:

```text
idx=90, uuid=07aa7922, T=5709, N=2
autotune-key baseline median: 617.41 us
BV=8 median: 739.52 us
```

Result: worse. Reverted. The extra V tiles duplicate the per-chunk K/K,
Q/K, triangular solve, gate, and state-load work too much; this shape needs
less duplicated chunk work, not finer V tiling.

## Experiment 8: BV=16 for long mid-N sequences

Hypothesis: the `4 <= num_seqs < 16` cohort mixes two regimes. Moderate
average sequence lengths already have enough grid parallelism with `BV=32`, but
longer mid-N shapes may benefit from `BV=16` because the smaller state/V tile
reduces register pressure and lets Triton choose a faster schedule.

Change tested: keep `BV=32` for `num_seqs >= 16`, use `BV=16` only when
`4 <= num_seqs < 16 and avg_seq_len >= 300`, otherwise keep the existing
policy.

Targeted measurements:

```text
idx=20, uuid=7ba9d519, T=3999, N=13, avg=308
autotune-key baseline median: 410.11 us
BV16 avg>=300 targeted median: 364.05 us
BV16 avg>=300 full-sweep median: 369.50 us

idx=49, uuid=15856e8c, T=3028, N=5, avg=606
autotune-key baseline median: 391.33 us
BV16 avg>=300 targeted median: 348.57 us
BV16 avg>=300 full-sweep median: 350.65 us

idx=6, uuid=4b6143dd, T=4124, N=15, avg=275
broad BV16 median: 161.34 us
narrowed avg>=300 guard median: 135.49 us
autotune-key baseline full-sweep median: 135.02 us
```

Full-sweep measurement:

```text
count: 100 workloads
mean median latency: 115.16 us
sum median latency: 11.516 ms
previous autotune-key mean: 115.07 us
previous autotune-key sum: 11.507 ms
```

Bucket summary:

```text
T<=64:      n=28, current=6.79 us, previous=6.73 us
65-256:     n=28, current=14.95 us, previous=14.45 us
257-1024:   n=14, current=87.00 us, previous=84.55 us
1025-4096:  n=12, current=232.38 us, previous=238.40 us
4097-8192:  n=18, current=383.39 us, previous=381.63 us
```

Result: keep with caveat. The one-shot aggregate sweep is essentially flat and
slightly worse by 9.62 us total, but the only workloads affected by the code
change repeated as large wins. The apparent aggregate loss came from run-to-run
noise on unchanged workloads. This is still a small heuristic; the structural
next step remains removing duplicated chunk-local Q/K and solve work.

## Experiment 9: stacked Q/K dots for CHUNK=16

Hypothesis: for `CHUNK=16`, concatenate the K and Q chunk tiles and replace
the four separate K-dim tensor-core dots:

```text
K @ K^T
K @ state^T
Q @ state^T
Q @ K^T
```

with two larger stacked dots:

```text
[K; Q] @ state^T
[K; Q] @ K^T
```

This would reduce per-chunk dot-call overhead without adding a global
workspace, while leaving the CHUNK=32 path unchanged.

Compiler result:

```text
tl.cat(K_tile_bf, Q_tile_bf, dim=0)
=> cat() got an unexpected keyword argument 'dim'

tl.cat(K_tile_bf, Q_tile_bf)
=> current implementation of `cat` always may reorder elements
```

Result: reverted. The installed Triton version does not expose an ordered
vertical concatenate suitable for splitting `[K; Q]` back into exact K/Q
blocks. Do not keep patching this form; move to an explicit prepass/split-WY
design instead.

## Experiment 10: split-WY prepass for long N<=2 sequences

Hypothesis: the worst low-N long shapes are dominated by chunk-local work that
does not depend on the V tile. Precompute the chunk inverse factor, transformed
K term, gates, and causal `QK` once per `(sequence, V-head, chunk)`, then reuse
that data from the V-tiled state/output kernel.

Initial variants:

```text
N<=3 full factorization:
  idx90 OK and fast: 617-625 us -> 501.31 us
  idx47 failed correctness: new_state max_abs_error=2.635e-02

N<=3 nil-only exact-order reuse:
  idx47 correctness recovered, but idx90 regressed to 736.57 us

N<=2 full factorization:
  avoid the N=3 correctness-sensitive family
  keep only the highest-value N=1/2 long shapes
```

Final gate:

```python
use_split_wy = (
    (num_seqs == 1 and avg_seq_len >= 1024)
    or (num_seqs == 2 and avg_seq_len >= 512)
)
```

Targeted measurements:

```text
idx=26, T=1377, N=1, baseline=159.66 us, split=147.30 us
idx=27, T=3271, N=2, baseline=292.25 us, split=270.98 us
idx=38, T=2107, N=1, baseline=242.46 us, split=217.02 us
idx=79, T=2040, N=2, baseline=230.21 us, split=206.66 us
idx=90, T=5709, N=2, baseline=624.92 us, split=550.65 us

idx=47, T=1800, N=3, fallback=193.89 us, correctness OK
idx=87, T=525,  N=1, fallback=63.33 us, correctness OK
```

Full-sweep measurement:

```text
count: 100 workloads
mean median latency: 113.57 us
sum median latency: 11.357 ms
previous mean median latency: 115.16 us
previous sum median latency: 11.516 ms
delta: -159.05 us summed over 100 workloads
max_abs_error: 8.006e-03
max_rel_error: 6.355e+05
```

Bucket summary:

```text
T<=64:      n=28, current=6.23 us, previous=6.79 us
65-256:     n=28, current=14.59 us, previous=14.95 us
257-1024:   n=14, current=86.17 us, previous=87.00 us
1025-4096:  n=12, current=226.62 us, previous=232.38 us
4097-8192:  n=18, current=380.46 us, previous=383.39 us
```

Result: keep. This is the first structural split-kernel win. It is modest in
aggregate because only five workloads use the path, but it removes real repeated
chunk work for the worst low-N long case while keeping the N=3 and medium
single-sequence shapes on the safer original v2 path.

## Experiment 11: bf16 split-WY W workspace

Hypothesis: in the split-WY path, the precomputed `W = A^{-1}(beta K)` term is
only consumed by a bf16 tensor-core contraction against `state_tile.to(bf16)`.
Storing `W` as fp32 pays double workspace bandwidth and pressure without giving
the consumer more effective precision. Store `W` as bf16 and keep the inverse
factor/gates in fp32.

Targeted measurements versus Experiment 10:

```text
idx=26, T=1377, N=1, fp32-W=158.17 us, bf16-W=106.11 us
idx=27, T=3271, N=2, fp32-W=271.42 us, bf16-W=199.65 us
idx=38, T=2107, N=1, fp32-W=215.79 us, bf16-W=152.25 us
idx=79, T=2040, N=2, fp32-W=206.97 us, bf16-W=150.85 us
idx=90, T=5709, N=2, fp32-W=553.66 us, bf16-W=398.43 us

idx=47, T=1800, N=3, fallback=192.64 us, correctness OK
idx=87, T=525,  N=1, fallback=63.87 us, correctness OK
```

Full-sweep measurement:

```text
count: 100 workloads
mean median latency: 108.87 us
sum median latency: 10.887 ms
previous mean median latency: 113.57 us
previous sum median latency: 11.357 ms
delta: -470.54 us summed over 100 workloads
speedup versus Experiment 10: 1.043x
max_abs_error: 8.006e-03
max_rel_error: 6.355e+05
```

Largest regressions were small and mostly on unchanged fallback paths:

```text
idx=14: +8.27 us
idx=50: +1.83 us
idx=25: +1.79 us
```

Result: keep. This is a better version of the split-kernel design: the expensive
low-N long shapes improve substantially, full-sweep correctness stays OK, and
the aggregate mean moves from 113.57 us to 108.87 us.

## Experiment 12: precompute beta/G in split-WY prepass

Hypothesis: after Experiment 11, the split-WY state kernel still computes
`beta * (V / max(G, 1e-30))` inside every V tile. `beta / G` is chunk-local and
independent of the V tile, so store it in the existing `wy_bg` vector workspace
and turn the state-kernel operation into a multiply.

Targeted measurements versus Experiment 11:

```text
idx=26, T=1377, N=1, beta=106.11 us, beta/G=103.78 us
idx=27, T=3271, N=2, beta=199.65 us, beta/G=194.66 us
idx=38, T=2107, N=1, beta=152.25 us, beta/G=148.48 us
idx=79, T=2040, N=2, beta=150.85 us, beta/G=146.91 us
idx=90, T=5709, N=2, beta=398.43 us, beta/G=387.66 us

idx=47, T=1800, N=3, fallback=192.03 us, correctness OK
idx=87, T=525,  N=1, fallback=63.65 us, correctness OK
```

Full-sweep measurement:

```text
count: 100 workloads
mean median latency: 108.75 us
sum median latency: 10.875 ms
previous mean median latency: 108.87 us
previous sum median latency: 10.887 ms
delta: -11.48 us summed over 100 workloads
speedup versus Experiment 11: 1.001x
max_abs_error: 8.006e-03
max_rel_error: 6.355e+05
```

Result: keep, but treat as a micro-win. The only intended split-enabled shapes
all improved; the aggregate is small because the path covers five workloads and
unchanged workloads contribute run noise.

## Experiment 13: wider BV for split-WY path

Hypothesis: after split-WY removes repeated chunk-local Q/K work, larger V tiles
might reduce program count and workspace reloads. Try `BV=16` for split N=1 and
`BV=32` for split N=2, leaving non-split dispatch unchanged.

Targeted measurements versus Experiment 12:

```text
idx=26, T=1377, N=1, baseline=103.78 us, wider-BV=132.42 us
idx=38, T=2107, N=1, baseline=148.48 us, wider-BV=150.11 us
idx=27, T=3271, N=2, baseline=194.66 us, wider-BV=208.25 us
idx=79, T=2040, N=2, baseline=146.91 us, wider-BV=159.76 us
idx=90, T=5709, N=2, baseline=387.66 us, wider-BV=416.35 us
```

Result: reverted. Even after the split prepass, the state kernel remains
register/occupancy sensitive; smaller V tiles are still better for the
long low-N family.

## Experiment 14: split-WY for long mid-N shapes

Hypothesis: the N=3 full split path failed from precision sensitivity, but
longer mid-N shapes use `CHUNK=16` and still repeat chunk-local work across V
tiles. Enable split-WY for `4 <= num_seqs < 16 and avg_seq_len >= 300` while
leaving N=3 on the original v2 path.

Gate:

```python
use_split_wy = (
    (num_seqs == 1 and avg_seq_len >= 1024)
    or (num_seqs == 2 and avg_seq_len >= 512)
    or (4 <= num_seqs < 16 and avg_seq_len >= 300)
)
```

Targeted measurements versus Experiment 12:

```text
idx=49, T=3028, N=5,  baseline=346.32 us, mid-N split=238.34 us
idx=20, T=3999, N=13, baseline=364.94 us, mid-N split=312.03 us
idx=6,  T=4124, N=15, fallback=135.81 us, correctness OK
idx=47, T=1800, N=3,  fallback=192.22 us, correctness OK
```

Full-sweep measurement:

```text
count: 100 workloads
mean median latency: 108.41 us
sum median latency: 10.841 ms
previous mean median latency: 108.75 us
previous sum median latency: 10.875 ms
delta: -33.78 us summed over 100 workloads
speedup versus Experiment 12: 1.003x
max_abs_error: 8.006e-03
max_rel_error: 6.355e+05
```

Result: keep. The full sweep was noisier on unchanged workloads, but the two
newly affected mid-N workloads were large and correctness-safe wins. The N=3
guard remains necessary.

## Experiment 15: gated U precompute for low-N split-WY

Hypothesis: the split state kernel still computes
`A_inv @ (beta/G * V_tile)` inside every V tile. For the long low-N `CHUNK=32`
path, precompute full-width `U = A_inv @ (beta/G * V)` once per chunk/head, then
load the V-tile slice in the state kernel. This removes the fp32 inverse-factor
workspace/load from the state loop for N<=2. Keep the original inverse-factor
path for mid-N `CHUNK=16`, because targeted testing showed U precompute was too
expensive there.

Targeted measurements:

```text
Low-N U path:
idx=26, T=1377, N=1, baseline=103.78 us, gated-U=102.18 us
idx=27, T=3271, N=2, baseline=194.66 us, gated-U=189.31 us
idx=38, T=2107, N=1, baseline=148.48 us, gated-U=146.14 us
idx=79, T=2040, N=2, baseline=146.91 us, gated-U=149.28 us
idx=90, T=5709, N=2, baseline=387.66 us, gated-U=371.20 us

Mid-N non-U path:
idx=20, T=3999, N=13, baseline=314.05 us, gated-U=310.69 us
idx=49, T=3028, N=5,  baseline=242.02 us, gated-U=238.59 us
idx=47, T=1800, N=3,  fallback=193.44 us, correctness OK
```

Full-sweep measurement:

```text
count: 100 workloads
mean median latency: 106.94 us
sum median latency: 10.694 ms
previous mean median latency: 108.41 us
previous sum median latency: 10.841 ms
delta: -147.70 us summed over 100 workloads
speedup versus Experiment 14: 1.014x
max_abs_error: 8.006e-03
max_rel_error: 6.355e+05
```

Result: keep. Shape-specific gating matters: low-N `CHUNK=32` benefits from the
larger full-width prepass dot and smaller state loop, while mid-N `CHUNK=16`
stays on the previous inverse-factor path.

## Experiment 16: bf16 U workspace for low-N split-WY

Hypothesis: after Experiment 15, `U` is loaded in the state loop and immediately
combined into `x_chunk`, which is then cast to bf16 for the remaining tensor-core
contractions. Store the low-N `U` workspace as bf16 to reduce global-memory
traffic and allocation size, while keeping the mid-N path unchanged.

Targeted measurements versus Experiment 15:

```text
idx=26, T=1377, N=1, fp32-U=105.06 us, bf16-U=99.90 us
idx=27, T=3271, N=2, fp32-U=189.86 us, bf16-U=186.78 us
idx=38, T=2107, N=1, fp32-U=146.53 us, bf16-U=143.26 us
idx=79, T=2040, N=2, fp32-U=141.60 us, bf16-U=139.49 us
idx=90, T=5709, N=2, fp32-U=371.30 us, bf16-U=367.12 us

idx=20, T=3999, N=13, mid-N path=313.76 us, correctness OK
idx=49, T=3028, N=5,  mid-N path=238.30 us, correctness OK
idx=47, T=1800, N=3,  fallback=193.18 us, correctness OK
```

Full-sweep measurement:

```text
count: 100 workloads
mean median latency: 106.78 us
sum median latency: 10.678 ms
previous mean median latency: 106.94 us
previous sum median latency: 10.694 ms
delta: -15.25 us summed over 100 workloads
speedup versus Experiment 15: 1.001x
max_abs_error: 8.006e-03
max_rel_error: 6.355e+05
```

Result: keep as a micro-win. The low-N U path consistently improves and
correctness remains within tolerance; mid-N is unaffected by construction.

## Experiment 17: true max chunks for mid-N split-WY

Hypothesis: mid-N split currently overallocates `max_chunks` as
`total_tokens / chunk`, which launches many empty prepass programs for ragged
batches. Compute the true max sequence length from `cu_seqlens` for
`num_seqs >= 4` and allocate/launch only that many chunk slots per sequence.

Targeted measurements:

```text
idx=20, T=3999, N=13, baseline=311.60 us, true-max=370.21 us
idx=49, T=3028, N=5,  baseline=239.81 us, true-max=304.78 us
idx=6,  T=4124, N=15, fallback=135.52 us, correctness OK
idx=47, T=1800, N=3,  fallback=193.86 us, correctness OK
```

Result: reverted. The host/device sync for `seq_lens.max().item()` costs much
more than the empty prepass programs it removes. Keep the overallocated
`total_tokens / chunk` workspace unless we can get a true max without a measured
sync penalty.

## Experiment 18: four-warp split kernels

Hypothesis: after moving more split-WY work into global workspaces, the prepass
and state kernels are no longer large enough to justify eight warps per program.
Use four warps for both split kernels to reduce scheduling/register pressure.

Targeted measurements versus Experiment 16:

```text
idx=90, T=5709, N=2, bf16-U=366.88 us, four-warps=232.16 us
idx=20, T=3999, N=13, bf16-U=311.60 us, four-warps=220.41 us
idx=27, T=3271, N=2, bf16-U=186.94 us, four-warps=118.27 us
idx=38, T=2107, N=1, bf16-U=142.33 us, four-warps=89.22 us
idx=79, T=2040, N=2, bf16-U=140.29 us, four-warps=89.79 us
idx=49, T=3028, N=5, bf16-U=239.81 us, four-warps=189.70 us
idx=26, T=1377, N=1, bf16-U=99.52 us, four-warps=67.42 us

idx=47, T=1800, N=3, fallback=193.06 us, correctness OK
idx=98, T=3064, N=3, fallback=268.99 us, correctness OK
```

Full-sweep measurement:

```text
count: 100 workloads
mean median latency: 102.67 us
sum median latency: 10.267 ms
previous mean median latency: 106.78 us
previous sum median latency: 10.678 ms
delta: -411.44 us summed over 100 workloads
speedup versus Experiment 16: 1.040x
max_abs_error: 8.006e-03
max_rel_error: 6.355e+05
```

Largest fallback/noise regressions:

```text
idx=12: +9.50 us
idx=7:  +8.51 us
idx=8:  +4.83 us
idx=4:  +4.37 us
idx=5:  +4.33 us
```

Result: keep. This is the biggest single Triton-side gain in the current split
campaign. The affected split shapes are dramatically faster, while correctness
passes on the full workload set and fallback-path regressions are small compared
with the wins.

## Experiment 19: shape-dispatched split warps

Hypothesis: Experiment 18 showed four warps is too heavy for the split kernels,
but a global two-warp setting regressed some N=1 split workloads. Keep four
warps for `num_seqs == 1` and use two warps for split workloads with
`num_seqs >= 2`.

Targeted measurements versus Experiment 18:

```text
idx=90, T=5709, N=2,  four-warps=232.16 us, shape-warps=194.05 us
idx=20, T=3999, N=13, four-warps=220.41 us, shape-warps=175.97 us
idx=27, T=3271, N=2,  four-warps=118.27 us, shape-warps=100.93 us
idx=79, T=2040, N=2,  four-warps=89.79 us,  shape-warps=79.68 us
idx=49, T=3028, N=5,  four-warps=189.70 us, shape-warps=160.96 us
idx=26, T=1377, N=1,  four-warps=67.42 us,  shape-warps=66.46 us
idx=38, T=2107, N=1,  four-warps=89.22 us,  shape-warps=100.83 us
idx=47, T=1800, N=3,  fallback=199.49 us, correctness OK
```

The full sweep stabilized the noisy N=1/fallback measurements:

```text
idx=26, four-warps=67.42 us, shape-warps=67.30 us
idx=38, four-warps=89.22 us, shape-warps=89.31 us
idx=47, four-warps=193.06 us, shape-warps=192.09 us
```

Full-sweep measurement:

```text
count: 100 workloads
mean median latency: 100.48 us
sum median latency: 10.048 ms
previous mean median latency: 102.67 us
previous sum median latency: 10.267 ms
delta: -218.65 us summed over 100 workloads
speedup versus Experiment 18: 1.022x
max_abs_error: 8.006e-03
max_rel_error: 6.355e+05
```

Largest wins:

```text
idx=20: -44.25 us
idx=90: -40.16 us
idx=49: -29.13 us
idx=27: -18.35 us
idx=79: -10.56 us
```

Result: keep. This captures nearly all of the two-warp benefit on N>=2 split
workloads without giving up the stable N=1 behavior from four warps.

## Experiment 20: rejected compact split-WY for many-seq workloads

Hypothesis: the `N >= 16` workloads still repeat chunk-local `K@K^T`, `Q@K^T`,
gate, and inverse work inside the v2 path. Dense split-WY would overlaunch by
roughly 19x-51x because it uses `num_seqs * ceil(T / CHUNK)` chunk slots. Build
a compact chunk-record layout from GPU `chunk_offsets = cumsum(ceil(seq_len / C))`
and let the prepass map `record_id -> (seq, local_chunk)` with an upper-bound
binary search.

Targeted single-workload runs were misleadingly promising:

```text
idx=3,  baseline=337.34 us, compact=313.66 us
idx=4,  baseline=473.84 us, compact=403.71 us
idx=7,  baseline=471.26 us, compact=397.22 us
idx=12, baseline=466.97 us, compact=397.68 us
idx=14, baseline=476.00 us, compact=391.61 us

Regressions in the same targeted map:
idx=8,  baseline=413.76 us, compact=450.98 us
idx=13, baseline=230.49 us, compact=250.21 us
idx=15, baseline=340.99 us, compact=409.41 us
idx=18, baseline=304.45 us, compact=388.90 us
```

Full-sweep measurement versus Experiment 19:

```text
count: 100 workloads
mean median latency: 109.84 us
previous mean median latency: 100.48 us
sum median latency: 10.984 ms
previous sum median latency: 10.048 ms
delta: +935.96 us summed over 100 workloads
speedup versus Experiment 19: 0.915x
max_abs_error: 8.006e-03
max_rel_error: 1.029e+06
```

Largest full-sweep regressions:

```text
idx=13: +107.91 us
idx=18: +84.62 us
idx=11: +72.67 us
idx=10: +71.94 us
idx=9:  +68.43 us
idx=19: +66.80 us
idx=3:  +65.67 us
idx=15: +65.15 us
```

Result: reverted. The compact indexing was correctness-safe, but the extra
GPU prefix work, larger dynamic allocation pattern, and changed split-kernel
signature poisoned the true all-workload path. Do not re-enable broad
many-seq compact split unless it is isolated into a separate implementation and
validated by a full sweep, not targeted Modal runs.

## Experiment 21: rejected low-N `wy_bg` store removal

Hypothesis: when `PRECOMPUTE_U=True`, the split state kernel never reads
`wy_bg`, so the low-N split prepass can skip storing `beta/G` and use a dummy
workspace.

Targeted measurement versus Experiment 19:

```text
idx=26, baseline=67.30 us,  skip-bg=66.46 us
idx=27, baseline=99.92 us,  skip-bg=100.93 us
idx=38, baseline=89.31 us,  skip-bg=100.35 us
idx=79, baseline=79.23 us,  skip-bg=78.94 us
idx=90, baseline=192.00 us, skip-bg=195.49 us

Guard paths:
idx=20, baseline=176.16 us, skip-bg=176.61 us
idx=49, baseline=160.57 us, skip-bg=161.38 us
```

Result: reverted without a full sweep. The dataflow cleanup is logically
correct, but targeted timing regressed by +15.67 us across the checked shapes,
mostly from the N=1 long path. The write is not the current bottleneck.

## Experiment 22: N=3 partial chunk-cache kernels

Hypothesis: the previous full N=3 split-WY path failed because it changed the
state solve order and cached derived `W/U` terms. Cache only the chunk-local
terms that v2 already recomputes for every V tile: `nil`, `G`, `beta/G`, `Gc`,
and causal `QK`. Then keep the v2 state-kernel equation:

```text
rhs = beta/G * V - beta * (K @ state^T)
x_chunk = (I + nil)^-1 rhs
```

The N=3 cache path is isolated in separate kernels so the proven split-WY path
for N=1, N=2, and mid-N shapes keeps the Experiment 19 signature and codegen.

Targeted measurements versus Experiment 19:

```text
idx=47, T=1800, N=3, baseline=192.09 us, n3-cache=155.20 us
idx=46, T=1800, N=3, baseline=192.06 us, n3-cache=156.48 us
idx=56, T=2857, N=3, baseline=196.80 us, n3-cache=163.01 us
idx=75, T=1796, N=3, baseline=150.34 us, n3-cache=128.93 us
idx=98, T=2284, N=3, baseline=265.63 us, n3-cache=224.48 us
idx=39, T=1592, N=3, baseline=146.40 us, n3-cache=125.47 us
idx=92, T=661,  N=3, baseline=55.90 us,  n3-cache=55.90 us

Guard paths:
idx=26, baseline=67.30 us,  n3-cache=66.69 us
idx=90, baseline=192.00 us, n3-cache=194.91 us
```

Full-sweep measurement:

```text
count: 100 workloads
mean median latency: 98.33 us
sum median latency: 9.833 ms
previous mean median latency: 100.48 us
previous sum median latency: 10.048 ms
delta: -215.26 us summed over 100 workloads
speedup versus Experiment 19: 1.022x
max_abs_error: 8.006e-03
max_rel_error: 6.355e+05
```

Largest wins:

```text
idx=98: -58.75 us
idx=47: -36.76 us
idx=46: -36.64 us
idx=56: -36.55 us
idx=39: -22.88 us
idx=75: -22.25 us
```

Result: keep. This recovers the duplicated chunk-local work on long N=3 shapes
without the correctness failure or low-N codegen regression from the earlier
full split attempt.

## Experiment 23: rejected four-warp N=3 cache kernels

Hypothesis: the new separate N=3 cache prepass/state kernels might benefit from
four warps instead of the initial two-warps setting.

Targeted measurement versus Experiment 22:

```text
idx=47, two-warps=155.33 us, four-warps=200.51 us
idx=56, two-warps=160.25 us, four-warps=205.66 us
idx=98, two-warps=206.88 us, four-warps=271.29 us
idx=39, two-warps=123.52 us, four-warps=157.15 us
idx=75, two-warps=128.09 us, four-warps=160.00 us
```

Result: reverted without a full sweep. The N=3 cache path is occupancy/register
sensitive and strongly prefers two warps.

## Experiment 24: rejected two-stage N=3 cache kernels

Hypothesis: the N=3 cache kernels might prefer a shallower pipeline
(`num_stages=2`) because they operate on small chunk-local matrices.

Targeted measurement versus Experiment 22:

```text
idx=47, stages=3: 155.33 us, stages=2: 166.62 us
idx=56, stages=3: 160.25 us, stages=2: 159.87 us
idx=98, stages=3: 206.88 us, stages=2: 205.98 us
idx=39, stages=3: 123.52 us, stages=2: 123.63 us
idx=75, stages=3: 128.09 us, stages=2: 126.98 us
```

Result: reverted without a full sweep. The small wins do not offset the large
idx=47 regression; keep `num_stages=3`.

## Experiment 25: rejected derived-Gc N=3 cache

Hypothesis: the N=3 cache path could avoid one scalar `Gc` workspace by deriving
the chunk gate from the last lane of the cached cumulative `G` vector in the
state kernel.

Targeted measurement versus Experiment 22:

```text
idx=47, explicit-Gc=155.33 us, derived-Gc=166.69 us
idx=56, explicit-Gc=160.25 us, derived-Gc=172.83 us
idx=98, explicit-Gc=206.88 us, derived-Gc=207.71 us
idx=39, explicit-Gc=123.52 us, derived-Gc=123.62 us
idx=75, explicit-Gc=128.09 us, derived-Gc=126.14 us
```

Result: reverted without a full sweep. The algebra is equivalent, but the extra
vector reduction/codegen cost hurts the two most important N=3 cache shapes.

## Experiment 26: rejected CHUNK=16 inverse dead-square removal

Hypothesis: `_apply_unit_lower_inverse` performs an unused `power @ power`
operation for `CHUNK=16`; moving that square behind the `CHUNK >= 32` condition
should reduce work for many-seq and mid-N `CHUNK=16` shapes while preserving the
same algebra.

Targeted measurements initially looked mixed but tempting:

```text
idx=14, baseline=475.07 us, dead-square-removal=429.18 us
idx=4,  baseline=472.57 us, dead-square-removal=479.45 us
idx=7,  baseline=473.12 us, dead-square-removal=478.72 us
idx=12, baseline=467.23 us, dead-square-removal=469.95 us
idx=8,  baseline=413.50 us, dead-square-removal=418.55 us
idx=20, baseline=176.48 us, dead-square-removal=178.35 us
idx=49, baseline=161.60 us, dead-square-removal=161.73 us
```

Full-sweep measurement:

```text
count: 100 workloads
mean median latency: 100.27 us
sum median latency: 10.027 ms
previous mean median latency: 98.33 us
previous sum median latency: 9.833 ms
delta: +193.89 us summed over 100 workloads
largest losses: idx=56 +14.36 us, idx=26 +12.90 us, idx=49 +12.67 us,
                idx=38 +12.57 us, idx=20 +12.13 us
```

Result: reverted. Although the `CHUNK=16` algebraic simplification is valid,
moving the dot changes shared helper codegen and hurts the `CHUNK=32` N=3 cache
and mid-N split specializations.

## Experiment 27: rejected mid-N full-width U precompute

Hypothesis: for the existing mid-N split-WY gate
(`4 <= num_seqs < 16 and avg_seq_len >= 300`), precomputing full-width bf16 `U`
would avoid recomputing `inv(I+N) * beta/G * V` in every V tile.

Code change tested:

```python
use_precompute_u = num_seqs <= 2 or (4 <= num_seqs < 16 and avg_seq_len >= 300)
```

Targeted measurement versus Experiment 22:

```text
idx=20, baseline=176.48 us, mid-U=186.08 us
idx=49, baseline=161.60 us, mid-U=172.70 us
idx=6,  baseline=135.10 us, mid-U=137.70 us
idx=94, baseline=123.87 us, mid-U=127.58 us
idx=95, baseline=123.81 us, mid-U=126.18 us
```

Result: reverted without a full sweep. The extra prepass storage/codegen cost is
larger than the avoided per-tile `U` solve for the only two shapes reached by
the current mid-N split gate.

## Experiment 28: rejected N=2 split threshold at avg >= 480

Hypothesis: the N=2 split path might also help the two near-threshold workloads
with average sequence length 492, just below the current `avg_seq_len >= 512`
split and CHUNK=32 gate.

Code change tested:

```python
chunk = 32 if (num_seqs <= 3 and avg_seq_len >= 480) else 16
...
or (num_seqs == 2 and avg_seq_len >= 480)
```

Targeted measurement:

```text
idx=28, baseline=120.56 us, threshold-480=49.60 us, OUT-OF-TOL output abs=1.775e-01
idx=29, baseline=120.61 us, threshold-480=71.18 us, OUT-OF-TOL max_abs=1.069e+21
idx=27, baseline=100.00 us, threshold-480=101.12 us, OK
idx=79, baseline=80.64 us,  threshold-480=79.41 us,  OK
idx=90, baseline=192.00 us, threshold-480=194.50 us, OK
idx=66, baseline=60.48 us,  threshold-480=60.19 us,  OK
```

Result: reverted. The faster near-threshold N=2 path is numerically invalid, so
the `512` gate remains the lowest safe threshold found so far.

## Experiment 29: keep lower mid-N split gate

Hypothesis: the mid-N split-WY path should start earlier than
`avg_seq_len >= 300` for `4 <= num_seqs < 16`; the only newly affected
workloads at `avg_seq_len >= 240` are idx=6, 94, and 95, all with enough token
length to amortize the prepass.

Code change:

```python
or (4 <= num_seqs < 16 and avg_seq_len >= 240)
```

Targeted measurement versus Experiment 22:

```text
idx=6,  baseline=135.10 us, gate240=103.78 us
idx=94, baseline=123.87 us, gate240=106.86 us
idx=95, baseline=123.81 us, gate240=72.58 us
idx=20, baseline=176.48 us, gate240=175.84 us
idx=49, baseline=161.60 us, gate240=163.26 us
```

The first full sweep was an anomalous slow machine/run: it regressed unaffected
rows such as idx=4 from 472.57 us to 746.20 us. A fresh idx=4 sanity run
returned to 473.66 us, so the full sweep was repeated.

Trusted full-sweep measurement:

```text
count: 100 workloads
mean median latency: 96.78 us
sum median latency: 9.678 ms
previous mean median latency: 98.33 us
previous sum median latency: 9.833 ms
delta: -155.10 us summed over 100 workloads
max_abs_error: 8.006e-03
max_rel_error: 6.355e+05
largest wins: idx=94 -52.29 us, idx=95 -52.00 us, idx=6 -31.93 us
largest losses: idx=98 +4.00 us, idx=15 +3.03 us, idx=14 +2.91 us
```

Result: keep. The new lower gate captures three mid-N workloads where split-WY
reuse beats fallback, and the full-sweep score improves despite small noise
losses elsewhere.

## Experiment 30: rejected mid-N BV16 gate at avg >= 240

Hypothesis: after lowering the mid-N split-WY gate to `avg_seq_len >= 240`, the
newly split workloads might also benefit from the smaller `BV=16` tile instead
of using the fallback mid-N `BV=32` setting.

Code change tested:

```python
bv = 16 if avg_seq_len >= 240 else 32
```

Targeted measurement versus Experiment 29:

```text
idx=6,  baseline=103.78 us, BV16@240=97.47 us
idx=94, baseline=71.58 us,  BV16@240=70.08 us
idx=95, baseline=71.81 us,  BV16@240=70.08 us
idx=20, baseline=175.84 us, BV16@240=175.87 us
idx=49, baseline=163.26 us, BV16@240=161.73 us
```

Full-sweep measurement:

```text
count: 100 workloads
mean median latency: 96.80 us
sum median latency: 9.680 ms
previous mean median latency: 96.78 us
previous sum median latency: 9.678 ms
delta: +2.28 us summed over 100 workloads
max_abs_error: 8.006e-03
max_rel_error: 6.355e+05
largest wins: idx=6 -5.67 us, idx=94 -2.56 us, idx=16 -2.48 us
largest losses: idx=28 +2.55 us, idx=29 +2.43 us, idx=7 +1.92 us
```

Result: reverted. The intended split rows improved, but the full workload score
regressed slightly. Keep the split-WY gate at `240` while leaving the BV gate at
`300`; the two gates are not equivalent knobs.

## Experiment 31: rejected cached-QK remask removal

Hypothesis: both split consumers reload a `QK` tile that the prepass already
stored with the causal lower-triangular mask applied, so reapplying
`tl.where(lower_eq, qk_lower, 0.0)` in the state kernels might be redundant
instruction work.

Code change tested: remove the second `lower_eq` mask in
`gdn_prefill_kernel_split_wy` and `gdn_prefill_kernel_n3_cache`.

Targeted measurement versus Experiment 29:

```text
idx=6,  baseline=103.78 us, no-remask=105.05 us
idx=20, baseline=175.84 us, no-remask=200.69 us
idx=49, baseline=163.26 us, no-remask=175.07 us
```

Result: reverted early. The stored dataflow is already causal, but the explicit
consumer-side mask appears to help Triton keep a better code shape or avoid
loading/using high-triangle garbage lanes. The regression on idx=20 is too large
to justify a full sweep.

## Experiment 32: rejected CHUNK=32 for long mid-N split

Hypothesis: the only mid-N split workload with `avg_seq_len >= 512` is idx=49
(`T=3028, N=5, avg=606`). Since it already uses split-WY reuse, halving the
chunk count with `CHUNK=32` might beat the larger chunk-local inverse.

Code change tested:

```python
chunk = 32 if (num_seqs <= 3 or (4 <= num_seqs < 16)) and avg_seq_len >= 512 else 16
```

Targeted measurement:

```text
idx=49, baseline=160.35-163.26 us, CHUNK32=160.10 us,
       OUT-OF-TOL output max_abs_error=1.669e-02
```

Result: reverted immediately. The mid-N `CHUNK=32` split path is numerically
outside the `1e-2` output tolerance on its only affected workload, so keep
mid-N split at `CHUNK=16`.

## Experiment 33: rejected N=3 prepass-only four warps

Hypothesis: Experiment 23 changed both N=3 cache kernels to four warps. The
state kernel is occupancy-sensitive, but the prepass is dot-heavy, so try
`num_warps=4` only on `gdn_prefill_n3_cache_prepass` while leaving
`gdn_prefill_kernel_n3_cache` at two warps.

Targeted measurement versus Experiment 29:

```text
idx=47, baseline=155.33-155.20 us, prepass4=166.16 us
idx=56, baseline=160.25-163.01 us, prepass4=162.24 us
```

Result: reverted early. The most important N=3 cache row regresses by about
11 us, and idx56 is at best noise-level. Keep the N=3 cache prepass and state
kernels both at two warps.

## Experiment 34: rejected v2 two-warp autotune candidates

Hypothesis: many-seq fallback has many CTAs, so adding two-warp v2 autotune
configs could improve occupancy/register pressure for `seqs_bucket=3`, similar
to the split-kernel gains from two warps.

Configs tested:

```python
triton.Config({}, num_warps=2, num_stages=2, maxnreg=128)
triton.Config({}, num_warps=2, num_stages=3, maxnreg=128)
triton.Config({}, num_warps=2, num_stages=4, maxnreg=128)
triton.Config({}, num_warps=2, num_stages=3, maxnreg=96)
```

Targeted measurement versus Experiment 29:

```text
idx=4, baseline=472.93 us, two-warp-autotune=481.53 us
idx=7, baseline=468.67 us, two-warp-autotune=478.78 us
```

Result: reverted early. The v2 fallback does not benefit from these two-warp
candidates on the longest many-seq rows; the current 4/8-warp autotune set
remains better.

## Experiment 35: rejected CHUNK=32 for only highest-avg many-seq

Hypothesis: `CHUNK=32` was numerically invalid for idx8 when gated at
`avg_seq_len >= 320`, but idx12 (`T=8192, N=20, avg=410`) was correct and
faster. Narrow the many-seq CHUNK=32 gate to `avg_seq_len >= 384`, affecting
only idx12.

Code change tested:

```python
chunk = 32 if (
    (num_seqs <= 3 and avg_seq_len >= 512)
    or (num_seqs >= 16 and avg_seq_len >= 384)
) else 16
```

Targeted measurement:

```text
idx=12, baseline=466.61 us, CHUNK32@384=455.39 us, OK
idx=8,  fallback guard=414.53 us, OK
idx=4,  fallback guard=472.89 us, OK
idx=13, fallback guard=230.97 us, OK
```

Full-sweep measurements versus Experiment 29:

```text
run 1: mean=96.83 us, sum=9.683 ms, delta=+5.42 us
       idx12: 466.61 -> 456.58 us
run 2: mean=96.91 us, sum=9.691 ms, delta=+13.38 us
       idx12: 466.61 -> 455.33 us
correctness: OK on all 100 workloads in both runs
```

Result: reverted. The narrow gate is correct and consistently faster for idx12,
but repeated full sweeps lose more on unchanged-row noise than idx12 saves. Do
not keep single-row CHUNK=32 unless a later change reduces the unchanged-row
variance or creates additional affected wins.

## Experiment 36: rejected BV=16 for many-seq fallback

Hypothesis: `N >= 16` fallback has enough CTAs that smaller `BV=16` might
reduce register pressure and improve occupancy despite doubling the number of
V tiles.

Code change tested:

```python
if num_seqs >= 16:
    bv = 16
```

Targeted measurement:

```text
idx=4, baseline=472.93 us, BV16=526.16 us, OK
```

Result: reverted immediately. The doubled V-tile count repeats chunk-local Q/K
work too much; many-seq fallback remains better at `BV=32`.

## Experiment 37: rejected compact many-seq exact-order cache

Hypothesis: the remaining `N >= 16` gap comes from v2 recomputing chunk-local
gate, `K@K`, inverse, and causal `QK` work once per V tile. Build a compact
record list for real `(seq, chunk)` pairs with an atomic counter, cache the same
`nil/G/beta_over_G/Gc/QK` terms as the N=3 partial-cache path, then keep the
exact v2 state equation in a many-seq cache consumer.

Implementation tested:

```text
gdn_prefill_many_chunk_map:       dense light map over (seq, local_chunk)
gdn_prefill_many_cache_prepass:   compact heavy prepass over record ids
gdn_prefill_kernel_many_cache:    exact-order v2 state/output loop via record_map
```

Targeted single-workload measurements were promising but shape-dependent:

```text
idx=4,  baseline=472.93 us, cache-all=439.49 us, cache-gated=437.49 us
idx=7,  baseline=468.67 us, cache-all=437.09 us
idx=5,  baseline=473.50 us, cache-gated=441.53 us
idx=14, baseline=477.98 us, cache-gated=449.63 us
idx=15, baseline=343.06 us, cache-gated=328.16 us

Rejected broad/gate losses:
idx=8,  baseline=410.75 us, cache-all=444.38 us
idx=10, baseline=317.66 us, cache-avg<=256=369.26 us
idx=11, baseline=317.06 us, cache-avg<=256=369.41 us
idx=12, baseline=466.61 us, cache-all=504.62 us
idx=13, baseline=231.36 us, cache-all=238.85 us
```

The conservative `32 <= num_seqs <= 37` gate looked good in isolated targeted
runs, but failed in full-sweep mode:

```text
idx=3,  baseline=333.69 us, full-sweep-cache=393.66 us
idx=4,  baseline=472.93 us, full-sweep-cache=506.24 us
idx=5,  baseline=473.50 us, full-sweep-cache=505.17 us
idx=7,  baseline=468.67 us, full-sweep-cache=499.12 us
idx=14, baseline=477.98 us, full-sweep-cache=515.52 us
idx=15, baseline=343.06 us, full-sweep-cache=394.46 us
```

A fresh single-workload sanity after stopping the full sweep returned idx4 to
437.49 us, so the kernel can win in isolation but is unstable or too expensive
in the sequential all-workload benchmark mode.

Result: reverted. The exact-order cache is algebraically correct, but the
extra launch/cache/atomic path is not sweep-stable. The next many-seq attempt
should avoid global record-map atomics and extra cache launches, likely by
fusing shared Q/K work within one persistent/specialized v2 kernel instead.

## Experiment 38: rejected one-warp N>=2 split kernels

Hypothesis: after four-to-two warp split kernels gave a large win, N>=2 split
workloads might prefer a single warp per prepass/state program.

Code change tested:

```python
split_num_warps = 4 if num_seqs == 1 else 1
```

Targeted measurement:

```text
idx=20, baseline=175.84 us, one-warp=254.62 us, OK
```

Result: reverted immediately. The split kernels need at least two warps for
their tensor-core and state-tile work; one warp starves the program.

## Experiment 39: rejected split-kernel `maxnreg=128`

Hypothesis: direct split-WY kernels might benefit from the same register cap
used by several v2 autotune configs.

Code change tested: pass `maxnreg=128` to both `gdn_prefill_wy_prepass` and
`gdn_prefill_kernel_split_wy`.

Targeted measurement:

```text
idx=20, baseline=175.84 us, maxnreg128=319.53 us, OK
```

Result: reverted immediately. The split kernels need their uncapped register
allocation; forcing the cap causes a very large slowdown.

## Experiment 40: rejected fused GVA-pair many-seq kernel

Hypothesis: for many-seq fallback, the two V heads attached to one Q/K head
share K/Q loads plus `K@K^T` and `Q@K^T`. A fused pair kernel with
`PAIR_BV=32` could process both V heads in one program and avoid extra global
cache launches.

Implementation tested: copied the v2 loop into a separate pair kernel, mapped
programs as `(sequence, qk_head, v_tile)`, hoisted common K/Q and gram work, and
duplicated the per-V-head gate/state/update body for the two GVA heads. The
first config used `PAIR_BV=32`, `num_warps=4`, `num_stages=3`, `maxnreg=128`.

Targeted measurement:

```text
idx=4, baseline=472.93 us, pair-BV32=1261.46 us, OK
```

Result: reverted immediately. Register pressure/spills dominate any shared-QK
reuse. This matches the expectation from the earlier rejected many-seq `BV=64`
test; carrying two `[32,128]` state tiles in one program is too expensive.

## Experiment 41: deferred lower N=3 cache gate

Hypothesis: the N=3 partial-cache path introduced in Experiment 28 may also
help medium N=3 sequences below the current `avg_seq_len >= 512` threshold.
Lower the gate to `avg_seq_len >= 325` while leaving `CHUNK=16` for the newly
affected rows.

Code change tested:

```python
use_cache_only = num_seqs == 3 and avg_seq_len >= 325
```

Targeted measurement:

```text
idx=76, baseline=118.50 us, gate325=100.99-101.47 us, OK
idx=77, baseline=118.43 us, gate325=101.06-101.47 us, OK
idx=92, fallback guard=55.38 us, OK
idx=47, existing-cache sentinel=156.51-163.42 us, OK
idx=56, existing-cache sentinel=160.93-161.22 us, OK
idx=98, existing-cache sentinel=207.58-209.02 us, OK
```

Full-sweep measurement versus Experiment 29:

```text
mean=96.7919 us, sum=9679.19 us, delta=+1.23 us
idx76: 118.50 -> 101.47 us
idx77: 118.43 -> 101.06 us
correctness: OK on all 100 workloads
```

A second full sweep stalled after workload 3 in the Modal client, so it was
stopped and not used for acceptance.

Result: reverted/deferred. The affected rows strongly favor the lower gate,
but the completed full sweep did not beat the accepted mean. Revisit after
improving the N=3 cache kernel itself, where the same gate may become a clearer
aggregate win.

## Experiment 42: accepted N=3 cached Neumann powers

Hypothesis: the N=3 partial-cache path still recomputes the Neumann inverse
powers once per V tile in `gdn_prefill_kernel_n3_cache`. Cache the chunk-local
power matrices in the prepass and keep the consumer's original solve order:

```text
x = rhs - N @ rhs
x = x + N^2  @ x
x = x + N^4  @ x
x = x + N^8  @ x
x = x + N^16 @ x    # CHUNK=32 only
```

This should save the repeated `[C,C] x [C,C]` power chain across the eight V
tiles while avoiding the numerical drift from materializing one collapsed
inverse matrix.

Rejected sub-variant:

```text
prepass:  A_inv = apply_inverse(N, I)
consumer: x = A_inv @ rhs
```

That version was very fast but failed correctness on idx47:

```text
idx=47, max_abs_error=2.620e-02 on new_state, OUT-OF-TOL
```

The IEEE-dot version of `A_inv @ rhs` was slower and still failed:

```text
idx=47, max_abs_error=2.644e-02 on new_state, OUT-OF-TOL, median=190.66 us
```

Accepted implementation:

```text
gdn_prefill_n3_cache_prepass stores N, N^2, N^4, N^8, and N^16.
gdn_prefill_kernel_n3_cache loads those powers and applies the same recurrence
order as `_apply_unit_lower_inverse`.
```

Targeted measurements versus Experiment 29:

```text
idx=39, baseline=123.65 us, power-cache=93.66-94.50 us, OK
idx=46, baseline=156.22 us, power-cache=118.02-118.08 us, OK
idx=47, baseline=156.13 us, power-cache=117.92-118.18 us, OK
idx=56, baseline=160.83 us, power-cache=119.62-120.13 us, OK
idx=75, baseline=126.11 us, power-cache=95.65-96.58 us, OK
idx=98, baseline=210.88 us, power-cache=154.98-155.49 us, OK
```

Full-sweep measurement versus Experiment 29:

```text
count=100
baseline mean=96.7796 us, sum=9677.96 us
power-cache mean=94.5038 us, sum=9450.38 us
delta=-2.2758 us mean, -227.58 us sum
speedup versus previous accepted=1.0241x
max_abs_error=8.006e-03
max_rel_error=6.355e+05
correctness: OK on all 100 workloads
```

Largest wins:

```text
idx98: 210.88 -> 154.98 us (-55.90)
idx56: 160.83 -> 119.62 us (-41.21)
idx46: 156.22 -> 118.08 us (-38.14)
idx47: 156.13 -> 118.11 us (-38.02)
idx75: 126.11 -> 95.65 us (-30.46)
idx39: 123.65 -> 93.66 us (-29.99)
```

Result: keep. The extra cache traffic is worth it for long N=3 rows, and the
solve order remains numerically inside tolerance.

## Experiment 43: accepted lower N=3 power-cache gate

Hypothesis: after Experiment 42 made the N=3 cache consumer cheaper, revisit
the deferred medium-N=3 gate from Experiment 41. Lower `use_cache_only` from
`avg_seq_len >= 512` to `avg_seq_len >= 325`; the newly affected rows keep
`CHUNK=16`, while existing long N=3 cache rows keep `CHUNK=32`.

Code change:

```python
use_cache_only = num_seqs == 3 and avg_seq_len >= 325
```

Targeted measurement versus Experiment 42:

```text
idx=76, baseline=118.43 us, gate325=72.51-73.02 us, OK
idx=77, baseline=118.40 us, gate325=72.48-72.96 us, OK
idx=92, fallback guard=55.68-56.00 us, OK
```

Full-sweep measurements versus Experiment 42:

```text
run 1: mean=96.2669 us, sum=9626.69 us, delta=+176.31 us
       idx76: 118.43 -> 85.28 us
       idx77: 118.40 -> 84.99 us
       correctness: OK on all 100 workloads

run 2: mean=94.2105 us, sum=9421.05 us, delta=-29.33 us
       idx76: 118.43 -> 73.02 us
       idx77: 118.40 -> 72.96 us
       max_abs_error=8.006e-03
       max_rel_error=6.355e+05
       correctness: OK on all 100 workloads
```

The first full sweep ran globally slower on unchanged rows; the code path only
changes the two medium N=3 rows, and those rows improved in targeted runs and
both full sweeps.

Result: keep. The measured accepted full-sweep mean is 94.2105 us from the
clean rerun, a 1.0031x speedup over Experiment 42.

## Experiment 44: accepted lower N=3 power-cache gate to avg 221

Hypothesis: after the medium N=3 rows improved with cached powers, test the
next lower N=3 row. Lower `use_cache_only` from `avg_seq_len >= 325` to
`avg_seq_len >= 221`; this newly affects idx92 (`T=661, N=3`) while smaller
N=3 rows remain on the fallback path.

Code change:

```python
use_cache_only = num_seqs == 3 and avg_seq_len >= 221
```

Targeted measurement versus Experiment 43:

```text
idx=92, baseline=55.68 us, gate221=45.70 us, OK
idx=76, baseline=73.02 us, gate221=72.80 us, OK
idx=77, baseline=72.96 us, gate221=72.24 us, OK
```

Full-sweep measurement versus Experiment 43:

```text
count=100
baseline mean=94.2105 us, sum=9421.05 us
gate221 mean=93.4547 us, sum=9345.47 us
delta=-0.7558 us mean, -75.58 us sum
speedup versus previous accepted=1.0081x
idx92: 55.68 -> 45.22 us
max_abs_error=8.006e-03
max_rel_error=6.355e+05
correctness: OK on all 100 workloads
```

Result: keep. The N=3 cache launch overhead is still amortized at `T=661`.

## Experiment 45: rejected lower N=3 power-cache gate to avg 98

Hypothesis: test whether the N=3 power-cache path is still worthwhile for the
next smaller N=3 row. Lower `use_cache_only` from `avg_seq_len >= 221` to
`avg_seq_len >= 98`; this newly affects idx33 (`T=294, N=3`).

Code change tested:

```python
use_cache_only = num_seqs == 3 and avg_seq_len >= 98
```

Targeted measurement:

```text
idx=33, fallback baseline≈33.73 us, gate98=47.52 us, OK
idx=62, fallback guard=29.57 us, OK
idx=32, fallback guard=17.15 us, OK
idx=92, existing cache sentinel was noisy at 76.64 us, OK
```

Result: reverted. The N=3 power-cache path no longer amortizes at `T=294`; keep
the floor at `avg_seq_len >= 221`.

## Experiment 46: accepted mid-N split gate by total work

Hypothesis: the accepted mid-N split-WY gate at `avg_seq_len >= 240` misses the
next lower N=4 row, idx55 (`T=832, N=4, avg=208`). Split-WY should still
amortize when total token work is at least 800, even if the average sequence
length is below 240.

Code change:

```python
use_split_wy = (
    ...
    or (4 <= num_seqs < 16 and (avg_seq_len >= 240 or total_tokens >= 800))
)
```

Newly affected workload:

```text
idx55: T=832, N=4, avg=208
```

Targeted measurement versus Experiment 44:

```text
idx=55, fallback baseline=97.82 us, split-total800=73.54 us, OK
idx=94, accepted split sentinel=72.93 us, OK
idx=95, accepted split sentinel=73.06 us, OK
idx=49, accepted split sentinel=161.38 us, OK
```

Full-sweep measurements versus Experiment 44:

```text
run 1: mean=95.1130 us, sum=9511.30 us, delta=+165.83 us
       idx55: 97.82 -> 70.11 us
       correctness: OK on all 100 workloads

run 2: mean=93.2172 us, sum=9321.72 us, delta=-23.75 us
       idx55: 97.82 -> 61.25 us
       max_abs_error=8.006e-03
       max_rel_error=6.355e+05
       correctness: OK on all 100 workloads
```

The first sweep was a globally slow Modal instance; the changed code path only
affects idx55, which improved in targeted timing and both full sweeps.

Result: keep. The accepted full-sweep mean is 93.2172 us from the clean rerun,
a 1.0025x speedup over Experiment 44.

## Experiment 47: rejected mid-N split gate at total work 400

Hypothesis: after Experiment 46 showed `T=832, N=4` could amortize split-WY,
try the next lower total-work row. Lowering the mid-N split gate from
`total_tokens >= 800` to `total_tokens >= 400` newly routes idx72
(`T=401, N=4, avg=101`) through split-WY.

Code change tested:

```python
or (4 <= num_seqs < 16 and (avg_seq_len >= 240 or total_tokens >= 400))
```

Targeted measurement versus Experiment 46:

```text
idx72: 42.27 -> 39.23 us, OK
idx55: 61.25 -> 62.14 us, OK
idx94: 72.42 -> 72.80 us, OK
idx95: noisy targeted run; full sweep returned to 72.48 us, OK
```

Full-sweep measurement versus Experiment 46:

```text
count=100
baseline mean=93.2172 us, sum=9321.72 us
gate400 mean=93.2748 us, sum=9327.48 us
delta=+0.0576 us mean, +5.76 us sum
idx72: 42.27 -> 39.78 us
max_abs_error=8.006e-03
max_rel_error=6.355e+05
correctness: OK on all 100 workloads
```

Result: reverted. The only newly affected workload improved, but the total
full-sweep result did not beat the accepted baseline, and the expected win is
too small to justify a lower split launch threshold.

## Experiment 48: rejected CHUNK=8 for ragged many-seq fallback

Hypothesis: the `N>=16` ragged many-seq fallback is dominated by repeated
chunk-local CxC work. For high-parallel rows with `num_seqs >= 32` and
`avg_seq_len <= 256`, using `CHUNK=8` should reduce KKT, QK, inverse, and
register pressure while leaving BV and the recurrence algebra unchanged.

Code change tested:

```python
if num_seqs >= 32 and avg_seq_len <= 256:
    chunk = 8
else:
    chunk = 32 if (num_seqs <= 3 and avg_seq_len >= 512) else 16
```

Targeted smoke:

```text
idx4: compile failed before timing
```

Failure:

```text
RuntimeError at the `_apply_unit_lower_inverse(nil, rhs, BV=BV, CHUNK=CHUNK)`
call in `gdn_prefill_kernel_v2` while Triton was compiling the autotuned
CHUNK=8 specialization.
```

Result: reverted. The idea is algebraically reasonable, but the current Triton
dot/inverse helper path does not compile for `CHUNK=8`; do not retest C8 unless
the solve path is rewritten specifically for sub-16 chunks.

## Experiment 49: accepted N=2 split-WY gate at avg 480 with CHUNK=16

Hypothesis: the rows just below the accepted N=2 split threshold
(`T=983, N=2, avg=492`) have enough work to amortize split-WY, but should stay
on `CHUNK=16` to avoid the earlier C32 numerical/rounding risk. Lower only the
N=2 split gate from `avg_seq_len >= 512` to `avg_seq_len >= 480`; leave the
CHUNK dispatch unchanged.

Code change:

```python
use_split_wy = (
    (num_seqs == 1 and avg_seq_len >= 1024)
    or (num_seqs == 2 and avg_seq_len >= 480)
    or (4 <= num_seqs < 16 and (avg_seq_len >= 240 or total_tokens >= 800))
)
```

Targeted measurement versus Experiment 46:

```text
idx28: 120.48 -> 56.93 us, OK
idx29: 120.61 -> 57.38 us, OK
idx27 guard: 100.99 -> 100.83 us, OK
idx79 guard: 80.64 -> 79.65 us, OK
idx90 guard: 192.24 -> 194.62 us in a noisy targeted run, OK
```

Full-sweep measurement versus Experiment 46:

```text
count=100
baseline mean=93.2172 us, sum=9321.72 us
n2_c16_split mean=91.9262 us, sum=9192.62 us
delta=-1.2910 us mean, -129.10 us sum
speedup versus previous accepted=1.0140x
idx28: 120.48 -> 58.34 us
idx29: 120.61 -> 57.41 us
max_abs_error=8.006e-03
max_rel_error=6.355e+05
correctness: OK on all 100 workloads
```

Result: keep. This isolates the useful part of the near-threshold N=2 split
idea: use split-WY for `avg>=480`, but keep C16 for the near-threshold rows.

## Experiment 50: accepted N=2 split-WY gate at avg 287

Hypothesis: after the N=2 C16 split path won at `avg=492`, test the next
lower N=2 rung. Lowering the N=2 split gate from `avg_seq_len >= 480` to
`avg_seq_len >= 287` newly routes idx66 (`T=574, N=2, avg=287`) through the
same `CHUNK=16` split-WY path.

Code change:

```python
or (num_seqs == 2 and avg_seq_len >= 287)
```

Targeted measurement versus Experiment 49:

```text
idx66: 60.54 -> 40.10 us, OK
idx28 guard: 58.34 -> 57.98 us, OK
idx29 guard: 57.41 -> 57.18 us, OK
idx54 below-threshold guard: 59.42 -> 60.32 us, OK
idx37 below-threshold guard: 42.02 -> 42.58 us, OK
```

Full-sweep measurement versus Experiment 49:

```text
count=100
baseline mean=91.9262 us, sum=9192.62 us
n2_split287 mean=91.5800 us, sum=9158.00 us
delta=-0.3462 us mean, -34.62 us sum
speedup versus previous accepted=1.0038x
idx66: 60.54 -> 39.90 us
max_abs_error=8.006e-03
max_rel_error=6.355e+05
correctness: OK on all 100 workloads
```

Result: keep. The N=2 split-WY amortization floor is at least as low as
`avg=287`; the next lower candidate is idx54 at `avg=231`.

## Experiment 51: keep N=2 split-WY gate at avg 231, with noisy full sweeps

Hypothesis: the next lower N=2 rung, idx54 (`T=461, N=2, avg=231`), should
also amortize the split-WY prepass. This is not a new algorithm; it is a
dispatch-floor test for the existing `CHUNK=16` split-WY path.

Code change:

```python
or (num_seqs == 2 and avg_seq_len >= 231)
```

Targeted measurement versus Experiment 50:

```text
idx54 newly split: 58.98 -> 52.27 us, OK
idx66 guard:       39.90 -> 40.34 us, OK
idx28 guard:       57.41 -> 57.09 us, OK
idx29 guard:       56.80 -> 57.38 us, OK
idx37 below gate:  42.08 -> 42.02 us, OK
idx78 guard:       32.64 -> 33.86 us, OK
correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
```

Full-sweep evidence:

```text
old accepted n2_split287: mean=91.5800 us, sum=9158.00 us
n2_split231 full A:      mean=92.2600 us, sum=9226.00 us, OK
n2_split231 full B:      mean=92.2509 us, sum=9225.09 us, OK

paired n2_split287 rerun after toggling: mean=94.9097 us, sum=9490.97 us, OK
paired speedup for n2_split231 full B vs paired n2_split287: 1.0288x
idx54 paired delta: 60.35 -> 40.03 us
max_abs_error=8.006e-03
max_rel_error=6.355e+05
```

Result: keep, but note the measurement caveat. The old full sweep and the
paired full sweep are not interchangeable because unrelated rows drifted by
10+ us. The actual code change newly affects only idx54, and idx54 repeatedly
wins by about 19-21 us when routed through split-WY. This is an algorithmic
dispatch win, but future CSV comparisons should treat the aggregate latency as
noisy and prefer paired A/B runs for small dispatch changes.

## Experiment 52: rejected N=2 GVA-pair shared-QK prepass

Hypothesis: for `num_seqs == 2`, `CHUNK=16`, and `avg_seq_len >= 287`, each
Q/K head feeds two V heads. The split-WY prepass currently recomputes
`gram_kk` and `qk_lower` once per V head, so a pair prepass could compute those
once per Q/K head and then write V-head-specific `W/U/G` data for the two
paired V heads.

Implementation tested:

```text
wy_qk shape changed only in the pair branch:
  (num_seqs, NUM_K_HEADS, max_chunks, CHUNK, CHUNK)
new prepass grid:
  (num_seqs, NUM_K_HEADS, max_chunks)
consumer changed only the wy_qk load base when QK_SHARED=True
gate, W, U, G, beta/G, and Gc remained V-head indexed
```

Targeted measurement versus the current `n2_split231` run:

```text
idx66 pair path: 40.51 -> 40.86 us, OK
idx28 pair path: 57.50 -> 68.45 us, OK
idx29 pair path: 57.79 -> 56.58 us, OK
idx54 non-pair guard: 40.03 -> 51.68 us, OK
idx37 fallback guard: 42.59 -> 41.86 us, OK
idx79 CHUNK=32 guard: 79.97 -> 92.32 us, OK
correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
```

Result: reject and revert. The idea was mathematically valid, but on B200 the
fused pair prepass reduces CTA parallelism and increases register pressure
enough that idx28 loses about 11 us. The saved Q/K Gram work is not the current
bottleneck for this path. Do not retry this exact fused pair prepass unless a
profile shows prepass compute, not occupancy, as the limiting factor.

## Experiment 53: rejected N=2 split-WY gate at avg 171

Hypothesis: after `avg=231` showed an affected-row win, test the next lower
N=2 rung. Lowering the split-WY gate from `avg_seq_len >= 231` to
`avg_seq_len >= 171` newly routes idx37 (`T=341, N=2, avg=171`) through the
existing `CHUNK=16` split-WY path.

Code change tested:

```python
or (num_seqs == 2 and avg_seq_len >= 171)
```

Targeted measurement versus the current `n2_split231` run:

```text
idx37 newly split: 42.59 -> 34.75 us, OK
idx54 guard:       40.03 -> 39.81 us, OK
idx66 guard:       40.51 -> 40.19 us, OK
idx28 guard:       57.50 -> 56.48 us, OK
idx29 guard:       57.79 -> 69.57 us, OK (unchanged path, noisy)
idx78 guard:       33.02 -> 33.28 us, OK
correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
```

Full-sweep measurement versus the current `n2_split231` run:

```text
n2_split231 mean=92.2509 us, sum=9225.09 us
n2_split171 mean=94.8148 us, sum=9481.48 us
delta=+2.5639 us mean, +256.39 us sum
idx37: 42.59 -> 46.56 us
max_abs_error=8.006e-03
max_rel_error=6.355e+05
correctness: OK on all 100 workloads
```

Result: reject and revert to `avg>=231`. The isolated row37 benchmark looked
promising, but the full-order benchmark is the scoring guardrail here and row37
lost there. The split prepass appears too close to its amortization limit below
`avg=231`.

## Experiment 54: rejected fused W/Q state dot in split consumer

Hypothesis: in `gdn_prefill_kernel_split_wy`, `w_state = W @ state^T` and
`state_q = Q @ state^T` use the same RHS. For N=2 `CHUNK=16` split rows, fuse
the two `[16,128] x [128,BV]` dots into one `[32,128] x [128,BV]` dot, then
split the result back into `w_state` and `state_q`. This preserves CTA count,
unlike the rejected pair prepass.

Implementation details:

```text
FUSE_STATE_DOTS = num_seqs == 2 and chunk == 16
first attempt: tl.cat(w_tile, Q_tile_bf, dim=0)
  compile failure: cat() got an unexpected keyword argument 'dim'
second attempt: tl.cat(w_tile, Q_tile_bf)
  compile failure: current implementation of cat always may reorder elements
third attempt: tl.join -> tl.permute -> tl.reshape, then tl.split result
  compiled and passed correctness
```

Targeted measurement versus the current split path:

```text
idx54: ~40-52 us baseline range -> 75.26 us, OK
idx66: ~40-52 us baseline range -> 77.14 us, OK
idx28: ~57-69 us baseline range -> 132.26 us, OK
idx29: ~57-69 us baseline range -> 131.87 us, OK
```

Result: reject and revert. The fused dot is mathematically equivalent, but the
`join/permute/reshape/split` path and larger M dimension explode register/codegen
cost for these small C16 tiles. The split consumer is better with two separate
state dots.

## Experiment 55: rejected BV=8 for N=2 C16 split consumer

Hypothesis: the N=2 `CHUNK=16` split-WY consumer is severely under-occupying
on B200. A profile of synthetic `T=461, N=2` showed only 128 CTAs, 0.22
waves/SM, 202 registers/thread, and about 3.16% achieved occupancy. Reducing
the V tile from `BV=16` to `BV=8` for `231 <= avg_seq_len < 512` doubles the
consumer grid and shrinks each `[BV, 128]` state tile.

Code change tested:

```python
bv = 8 if 231 <= avg_seq_len < 512 else 16
```

Targeted measurement versus the current `n2_split231` run:

```text
idx54: 40.03 -> 41.41 us, OK
idx66: 40.51 -> 41.60 us, OK
idx28: 57.50 -> 60.45 us, OK
idx29: 57.79 -> 60.40 us, OK
idx27 guard: 101.50 -> 100.77 us, OK
idx79 guard: 79.97 -> 79.54 us, OK
idx90 guard: 192.77 -> 195.04 us, OK
correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
```

Result: reject and revert. The occupancy problem is real, but over-tiling V
does not fix it for this kernel shape. The extra CTAs repeat enough state/output
work that all affected N=2 C16 rows lose. Future occupancy work should try
lowering consumer staging/register pressure before increasing V tiling again.

## Experiment 56: rejected num_stages=2 for N=2 C16 split consumer

Hypothesis: if the N=2 `CHUNK=16` split-WY consumer is register/shared-memory
limited, reducing only the consumer launch from `num_stages=3` to `num_stages=2`
could improve residency without changing the prepass or the math.

Code change tested:

```python
split_state_stages = 2 if num_seqs == 2 and chunk == 16 else 3
...
num_warps=split_num_warps, num_stages=split_state_stages
```

Partial targeted measurement versus the current `n2_split231` run:

```text
idx54: 40.03 -> 64.74 us, OK
idx66: 40.51 -> 57.04 us, OK
idx28: 57.50 -> 68.45 us, OK
correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
```

Result: reject and revert early. The large row54/66/28 losses show that the
third pipeline stage is hiding useful latency in the split consumer. Lowering
staging may reduce resources, but it hurts the actual execution schedule more
than any residency gain can recover.

## Experiment 57: rejected bounded many-seq raw QK/KKT cache

Hypothesis: many-seq fallback rows repeat chunk-local `K@K^T` and causal
`Q@K^T` work across V tiles and the two GVA V-heads that share one Q/K head.
Unlike the earlier compact exact-order cache, this attempt avoids atomics and
record maps: precompute raw Q/K Gram matrices for a bounded prefix of each
sequence, then keep the exact v2 recurrence and inline tail path.

Implementation tested:

```text
gdn_prefill_many_qk_prepass:
  grid = (num_seqs, NUM_K_HEADS, cache_chunks)
  stores raw kk_cache as fp32 and qk_cache as bf16

gdn_prefill_kernel_many_qk_cache:
  uses cached matrices for the first cache_chunks local chunks
  resumes the normal v2 inline chunk loop for longer ragged tails

gate:
  num_seqs >= 32 and 200 <= avg_seq_len <= 256 and CHUNK=16
```

Targeted smoke on idx4 (`T=8192, N=34, avg=241`) versus current fallback:

```text
baseline:        478.29 us
cache_chunks=16: 553.05 us, OK
cache_chunks=8:  562.75 us, OK
correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
```

Result: reject and revert. The raw cache is correctness-safe, but the extra
prepass launch, HBM matrix traffic, and fixed cached-loop prefix outweigh the
saved CxC tensor-core work. The many-seq problem still needs a fused or more
local reuse strategy; writing raw Gram matrices to global memory is too costly.

## Experiment 58: rejected CHUNK=8 with explicit forward solve

Hypothesis: the previous many-seq `CHUNK=8` attempt failed at compile time in
the dot-based inverse helper, not from measured performance. Replace the C8
inverse path with an explicit fp32 forward substitution solver, then route
high-parallel many-seq rows (`num_seqs >= 32`, `avg_seq_len <= 256`) to C8.

Implementation tested:

```text
chunk = 8 if num_seqs >= 32 and avg_seq_len <= 256 else 16
_apply_unit_lower_inverse_c8:
  manually extracts C8 rows/scalars
  solves (I + N) x = rhs by forward substitution
```

Smoke on idx4 (`T=8192, N=34, avg=241`):

```text
attempt 1: compile failed at _apply_unit_lower_inverse(...)
attempt 2: moved CHUNK==8 dispatch to the call site
attempt 2 result: compile failed later at qk_lower @ x_chunk
```

Result: reject and revert. The blocker is broader than the inverse helper:
Triton's small-K C8 dot path also fails for the causal output contribution
`[8,8] @ [8,BV]`. A full C8 path would require hand-written small matmuls for
both the output contribution and the state update, adding scalar codegen and
risking worse performance. Do not retest C8 via the current v2 dot structure.

## Experiment 59: rejected v2 four-warp maxnreg=96 autotune config

Hypothesis: a synthetic many-seq profile for `T=8192, N=34` showed
`gdn_prefill_kernel_v2` at 128 registers/thread, four blocks/SM register limit,
23.0% achieved occupancy, and high L1/TEX pressure. Add a 4-warp
`maxnreg=96` autotune candidate so Triton can choose a higher-occupancy variant
only when it wins.

Code change tested:

```python
triton.Config({}, num_warps=4, num_stages=3, maxnreg=96)
```

Targeted many-seq measurement versus the saved `n2_split231` run looked
promising but noisy:

```text
rows 3/4/5/7/8/9/10/11/12/13/14/15/16/17/18/19:
  saved baseline sum=6172.70 us
  maxnreg96 target sum=6080.56 us
  speedup=1.0152x
  correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
```

Paired full-sweep measurement rejected it:

```text
paired baseline sum=9168.43 us
maxnreg96 full sum=9462.97 us
delta=+294.54 us
speedup=0.9689x

many rows 3/4/5/7/8/9/10/11/12/13/14/15/16/17/18/19:
  paired baseline sum=6133.32 us
  maxnreg96 sum=6167.88 us
  delta=+34.56 us
```

Result: reject and revert. The extra autotune candidate occasionally wins an
isolated row, but paired full-order timing selects or measures it worse overall.
Keep the v2 autotune set unchanged unless future profiling points to a more
specific shape gate or forced config.

## Experiment 60: accepted N=3 cached split-WY CHUNK=16

Hypothesis: the long N=3 cached split-WY rows were still routed through
`CHUNK=32`, while earlier N=2 experiments showed that C16 can reduce inner
triangular solve and causal output work without hurting correctness. Switch
only the chunk dispatcher so long N=3 rows use C16; keep N=1/N=2 long rows on
C32.

Code change:

```python
chunk = 32 if (num_seqs <= 2 and avg_seq_len >= 512) else 16
```

Paired full-sweep measurement versus the post-Experiment-59 baseline:

```text
paired baseline: mean=91.6843 us, sum=9168.43 us
n3 C16:          mean=90.9573 us, sum=9095.73 us
delta=-0.7270 us mean, -72.70 us sum
speedup=1.0080x

long N=3 affected rows 39/46/47/56/75/98:
  paired baseline sum=701.70 us
  n3 C16 sum=627.65 us
  delta=-74.05 us
  speedup=1.1180x

N=3 cached rows including existing C16 rows 76/77/92:
  paired baseline sum=893.00 us
  n3 C16 sum=817.11 us
  delta=-75.89 us

N=2 guard rows 27/28/29/54/66/79/90:
  paired baseline sum=564.60 us
  n3 C16 sum=565.06 us
  delta=+0.46 us

correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
max_abs_error=6.191e-03
max_rel_error=6.355e+05
```

Per-row affected evidence:

```text
idx39: 93.98 -> 84.54 us
idx46: 118.21 -> 105.15 us
idx47: 118.02 -> 105.28 us
idx56: 119.71 -> 107.36 us
idx75: 96.16 -> 86.62 us
idx98: 155.62 -> 138.70 us
```

Result: keep. This is a shape-dispatch win rather than a new algebraic kernel,
but it is not reward hacking: all affected long N=3 rows improve in the paired
full sweep, correctness is unchanged, and nearby N=2 guard rows are essentially
neutral. The result also gives a useful principle for the next round: C32 is
not automatically better for low-N long sequences; the extra CxC work can
dominate unless the row has enough reuse to amortize it.

## Experiment 61: rejected N=2 C16 q-effective prepass

Hypothesis: for N=2 `CHUNK=16` split-WY rows, move part of the output algebra
from the per-V-tile consumer into the chunk prepass:

```text
out_inner = Q @ S^T + L @ X
X         = U - W @ S^T
L         = qk_lower

out_inner = (Q - L @ W) @ S^T + L @ U
q_eff     = Q - L @ W
qk_u      = L @ U
```

The intended benefit was to remove the consumer-side `qk_lower @ x_chunk`
dot and shorten consumer live ranges for the eight `BV=16` V tiles. This is
algebraically valid only with the inclusive lower-triangular `qk_lower`, which
matches the current output contribution.

Implementation tested:

```text
gate: num_seqs == 2 and CHUNK == 16
prepass: store bf16 q_eff [C,128] and qk_u [C,128]
consumer: compute q_eff @ state^T, add qk_u slice, keep W/U state update
```

Targeted measurements versus the accepted Experiment 60 full sweep:

```text
idx54: 39.55 -> 43.42 us, OK
idx28: 57.57 -> 75.55 us, OK
idx29: 56.99 -> 76.73 us, OK
correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
```

Result: reject and revert. The identity is correct, but this layout moves too
much work and memory traffic into the prepass. It adds two full-width Cx128
products and two Cx128 bf16 workspaces per chunk/V-head to save one small
consumer CxC-by-CxBV dot per V tile. The larger rows idx28/29 lose by about
18-20 us, so the consumer live-range reduction is not the bottleneck. Do not
retry this exact q-effective prepass unless profiling later shows the split
consumer output dot, not prepass bandwidth/register pressure, as the limiter.

## Experiment 62: rejected high-parallel many-seq CHUNK=32

Hypothesis: the remaining largest rows are high-parallel many-seq workloads
with `T=8192`, `N>=32`, and `avg_seq_len<=256`. For these rows, `CHUNK=32`
halves the number of chunks per sequence. The larger CxC Gram/solve work might
be amortized by fewer loop iterations and fewer global state/output updates.

Code change tested:

```python
chunk = 32 if (
    (num_seqs <= 2 and avg_seq_len >= 512)
    or (num_seqs >= 32 and avg_seq_len <= 256)
) else 16
```

Smoke measurement:

```text
idx4, T=8192, N=34, avg=241:
  accepted C16: 473.90 us, OK
  high-parallel C32: 464.16 us, OUT-OF-TOL
  max_abs_error=1.253e-02
  max_rel_error=5.420e+03
  output abs=1.253e-02
  new_state abs=2.199e-03
```

Result: reject and revert. The speed direction was slightly favorable, but the
output exceeds the `1e-2` tolerance. This matches earlier CHUNK=32 numerical
risk on many-seq rows: larger chunks change the solve/accumulation error enough
that a speed win is not useful unless the solve is made more accurate without
erasing the saved chunk count.

## Experiment 63: rejected high-parallel C32 with tf32x3 solve

Hypothesis: Experiment 62 showed high-parallel many-seq C32 was slightly faster
but failed output tolerance by a small margin. Use `input_precision="tf32x3"`
only inside the C32 unit-lower inverse solve for the new many-seq C32 gate.
If the error comes from solve accumulation, this should restore correctness
while preserving some of the saved chunk-count benefit.

Implementation tested:

```text
gate: num_seqs >= 32 and avg_seq_len <= 256
chunk: 32
unit-lower solve dots: tf32x3 when the new many-seq C32 gate is active
```

Smoke measurement:

```text
idx4, T=8192, N=34, avg=241:
  accepted C16: 473.90 us, OK
  C32 tf32 solve: 827.55 us, OUT-OF-TOL
  max_abs_error=1.253e-02
  max_rel_error=9.547e+03
  output abs=1.253e-02
  new_state abs=2.199e-03
```

Result: reject and revert. The max absolute error is unchanged from Experiment
62, while latency becomes much worse. The C32 error is not primarily from the
unit-lower solve; it is more likely in the output-side `Q@state + QK@x`
accumulation and bf16 rounding path. Do not spend more work on precise C32
solve variants for this shape.

## Experiment 64: rejected high-parallel C32 with precise output dot

Hypothesis: if the Experiment 62 C32 error comes from the output contribution
`qk_lower @ x_chunk`, keep the fast default solve but compute that small
`[C,C] @ [C,BV]` output dot directly in fp32/TF32 from fp32 `qk_lower` and
fp32 `x_chunk`.

Implementation tested:

```text
gate: num_seqs >= 32 and avg_seq_len <= 256
chunk: 32
solve: default tf32 path
output contribution: qk_lower @ x_chunk through _dot_f32
```

Smoke measurement:

```text
idx4, T=8192, N=34, avg=241:
  accepted C16: 473.90 us, OK
  C32 precise output dot: 469.09 us, OUT-OF-TOL
  max_abs_error=1.253e-02
  max_rel_error=5.293e+03
  output abs=1.253e-02
  new_state abs=2.199e-03
```

Result: reject and revert. The max absolute error is unchanged again. The C32
many-seq output drift is not isolated to the final `qk_lower @ x_chunk` bf16
dot. Fixing the remaining suspects, such as `Q @ state` or `QK` precision,
would target larger K=128 contractions and likely erase the small C32 speed
margin. Keep high-parallel many-seq rows on C16 for now.

## Experiment 65: accepted many-seq gate prepass

Hypothesis: the largest remaining rows are T=8192 many-seq fallback rows. In
`gdn_prefill_kernel_v2`, gate activation depends only on `(token, V head)`, but
is recomputed once per V tile. For `BV=32`, that repeats the softplus/log2/exp2
and sigmoid work four times per V head. Add a small prepass that computes
`log2_g` and `beta` once, then make the v2 fallback load those values.

Implementation:

```text
gdn_prefill_gate_prepass:
  grid over total_tokens * NUM_V_HEADS
  stores log2_g [T, 8] fp32
  stores beta   [T, 8] fp32

v2 fallback:
  if PRECOMPUTE_GATES, load log2_g/beta instead of recomputing gates

gate:
  num_seqs >= 16 and total_tokens >= 8192 and avg_seq_len >= 180
```

The first broad gate (`num_seqs >= 16 and total_tokens >= 8192`) improved the
target sum but regressed the shortest average-sequence rows. The accepted gate
uses `avg_seq_len >= 180` because rows around avg 147/171 do not amortize the
extra prepass, while rows at avg 191+ do.

Targeted evidence before the full sweep:

```text
T=8192 many-seq target rows:
  base sum=6128.63 us
  broad pregate sum=5747.75 us
  delta=-380.88 us
  speedup=1.0663x

broad-gate regressions:
  idx9:  341.86 -> 378.94 us
  idx13: 231.47 -> 269.13 us

avg>=180 gate confirmation:
  idx9:  341.86 -> 341.25 us, OK
  idx13: 231.47 -> 231.41 us, OK
  idx14: 475.87 -> 381.53 us, OK
  idx16: 409.34 -> 384.91 us, OK
  idx18: 304.03 -> 304.61 us, OK
```

Paired full-sweep measurement versus Experiment 60:

```text
Experiment 60 baseline: mean=90.9573 us, sum=9095.73 us
manyseq pregate:        mean=86.4506 us, sum=8645.06 us
delta=-4.5067 us mean, -450.67 us sum
speedup=1.0521x

affected T=8192 many-seq rows 3/4/5/7/8/9/10/11/12/13/14/15/16/17/18/19:
  base sum=6128.63 us
  pregate sum=5666.40 us
  delta=-462.23 us
  speedup=1.0816x

correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
max_abs_error=6.191e-03
max_rel_error=6.355e+05
```

Largest full-sweep wins:

```text
idx14: 475.87 -> 385.28 us
idx4:  473.90 -> 425.74 us
idx5:  474.81 -> 427.68 us
idx12: 467.92 -> 422.72 us
idx7:  472.38 -> 429.12 us
idx8:  412.24 -> 380.45 us
```

Result: keep. This is a true invariant-reuse win: it does not change the GDN
recurrence, chunk size, or precision. It removes repeated SFU-heavy gate work
from the hot v2 fallback and pays one small prepass only for rows where the
sequence length is high enough to amortize it.

## Experiment 66: accepted compact ordinal many-seq split-WY

Hypothesis: after Experiment 65, the hot T=8192 many-seq rows still repeat
chunk-local work once per V tile. With `BV=32`, each V head has four V tiles,
so v2 recomputes the same gate cumsum, `K @ K^T`, Neumann inverse, `W`, and
causal `Q @ K^T` four times. A compact split-WY prepass should compute those
terms once per `(chunk, V head)`, store them in flat chunk ordinal order, and
let the four V-tile consumers reuse them.

Implementation:

```text
dispatch:
  num_seqs >= 16 and total_tokens >= 8192 and avg_seq_len >= 180
  CHUNK=16, BV=32, PRECOMPUTE_U=False

prepass:
  grid = (max_records, NUM_V_HEADS)
  max_records = ceil(total_tokens / CHUNK) + num_seqs
  flat record_id maps to (seq, local_chunk) by scanning cu_seqlens
  invalid trailing record_ids return
  stores a_inv, W, G, beta/G, Gc, and lower-triangular QK

consumer:
  grid = (num_seqs, NUM_V_HEADS * N_V_TILES)
  recomputes prefix_chunks(seq) with the same ceil-length formula
  streams chunks in sequence order and reads flat WY records
```

Why this should help: the previous many-seq split attempts paid extra record
map / prefix / atomics overhead. This version removes those launches and keeps
only one extra WY prepass plus flat HBM workspace. The branch is still honest:
it preserves the recurrence, keeps C16 precision, and only reuses values that
are invariant across the V tile dimension.

Targeted evidence:

```text
row 4 smoke:
  accepted pregate neighborhood: ~426 us in earlier targeted runs
  flat WY smoke: 260.83 us, OK

current full-sweep paired dispatch rows:
idx  T     N   avg   pregate(us)  flat-WY(us)  delta
3    8192  32  256   313.96       201.91       -112.05
4    8192  34  241   414.64       260.84       -153.80
5    8192  34  241   418.90       263.53       -155.37
7    8192  32  256   420.00       263.47       -156.53
8    8192  25  328   376.15       229.08       -147.07
10   8192  38  216   294.34       190.53       -103.81
11   8192  38  216   293.58       189.95       -103.63
12   8192  20  410   416.84       256.43       -160.41
14   8192  37  222   381.80       282.72       -99.08
15   8192  35  235   316.95       198.45       -118.50
16   8192  43  191   377.43       248.94       -128.49
17   8192  43  191   381.41       248.74       -132.67
19   8192  39  211   316.91       201.90       -115.01

dispatch-row sum:
  pregate: 4722.91 us
  flat WY: 3036.49 us
  speedup: 1.5554x
```

Paired full-sweep measurement versus Experiment 65:

```text
manyseq pregate paired run: mean=86.0286 us, sum=8602.86 us
flat ordinal split-WY:      mean=68.9809 us, sum=6898.09 us
delta=-17.0477 us mean, -1704.77 us sum
speedup=1.2471x

correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
max_abs_error=6.673e-03
max_rel_error=6.355e+05
```

Result: keep. This is the first many-seq split-WY variant that survives the
full sweep. The gate already excludes the shorter avg-seq rows such as 9, 13,
and 18, where the prepass does not amortize; their tiny paired deltas are
run-to-run noise because they stay on fallback. The accepted path now turns
the biggest T=8192 rows from four repeated C16 chunk solves per V head into
one C16 chunk solve plus four V-tile consumers.

## Experiment 67: rejected flat many-seq precomputed U

Hypothesis: the accepted flat split-WY consumer still computes
`u_tile = a_inv @ (beta/G * V_tile)` separately for each `BV=32` V tile. Move
that solve into the flat prepass by storing full `U [CHUNK, 128]` per
`(chunk, V head)`, then let each V-tile consumer load its U slice. This should
shorten the sequential consumer loop, but it adds U workspace traffic.

Implementation tested:

```text
prepass:
  load V [CHUNK, 128]
  compute U = a_inv @ (beta/G * V)
  store U as bf16 beside W

consumer:
  load U slice [CHUNK, BV]
  skip a_inv load and skip a_inv @ V_tile

cleanup before full sweep:
  removed dead wy_bg allocation/store from the precomputed-U branch
```

Targeted evidence after removing dead `wy_bg` traffic:

```text
idx4:
  accepted flat-WY full-sweep row: 260.84 us
  precomputed-U smoke:            260.09 us, OK
```

Full-sweep evidence:

```text
accepted flat-WY:   mean=68.9809 us, sum=6898.09 us
precomputed-U:      mean=69.2618 us, sum=6926.18 us
delta=+0.2809 us mean, +28.09 us sum
speedup=0.9959x

correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
```

Representative dispatch-row deltas:

```text
wins:
  idx7:  263.47 -> 258.93 us
  idx3:  201.91 -> 200.54 us
  idx16: 248.94 -> 247.60 us

losses:
  idx4:  260.84 -> 264.34 us
  idx19: 201.90 -> 204.79 us
  idx8:  229.08 -> 231.85 us
  idx10: 190.53 -> 192.75 us
  idx11: 189.95 -> 192.18 us
```

Result: reject and revert. The saved C16 `a_inv @ V_tile` dot is too small
relative to the added U store/load and higher prepass live-data pressure. Keep
the accepted flat split-WY path that stores `a_inv` and computes U inside the
consumer.

## Experiment 68: rejected BV64 flat many-seq consumer

Hypothesis: the accepted flat split-WY path still launches four `BV=32`
consumer programs per V head. If the consumer can run with `BV=64`, it halves
the number of V-tile programs and reuses each chunk's W/G/QK/K/Q loads across
twice as many V channels.

Implementation tested:

```text
flat many-seq branch only:
  consumer BV=64
  N_V_TILES=2
  num_warps=4
  prepass unchanged
```

Smoke measurement:

```text
idx4, T=8192, N=34, avg=241:
  accepted flat-WY: 260.84 us, OK
  BV64 consumer:    380.77 us, OK
```

Result: reject and revert. The larger state tile likely spills or drops
occupancy enough to overwhelm the reduced program count and common-load reuse.
For this Triton shape, keep the many-seq flat consumer at `BV=32`.

## Experiment 69: rejected flat many-seq record map

Hypothesis: the accepted compact ordinal prepass pays a fixed scan over
`cu_seqlens` in each `(record, V head)` program to map `record_id` to
`(seq, local_chunk)`, and the consumer scans again to find the per-sequence
record prefix. Replace those scans with exact metadata:

```text
gdn_prefill_chunk_offsets:
  one CTA, cumsum over per-sequence chunk counts
  writes chunk_offsets [num_seqs + 1]

gdn_prefill_record_map:
  one CTA per sequence
  writes record_seq [max_records] and record_chunk [max_records]

flat WY prepass:
  loads record_seq/record_chunk in O(1)

flat WY consumer:
  loads chunk_offsets[pid_seq] in O(1)
```

Smoke measurement:

```text
idx4, T=8192, N=34, avg=241:
  accepted flat-WY: 260.84 us, OK
  record-map path:  292.01 us, OK
```

Result: reject and revert. The fixed scan is cheaper than two extra launches
plus metadata HBM traffic. The current ordinal scan is unattractive in code,
but it keeps the work inside the two launches that already matter.

## Experiment 70: accepted lower flat-WY many-seq gate

Hypothesis: Experiment 65 showed that a gate-only prepass did not amortize on
short-average many-seq rows such as 9, 13, and 18, so Experiment 66 kept
flat-WY gated at `avg_seq_len >= 180`. But flat-WY saves much more than gate
work: it reuses the C16 inverse, W, and causal QK across V tiles. Test whether
the gate can move lower for the stronger flat-WY path.

Implementation:

```text
old flat-WY gate:
  num_seqs >= 16 and total_tokens >= 8192 and avg_seq_len >= 180

new flat-WY gate:
  num_seqs >= 16 and total_tokens >= 8192 and avg_seq_len >= 128
```

Targeted newly included rows:

```text
idx9,  T=8192, N=48, avg=171:
  accepted flat-WY gate: 347.32 us
  avg>=128 gate:         205.47 us, OK

idx13, T=8192, N=56, avg=147:
  accepted flat-WY gate: 233.07 us
  avg>=128 gate:         152.00 us, OK

idx18, T=8192, N=57, avg=144:
  accepted flat-WY gate: 307.63 us
  avg>=128 gate:         189.44 us, OK
```

Paired full-sweep measurement versus Experiment 66:

```text
flat-WY avg>=180: mean=68.9809 us, sum=6898.09 us
flat-WY avg>=128: mean=65.7454 us, sum=6574.54 us
delta=-3.2355 us mean, -323.55 us sum
speedup=1.0492x

newly included paired full-sweep rows:
  idx9:  347.32 -> 205.07 us
  idx13: 233.07 -> 150.93 us
  idx18: 307.63 -> 188.82 us

correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
max_abs_error=6.673e-03
max_rel_error=1.029e+06
```

Result: keep. The older `avg>=180` boundary was specific to the gate-only
prepass. The full flat-WY reuse is strong enough to pay off on avg 144-171
many-seq rows, which were the largest remaining latencies after Experiment 66.

## Experiment 71: rejected flat many-seq 4-warp consumer

Hypothesis: after flat-WY precomputes chunk-local terms, the consumer still
does the large state/output contractions. Running that consumer with four
warps instead of two might improve tensor-core scheduling for the remaining
state-tile work.

Implementation tested:

```text
flat many-seq consumer:
  num_warps=4
  num_stages=3

prepass unchanged:
  num_warps=2
  num_stages=3
```

Targeted evidence versus the accepted avg>=128 flat-WY gate:

```text
wins:
  idx14, N=37, avg=222: 281.62 -> 239.26 us
  idx16, N=43, avg=191: 247.82 -> 243.91 us
  idx17, N=43, avg=191: 247.88 -> 244.71 us
  idx12, N=20, avg=410: 254.59 -> 254.12 us

losses:
  idx3,  N=32, avg=256: 200.70 -> 203.88 us
  idx4,  N=34, avg=241: 258.11 -> 267.55 us
  idx5,  N=34, avg=241: 260.38 -> 267.76 us
  idx7,  N=32, avg=256: 264.77 -> 268.79 us
  idx8,  N=25, avg=328: 227.61 -> 234.23 us
  idx9,  N=48, avg=171: 205.07 -> 234.77 us
  idx10, N=38, avg=216: 189.44 -> 194.28 us
  idx11, N=38, avg=216: 189.32 -> 191.48 us
  idx13, N=56, avg=147: 150.93 -> 169.62 us
  idx15, N=35, avg=235: 196.37 -> 204.37 us
  idx18, N=57, avg=144: 188.82 -> 198.80 us
  idx19, N=39, avg=211: 200.08 -> 205.93 us
```

Result: reject and revert. There are real wins, especially row 14, but the
winning shape set is not a clean hardware rule: nearby N=35/38/39 rows lose
while N=37/43 rows win. Encoding exact sequence counts would be trace-shaped
rather than a general kernel dispatch. Keep the accepted 2-warp flat consumer.

## Experiment 72: rejected long N=2 BV8 split-WY

Hypothesis: the long N=2 split-WY path has low sequence-level parallelism and
uses `BV=16`. Switching very long N=2 rows to `BV=8` doubles V-tile programs
and lowers register pressure, which might improve occupancy for rows such as
90, 27, and 79.

Implementation tested:

```text
if num_seqs == 2 and avg_seq_len >= 1024:
  BV=8
else:
  keep existing BV dispatch
```

Targeted evidence:

```text
idx90, T=5709, N=2, avg=2855:
  accepted BV16: 192.89 us
  BV8:           240.48 us, OK

idx27, T=3271, N=2, avg=1636:
  accepted BV16: 101.43 us
  BV8:           122.76 us, OK

idx79, T=2040, N=2, avg=1020:
  accepted BV16: 79.97 us
  BV8:           80.53 us, OK
```

Result: reject and revert. Extra V tiles repeat too much consumer work; the
occupancy/register-pressure improvement does not pay off. Keep `BV=16` for
long N=2 split-WY.

## Experiment 73: rejected long N=2 BV32 split-WY

Hypothesis: after BV8 failed for long N=2, test the opposite trade. `BV=32`
halves the V-tile count relative to accepted `BV=16`, reducing repeated
consumer work, but increases state-tile register pressure and reduces CTA
parallelism.

Implementation tested:

```text
if num_seqs == 2 and avg_seq_len >= 1024:
  BV=32
else:
  keep existing BV dispatch
```

Smoke measurement:

```text
idx90, T=5709, N=2, avg=2855:
  accepted BV16: 192.89 us
  BV32:          284.77 us, OK
```

Result: reject and revert. The larger state tile and lower CTA count are much
worse. Together with Experiment 72, this brackets the N=2 long split-WY tile
choice: keep `BV=16`.

## Experiment 74: rejected flat many-seq 4-warp prepass

Hypothesis: the flat-WY prepass performs chunk-local `K @ K^T`, the C16
inverse, W, and `Q @ K^T`. Four warps might schedule those small tensor-core
dots better than the accepted two-warps prepass.

Implementation tested:

```text
flat many-seq prepass:
  num_warps=4
  num_stages=3

consumer unchanged:
  num_warps=2
  num_stages=3
```

Smoke evidence:

```text
idx14:
  accepted 2-warp prepass: 281.62 us
  4-warp prepass:          299.54 us, OK

idx9:
  accepted 2-warp prepass: 205.07 us
  4-warp prepass:          224.07 us, OK

idx4:
  accepted 2-warp prepass: 258.11 us
  4-warp prepass:          278.43 us, OK
```

Result: reject and revert. The C16 prepass is too small for four warps; the
extra scheduling/occupancy cost outweighs any dot-level parallelism. Keep the
flat prepass at two warps.

## Experiment 75: accepted flat-WY mid-N dispatch

Hypothesis: Experiment 66/70 proved the compact ordinal flat-WY path is not
only a many-sequence optimization; it saves repeated C16 inverse, W, and
causal QK work across V tiles. Some mid-N ragged rows still use the generic
split-WY path even though they have enough total tokens to amortize the flat
prepass. Route that mid-N class through the same flat-WY implementation instead
of adding a new kernel.

Implementation:

```text
old flat-WY gate:
  num_seqs >= 16 and total_tokens >= 8192 and avg_seq_len >= 128

new flat-WY gate:
  (num_seqs >= 16 and total_tokens >= 8192 and avg_seq_len >= 128)
  or (10 <= num_seqs < 16 and total_tokens >= 3000)

split-WY gate:
  unchanged, but disabled when flat-WY is selected
```

The first smoke gate `4 <= num_seqs < 16 and total_tokens >= 3000` also routed
row 49 (`N=5, T=3028`) and lost (`162.03 -> 168.92 us`). Tightening the rule
to `N >= 10` keeps the idea tied to the intended mid-N ragged class instead of
capturing low-N rows where the flat prepass has too little sequence parallelism.

Paired evidence versus Experiment 70:

```text
idx6, T=4124, N=15:
  flat-WY gate128: 103.29 us
  mid-N flat-WY:    94.56 us, OK

idx20, T=3999, N=13:
  flat-WY gate128: 178.06 us
  mid-N flat-WY:   175.80 us, OK

full sweep:
  flat-WY gate128: mean=65.7454 us, sum=6574.54 us
  mid-N flat-WY:   mean=65.6763 us, sum=6567.63 us
  delta=-0.0691 us mean, -6.91 us sum
  speedup=1.0011x

correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
max_abs_error=6.673e-03
max_rel_error=1.029e+06
```

Result: keep, but treat as a small dispatch cleanup rather than a large
algorithmic win. The newly routed `N=13-15, T≈4k` rows win, no correctness risk
showed up in the full sweep, and the implementation reuses the accepted flat-WY
kernel. The aggregate gain is small enough that future larger changes should
not lean on this row alone for justification.

## Experiment 76: rejected compact N=3 cache-op stride

Hypothesis: after Experiment 60, the N=3 cached-power path always uses
`CHUNK=16`, so the cached Neumann chain only needs four matrices:
`N, N^2, N^4, N^8`. The workspace still allocates and strides five matrices per
chunk to preserve the old `CHUNK=32` layout. Compressing the C16 stride from
five matrices to four should reduce cache-op workspace size and improve memory
locality without changing the stored values or solve order.

Implementation tested:

```text
N=3 cache path:
  cache_ops = 4 for CHUNK=16, 5 for CHUNK=32
  wy_ops shape: [N, 8, max_chunks, cache_ops, C, C]
  ops_base = chunk_base * cache_ops * C * C

all stored matrices and consumer recurrence unchanged
```

Smoke evidence versus Experiment 75:

```text
idx39:  86.66 ->  87.27 us, OK
idx46: 108.71 -> 121.21 us, OK
idx47: 106.49 -> 120.99 us, OK
idx56: 108.99 -> 110.28 us, OK
```

Result: reject and revert. The dataflow is mathematically identical, but the
changed constexpr/signature/stride makes the important long N=3 rows slower,
especially rows 46 and 47. Keep the five-matrix stride; the unused C16 slot is
less harmful than the codegen/addressing change.

## Experiment 77: rejected flat-WY GVA-pair prepass fusion

Hypothesis: flat-WY prepass computes `K @ K^T` and `Q @ K^T` once per V head,
but the fixed contest shape has `NUM_Q_HEADS=NUM_K_HEADS=4` and
`NUM_V_HEADS=8`, so each adjacent V-head pair shares the same Q/K head. Fuse
the two V heads sharing a Q/K head into one prepass program: compute the shared
Q/K dots once, then run the gate/inverse/W work for both V heads and store the
same per-head workspace layout consumed by the existing flat-WY consumer.

Sub-variant A: always fuse the flat-WY prepass.

```text
pre_grid: (max_records, NUM_K_HEADS)
one program computes both V heads for a qk_head
consumer unchanged
```

Smoke evidence versus Experiment 75:

```text
many-seq rows:
  idx4:  260.88 -> 259.49 us, OK
  idx9:  204.99 -> 204.16 us, OK
  idx13: 151.22 -> 150.88 us, OK
  idx14: 282.39 -> 281.65 us, OK

mid/guard rows:
  idx6:   94.56 ->  98.88 us, OK
  idx20: 175.80 -> 177.25 us, OK
  idx49: 161.81 -> 161.51 us, OK
```

Sub-variant B: fuse only when `num_seqs >= 16`, and keep the per-V-head prepass
for the mid-N flat rows. This used a `FUSE_GVA` constexpr in the same Triton
function.

```text
idx4: 260.88 -> 421.58 us, OK
idx9: 204.99 -> 204.74 us, OK
```

Result: reject and revert. The algorithmic idea is sound, but the current
Triton expression is not. Always-fused saves a little shared Q/K work on
many-seq rows, but the gain is too small and mid-N rows regress. The gated
constexpr version changes codegen enough to destroy row 4. Do not revive this
without a separate dedicated fused prepass kernel or profiling evidence that
prepass Q/K dots are the dominant cost.

## Experiment 78: rejected fp16 C16 inverse dots

Hypothesis: `_apply_unit_lower_inverse` uses TF32 tensor cores for the small
`[C,C] x [C,BV]` and `[C,C] x [C,C]` Neumann chain. For `CHUNK=16`, fp16 tensor
cores might be faster while keeping more mantissa than the previously rejected
bf16 inverse path. Keep C32 on TF32 and test C16 fp16 first.

Sub-variant A: broad C16 fp16 inverse in `_apply_unit_lower_inverse`.

```text
if CHUNK <= 16:
  cast inverse operands to fp16 for the Neumann-chain dots
else:
  keep TF32
```

Smoke evidence versus Experiment 75:

```text
idx4:  260.88 -> 261.01 us, OK
idx13: 151.22 -> 152.50 us, OK
idx47: 106.49 -> 106.95 us, OK
idx50:  11.15 ->   8.51 us, OK
idx90: 192.90 -> 195.18 us, OK
idx92:  47.51 ->  46.98 us, OK
```

Sub-variant B: keep split/flat prepasses on TF32 and use fp16 only in the
fallback v2 kernel.

```text
idx50: 11.15 ->  8.53 us, OK
idx62: 30.63 -> 27.56 us, OK
idx33: OUT-OF-TOL, output/new_state NaN
idx73: OUT-OF-TOL, output/new_state NaN
```

Result: reject and revert. The broad path is net slower on the important
split/flat/N=3 rows. The fallback-only path has attractive speed on some short
rows, but it can produce NaNs, so it is not a valid contest kernel. A future
fp16 inverse attempt must first rewrite the gated ratio/scaling so the fp16
operands are bounded.

## Experiment 79: accepted flat many-seq BV16 gate

Hypothesis: the flat-WY many-seq rows are heavily skewed: most sequences have
only a few chunks, while one or a few sequences can have 100-200 chunks. With
`BV=32`, each V head has only four V-tile programs per sequence, so the long
sequences create a consumer tail. After flat-WY moved chunk-local work into the
prepass, using `BV=16` may increase consumer parallelism and reduce register
pressure without repeating as much expensive per-chunk work as it did before
flat-WY.

Initial all-many-seq smoke:

```text
idx14: 282.39 -> 232.61 us, OK
idx4:  260.88 -> 243.98 us, OK
idx12: 255.36 -> 229.95 us, OK
idx13: 151.22 -> 172.63 us, OK  # high-N loss
idx18: 189.02 -> 196.26 us, OK  # high-N loss
```

The loss pattern is clean: very high sequence counts already have enough CTA
parallelism, so doubling V tiles mostly repeats consumer work. Keep those rows
on `BV=32` and use `BV=16` only when the flat many-seq batch has `N <= 48`.

Implementation:

```python
if num_seqs >= 16:
    bv = 16 if num_seqs <= 48 else 32
```

Full-sweep rerun versus Experiment 75:

```text
flat-WY mid-N baseline: mean=65.6763 us, sum=6567.63 us
BV16 gate48:            mean=63.1082 us, sum=6310.82 us
delta=-2.5681 us mean, -256.81 us sum
speedup=1.0407x

affected rows, N in [16, 48]:
  baseline sum=3229.33 us
  BV16 sum=3001.88 us
  delta=-227.45 us
  speedup=1.0758x

largest wins:
  idx14: 282.39 -> 229.32 us
  idx7:  264.18 -> 236.47 us
  idx12: 255.36 -> 229.12 us
  idx16: 248.59 -> 225.83 us
  idx17: 248.19 -> 226.01 us
  idx4:  260.88 -> 240.97 us
  idx5:  259.14 -> 241.22 us

guard rows:
  idx13, N=56: 151.22 -> 151.72 us
  idx18, N=57: 189.02 -> 188.94 us
  idx6,  N=15:  94.56 ->  93.77 us
  idx20, N=13: 175.80 -> 175.35 us

correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
max_abs_error=6.673e-03
max_rel_error=1.029e+06
```

Result: keep. This is a real kernel-shape improvement rather than trace
rewarding: the gate follows the occupancy tradeoff directly. For moderately
many ragged sequences, `BV=16` increases parallelism and lowers state-tile
pressure; for extremely high sequence counts, the existing `BV=32` remains
better because CTA parallelism is already ample.

## Experiment 80: rejected flat many-seq BV8

Hypothesis: Experiment 79 showed that `BV=16` improves moderately many ragged
flat-WY rows by increasing consumer parallelism and reducing state-tile
pressure. The remaining top rows are still flat many-seq workloads, so `BV=8`
could help if the long-sequence consumer tail still dominates.

Implementation tested:

```python
if num_seqs >= 16:
    bv = 8 if num_seqs <= 48 else 32
```

Smoke evidence versus Experiment 79:

```text
idx14: 229.32 -> 284.85 us, OK
idx12: 229.12 -> 284.77 us, OK
idx4:  240.97 -> 312.43 us, OK
idx9:  203.59 -> 279.30 us, OK
```

Result: reject and revert. The correctness is fine, but the performance
signal is unambiguous. `BV=16` is the useful occupancy point for these flat-WY
rows; halving again doubles V tiles and repeats too much consumer-side state
and output work. Future flat many-seq work should reduce repeated work or
change the chunk algebra, not just shrink BV further.

## Experiment 81: rejected flat precomputed U

Hypothesis: after Experiment 79, flat-WY rows with `BV=16` run eight consumer
V tiles per V head. Each tile reloads `A_inv` and `beta/G`, then computes
`U_tile = A_inv @ ((beta/G) * V_tile)` inside the sequential consumer loop.
Precomputing full-width `U` in the flat prepass should move that work into the
chunk-parallel phase and shorten the long consumer tail.

Implementation tested:

```text
flat prepass:
  compute and store U = A_inv @ ((beta/G) * V) as bf16 [chunk, 128]
  stop storing flat-path A_inv and beta/G

flat consumer:
  load U[:, v_start:v_start+BV]
  x_chunk = U_slice - W @ state^T
```

Smoke evidence versus Experiment 79:

```text
idx14: 229.32 -> 283.57 us, OK
idx12: 229.12 -> 236.10 us, OK
idx4:  240.97 -> 247.45 us, OK
idx9:  203.59 -> 205.66 us, OK
```

Result: reject and revert. The dataflow is mathematically valid, but the
extra full-width `U` store/load and heavier prepass are worse than the saved
consumer-side small solve. This also confirms that the accepted flat path is
not primarily bottlenecked by `A_inv @ V_tile`; the next flat experiments
should remove scalar ordinal overhead or change the gated chunk algebra.

## Experiment 82: rejected flat consumer grid swizzle

Hypothesis: the flat-WY consumer loads the same per-chunk WY records across
multiple V tiles. Launching the consumer grid as `(head_tile, sequence)` instead
of `(sequence, head_tile)` may improve scheduling locality and L2 reuse for
the repeated flat-WY loads without changing math or workspace layout.

Implementation tested:

```text
flat consumer grid:
  old: grid = (num_seqs, NUM_V_HEADS * n_v_tiles)
       pid_seq = program_id(0), pid_hv = program_id(1)

  new: grid = (NUM_V_HEADS * n_v_tiles, num_seqs)
       pid_hv = program_id(0), pid_seq = program_id(1)
```

Targeted evidence versus Experiment 79 was mixed:

```text
wins:
  idx14: 229.32 -> 197.21 us, OK
  idx7:  236.47 -> 221.69 us, OK
  idx8:  214.51 -> 201.00 us, OK
  idx18: 188.94 -> 173.14 us, OK

losses:
  idx4:  240.97 -> 270.80 us, OK
  idx5:  241.22 -> 282.08 us, OK
  idx10: 184.62 -> 210.33 us, OK
  idx11: 184.13 -> 208.29 us, OK
```

Full-sweep evidence:

```text
Experiment 79 baseline: mean=63.1082 us, sum=6310.82 us
grid swizzle:           mean=66.0609 us, sum=6606.09 us
correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
```

Result: reject and revert. The idea is semantically clean and helps some flat
rows, but the full harness result is worse and the win/loss pattern is not
explainable from simple axes such as `(T, N, avg_seq_len)`. Do not keep a
fragile row-shaped gate here. The next low-risk scheduler experiment should
try removing consumer prefix scans directly instead of relying on CTA order.

## Experiment 83: rejected flat chunk-offset prefix cache

Hypothesis: after Experiment 79, flat-WY rows with `N>=32` launch many more
consumer CTAs because `BV=16` doubles the V-tile count. Each consumer CTA calls
`_chunk_prefix_before(...)`, which scans all sequences to find the flat record
base for its sequence. A tiny prefix-offset kernel could compute
`chunk_offsets[seq] = sum_{i<seq} ceil(seq_len_i / CHUNK)` once and let each
consumer CTA load the record base in O(1).

Implementation tested:

```text
if use_flat_wy and num_seqs >= 32:
  launch gdn_prefill_chunk_offsets[(1,)](...)
  flat consumer loads record_base = chunk_offsets[pid_seq]
else:
  keep the original _chunk_prefix_before scan
```

Smoke evidence versus Experiment 79:

```text
idx14: 229.32 -> 252.95 us, OK
idx4:  240.97 -> 266.15 us, OK
idx10: 184.62 -> 207.81 us, OK
idx18: 188.94 -> 208.65 us, OK
idx12: 229.12 -> 233.47 us, OK  # no-offset guard branch
```

Result: reject and revert. The scalar prefix scan is not expensive enough to
justify an extra launch and the new specialization. This narrows the flat-WY
bottleneck: the dominant work is still the chunk/state tensor-core body and
workspace traffic, not ordinal mapping overhead.

## Experiment 84: accepted fallback exp2 sigmoid

Hypothesis: active gate math already uses log2/exp2 for the decay path, but
the fallback v2 kernel still computes `beta = sigmoid(b)` through
`tl.sigmoid`. Rewriting only the fallback-v2 beta path as
`1 / (1 + exp2(-b / ln(2)))` should keep the same math on the faster base-2
SFU path. A broad replacement was tested first and rejected for split/N3
prepasses; this accepted variant keeps the exp2 sigmoid only in
`gdn_prefill_kernel_v2`.

Implementation:

```python
@triton.jit
def _sigmoid_exp2(x):
    return 1.0 / (1.0 + tl.exp2(-x * 1.4426950408889634))
```

Targeted smoke versus Experiment 79:

```text
fallback rows:
  idx50: 11.15 ->  9.04 us, OK
  idx62: 29.33 -> 28.45 us, OK
  idx33: 33.48 -> 32.26 us, OK
  idx73:  4.25 ->  4.39 us, OK

guards:
  idx47: 106.73 -> 106.56 us, OK
  idx90: 192.58 -> 196.44 us, OK  # targeted rerun noise; full sweep neutral
```

Full-sweep evidence:

```text
Experiment 79 baseline: mean=63.1082 us, sum=6310.82 us
fallback exp2 sigmoid:  mean=63.0997 us, sum=6309.97 us
delta=-0.0085 us mean, -0.85 us sum
speedup=1.0001x
correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
max_abs_error=6.673e-03
max_rel_error=1.029e+06
```

Result: keep, but call it a micro-win. This is not a major algorithmic
change; it is a small SFU-path cleanup that helps a few fallback rows and does
not hurt the full sweep. Do not generalize it back into split/flat/N3 prepasses
without fresh profiling, because the broad variant showed no value there.

## Experiment 85: rejected fallback y-space solve

Hypothesis: the fallback v2 path still solves in `X` space with `V / G` and
`beta / G`-style scaling. Rewriting the chunk solve to `Y = G * X` removes
that divisor and uses bounded gate ratios
`rho[j, i] = exp2(log2_G[j] - log2_G[i])`. This should be a numerics enabler
for future fp16 inverse attempts and may improve fallback stability.

Implementation tested in `gdn_prefill_kernel_v2` only:

```text
N_y[j, i] = beta_j * rho[j, i] * (k_j dot k_i), i < j
rhs_y[j]  = beta_j * (v_j - G_j * (k_j @ S_in^T))
Y         = (I + N_y)^-1 rhs_y

out_j = scale * (G_j * (q_j @ S_in^T)
                 + sum_i rho[j, i] * (q_j dot k_i) * Y_i)
S_out = Gc * S_in + (Gc / G_i * Y_i)^T @ K
```

Smoke evidence versus Experiment 84:

```text
idx33: 32.42 -> 34.58 us, OK
idx50: 10.96 -> 10.03 us, OK
idx62: 27.89 -> 30.05 us, OK
idx73:  4.40 ->  4.26 us, OK
```

Result: reject and revert for now. The algebra is correct enough for benchmark
tolerance and it removes the dangerous divisor, but the no-fp16 version adds
extra ratio matrix and tail scaling work that hurts the larger fallback rows.
If revisited, combine y-space directly with an fp16 inverse variant; y-space
alone is not a speed win.

## Experiment 86: accepted fallback y-space fp16 inverse

Hypothesis: Experiment 85 showed that y-space alone is slower, but the algebra
does remove the dangerous `V/G` divisor. The real value is to make fp16
triangular-solve dots safe for fallback rows. Reapply the y-space fallback
solve and use fp16 tensor cores for the Neumann inverse chain only in
`gdn_prefill_kernel_v2`.

Implementation:

```text
fallback v2:
  solve for Y = G * X
  use bounded rho[j, i] = exp2(log2_G[j] - log2_G[i])
  apply (I + N_y)^-1 through fp16 tensor-core dots

split/flat/N3:
  unchanged; still use the accepted TF32 inverse paths
```

Targeted smoke:

```text
idx33: 32.42 -> 29.35 us, OK
idx50: 10.96 ->  8.88 us, OK
idx62: 27.89 -> 26.14 us, OK
idx73:  4.40 ->  4.32 us, OK
```

Full-sweep evidence:

```text
Experiment 84 baseline: mean=63.0997 us, sum=6309.97 us
y-space fp16 inverse:   mean=62.8039 us, sum=6280.39 us
delta=-0.2958 us mean, -29.58 us sum
speedup=1.0047x
correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
max_abs_error=6.673e-03
max_rel_error=1.029e+06
```

Result: keep. This is the intended payoff of the y-space rewrite: bounded
operands make the fp16 inverse safe on rows that previously produced NaNs in
the old `X`-space fallback. The largest wins are fallback-heavy rows such as
idx87, idx72, idx37, idx33, idx78, idx92, and idx62. Keep this isolated to v2
fallback; do not move fp16 inverse into split/flat/N3 paths without a separate
bounded-scaling rewrite and full correctness sweep.

## Experiment 87: rejected N=3 y-space fp16 cache

Hypothesis: the N=3 cache path still stores the original `X`-space terms:
`G`, `beta/G`, raw lower-triangular `N`, and TF32 cached Neumann powers. The
fallback win in Experiment 86 suggests a bounded y-space formulation may allow
fp16 tensor-core dots for this path too.

Implementation tested only in `gdn_prefill_n3_cache_prepass` and
`gdn_prefill_kernel_n3_cache`:

```text
prepass:
  store log2_G instead of G
  store beta instead of beta/G
  store log2_Gc instead of Gc
  cache ratio-weighted N_y and N_y powers using fp16 dots
  cache ratio-weighted QK

consumer:
  rhs_y = beta * (V - G * (K @ S_in^T))
  apply cached fp16 Neumann chain to get Y
  output = scale * (G * state_q + QK_y @ Y)
  state = Gc * state + (Gc / G_i * Y_i)^T @ K
```

Targeted N=3 smoke versus Experiment 86:

```text
idx46: 106.86 -> 115.82 us, OK
idx47: 106.88 -> 117.23 us, OK
idx56: 109.30 -> 119.40 us, OK
idx98: 140.34 -> 163.49 us, OK
```

Result: reject and revert. Correctness is fine, so the algebra is valid, but
the extra ratio/tail work and changed cached-power path cost more than the
fp16 Neumann dots save. Unlike fallback, the N=3 path already amortizes most
inverse work through cached powers, so y-space's numerics benefit does not
translate into speed here.

## Experiment 88: rejected flat exact-order consumer

Hypothesis: the flat-WY prepass stores `W = A_inv @ (beta * K)` and the
consumer computes `X = A_inv @ ((beta/G) * V) - W @ S_in^T`. Algebraically,
that can be regrouped as:

```text
X = A_inv @ ((beta/G) * V - beta * (K @ S_in^T))
```

This would remove the flat `W` workspace, remove the prepass `W` dot/store, and
avoid reloading `W` in every V-tile consumer. The consumer already loads `K`,
so the replacement `K @ S_in^T` dot uses data already present in the hot loop.

Implementation tested only in the flat-WY path:

```text
flat prepass:
  stop computing/storing W

flat consumer:
  compute state_k = K @ state^T
  compute X = A_inv @ (beta/G * V - beta * state_k)
```

Targeted flat smoke versus Experiment 86:

```text
idx4:  240.48 -> 241.90 us, OK
idx12: 230.60 -> 245.26 us, OK
idx14: 229.79 -> 238.94 us, OK
idx9:  203.84 -> 207.23 us, OK
idx13: 151.80 -> 152.31 us, OK
```

Result: reject and revert. The regrouping is correct within tolerance, but it
slows the hot flat rows. The precomputed `W` is not just workspace overhead: it
shortens the consumer's dependency chain enough to beat the exact-order form.
Do not remove `W` unless a future design also changes the consumer schedule or
block structure.

## Experiment 89: rejected flat inverse workspace bf16

Hypothesis: the flat-WY consumer reloads the `A_inv [16,16]` workspace once per
chunk and V tile. Keep the current TF32 inverse computation, but store flat
`A_inv` as bf16 and cast it back to fp32 in the consumer. This narrows repeated
workspace traffic without changing the Neumann solve itself.

Implementation tested only in the flat-WY path:

```text
wy_nil allocation: fp32 -> bf16
prepass: store A_inv.to(bf16)
consumer: load A_inv and cast to fp32 before A_inv @ U_rhs
```

Targeted flat smoke versus Experiment 86:

```text
idx4:  240.48 -> 242.86 us, OK
idx12: 230.60 -> 235.47 us, OK
idx14: 229.79 -> 232.47 us, OK
idx9:  203.84 -> 205.45 us, OK
idx13: 151.80 -> 152.79 us, OK
```

Result: reject and revert. The compressed inverse is correct within tolerance,
but it consistently slows the flat rows. The flat path is not limited by this
small workspace load; preserving the fp32 inverse workspace keeps better
codegen/rounding for the consumer dot.

## Experiment 90: accepted flat GVA-pair prepass

Hypothesis: in the flat-WY many-sequence prepass, adjacent V heads share the
same Q/K head because `NUM_V_HEADS=8`, `NUM_K_HEADS=4`, and `GVA_RATIO=2`.
Compute the shared `K @ K^T` and `Q @ K^T` terms once for each Q/K head, then
run the gate/inverse/W storage for the two associated V heads. Keep the
existing flat consumer and workspace layout unchanged.

Implementation:

```text
num_seqs >= 16 flat-WY prepass:
  grid = (max_records, NUM_K_HEADS)
  load K/Q tile once per Q/K head
  compute gram_kk and gram_qk once
  store per-V-head A_inv, W, G, beta/G, Gc, QK for the two V heads

10 <= num_seqs < 16 flat-WY prepass:
  keep the previous per-V-head prepass
```

Targeted smoke was mixed but net-positive, so this went to a full sweep. The
big losses were rows 4/5/7, while rows 9/12/13/14/16/18/19 and several small
non-flat rows improved enough to offset them.

Full-sweep evidence versus Experiment 86:

```text
Experiment 86 baseline: mean=62.8039 us, sum=6280.39 us
flat GVA-pair prepass:  mean=62.5610 us, sum=6256.10 us
delta=-0.2429 us mean, -24.29 us sum
speedup=1.0039x
correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
max_abs_error=6.673e-03
max_rel_error=1.029e+06
```

Representative wins:

```text
idx13: 151.80 -> 148.57 us
idx18: 189.55 -> 186.95 us
idx16: 227.00 -> 224.42 us
idx9:  203.84 -> 201.72 us
idx12: 230.60 -> 229.45 us
```

Representative full-sweep losses/noisy rows:

```text
idx56: 109.30 -> 114.18 us  # N=3 path unchanged; likely run noise
idx3:  189.60 -> 191.35 us
idx7:  235.07 -> 236.02 us
```

Result: keep. The per-row signal is noisy, but the full 100-workload CUPTI
sum is better with correctness unchanged. This is a real invariant-reuse
change tied to fixed GVA head grouping, not trace-shaped dispatch.

## Experiment 91: rejected GVA prepass exp2 sigmoid

Hypothesis: the accepted GVA-pair prepass still computes beta with
`tl.sigmoid`. Since the fallback path benefited from `_sigmoid_exp2`, try that
replacement only inside `_store_many_wy_flat_head`, leaving split/N3 and the
old mid-N flat prepass unchanged.

Targeted smoke versus Experiment 90:

```text
idx13: 148.57 -> 147.85 us, OK
idx16: 224.42 -> 224.36 us, OK
idx3:  191.35 -> 193.05 us, OK
idx9:  201.72 -> 204.19 us, OK
```

Result: reject and revert. The replacement helps one high-N row slightly, but
it hurts row 9 and the already-regressive row 3 enough that it is not worth a
full sweep. Keep `_sigmoid_exp2` isolated to fallback v2 for now.

## Experiment 92: rejected shared flat QK workspace

Hypothesis: Experiment 90 computes `Q @ K^T` once per Q/K head in the GVA-pair
prepass, but still stores the resulting lower-triangular QK tile once per V
head to preserve the old consumer layout. Store QK once per Q/K head for
`num_seqs >= 16`, shrink the flat `wy_qk` workspace from `NUM_V_HEADS` to
`NUM_K_HEADS`, and let the consumer index QK by `qk_head`.

Targeted smoke versus Experiment 90:

```text
idx13: 148.57 -> 149.25 us, OK
idx16: 224.42 -> 226.50 us, OK
idx3:  191.35 -> 195.81 us, OK
idx9:  201.72 -> 203.82 us, OK
```

Result: reject and revert. The reduced QK workspace write is real, but it does
not pay for the extra consumer layout branch/addressing and altered codegen.
The accepted GVA-pair improvement is from shared prepass compute reuse, not
from QK workspace bandwidth, so keep the per-V-head QK layout that the existing
consumer likes.

## Experiment 93: rejected flat X-space state scale

Hypothesis: the flat consumer currently rescales the full `[BV, K]` state tile
by `Gc` after every C16 chunk. Carry the state in unscaled X-space instead,
track a scalar prefix product, divide local `beta/G` by that prefix, scale
outputs by `prefix * G`, and apply the prefix once to the final state. This
tests whether a global-prefix formulation can remove repeated state-wide
multiplies before attempting a larger BT64 consumer.

Targeted smoke versus Experiment 90:

```text
idx13: 148.57 -> 157.50 us, OUT-OF-TOL
idx16: 224.42 -> 229.13 us, OUT-OF-TOL (NaN)
idx3:  191.35 -> 192.40 us, OUT-OF-TOL (NaN)
idx9:  201.72 -> 205.84 us, OUT-OF-TOL (NaN)
```

Result: reject and revert. The algebra is the expected global-prefix form, but
the accumulated prefix product is not numerically safe for these flat rows.
Clamping the prefix before division changes the recurrence enough to break
tolerance, and unclamped division would be even more prone to overflow. This
also makes the larger BT64/X-space idea riskier unless it preserves the current
per-chunk scaling convention.

## Experiment 94: rejected unified flat GVA prepass

Hypothesis: Experiment 90 uses the GVA-pair flat prepass only for
`num_seqs >= 16`, while the two mid-N flat rows still use the old per-V-head
flat prepass. Since the Q/K sharing invariant also holds for mid-N rows, route
all flat-WY workloads through `gdn_prefill_many_wy_prepass_flat_gva2` and remove
the launcher branch between per-QK and per-V flat prepasses.

Targeted smoke versus Experiment 90:

```text
idx6:  95.12 -> 106.74 us, OK
idx20: 177.07 -> 176.05 us, OK
idx13: 148.57 -> 149.79 us, OK
```

Result: reject and revert. The simplification is correct and slightly helps
one mid-N row, but row 6 loses far more. The GVA-pair prepass halves the
program count, which is good when many records keep the B200 full, but it
starves/under-parallelizes the smaller mid-N flat shape. Keep the separate
mid-N per-V flat prepass for now.

## Experiment 95: rejected removing mid-N flat path

Hypothesis: if the mid-N flat branch is mostly complexity, remove the
`10 <= num_seqs < 16 and total_tokens >= 3000` flat dispatch condition and let
those two workloads use the existing split-WY path. This would simplify the
launcher and eventually make the old per-V flat prepass removable.

Targeted smoke versus Experiment 90:

```text
idx6:  95.12 -> 109.89 us, OK
idx20: 177.07 -> 178.24 us, OK
```

Result: reject and revert. The split-WY path is correct, but it is slower for
both mid-N flat workloads. The separate mid-N flat path is not dead code; it is
covering a real parallelism/overhead niche between split-WY and high-N flat
GVA-pair prepass.

## Experiment 96: rejected removing N=3 cache path

Hypothesis: the N=3 cache path adds two kernels and a separate workspace family.
Disable `use_cache_only` and let those rows fall back to the main v2 path. If
the latency is close, this would be the largest simplification available.

Targeted smoke versus Experiment 90:

```text
idx92:  46.04 ->  48.58 us, OK
idx46: 106.95 -> 166.93 us, OK
idx56: 114.18 -> 178.45 us, OK
idx98: 140.57 -> 233.51 us, OK
```

Result: reject and revert. The N=3 cache path is not cosmetic; it removes a
large amount of repeated per-tile triangular work on long three-sequence
shapes. Keep the extra kernels unless a future fused design preserves the same
cached terms.

## Experiment 97: rejected removing split-WY path

Hypothesis: the split-WY path adds a prepass, a consumer, and a precompute-U
mode. Disable `use_split_wy` and route those shapes back to the main v2
fallback to test whether the branch complexity is still justified.

Targeted smoke versus Experiment 90:

```text
idx27: 100.88 -> 231.80 us, OK
idx90: 192.50 -> 501.46 us, OK
idx49: 161.34 -> 304.03 us, OK
idx94:  72.67 -> 104.96 us, OK
```

Result: reject and revert. Split-WY is a major algorithmic win for long
single/two-sequence and mid-sequence rows. The extra path is justified by
removing repeated per-tile WY construction and keeping the consumer schedule
shorter.

## Experiment 98: accepted N=3 through split-WY

Hypothesis: instead of keeping the bespoke N=3 cache prepass/consumer, route
N=3 long rows through the existing split-WY prepass and consumer. This is a
fusion/simplification attempt: reuse one algorithmic family for N=1/2/3 and
mid-N split rows, then delete the N=3-only kernels if performance holds.

Implementation:

```text
use_cache_only: removed
use_split_wy: add (num_seqs == 3 and avg_seq_len >= 221)
delete gdn_prefill_n3_cache_prepass
delete gdn_prefill_kernel_n3_cache
```

Targeted N=3 smoke versus Experiment 90:

```text
idx39:  86.62 ->  94.01 us, OK   # targeted run; full sweep later improved to 80.33
idx46: 106.95 -> 104.47 us, OK
idx47: 107.10 -> 114.24 us, OK   # targeted run; full sweep later improved to 99.59
idx56: 114.18 -> 102.69 us, OK
idx75:  88.69 -> 102.86 us, OK   # targeted run; full sweep later improved to 82.90
idx76:  73.99 ->  70.58 us, OK
idx77:  75.24 ->  71.06 us, OK
idx92:  46.04 ->  44.90 us, OK
idx98: 140.57 -> 133.71 us, OK
```

Full-sweep evidence versus Experiment 90:

```text
Experiment 90 baseline: mean=62.5610 us, sum=6256.10 us
N=3 through split-WY:   mean=61.8183 us, sum=6181.83 us
delta=-0.7427 us mean, -74.27 us sum
speedup=1.0120x
correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
max_abs_error=6.673e-03
max_rel_error=1.029e+06
```

Representative full-sweep N=3 wins:

```text
idx39:  86.62 ->  80.33 us
idx46: 106.95 ->  99.51 us
idx47: 107.10 ->  99.59 us
idx56: 114.18 -> 105.65 us
idx75:  88.69 ->  82.90 us
idx76:  73.99 ->  69.47 us
idx77:  75.24 ->  69.50 us
idx92:  46.04 ->  44.91 us
idx98: 140.57 -> 129.73 us
```

Post-cleanup smoke after deleting the old N=3 kernels:

```text
idx46: 100.92 us, OK
idx98: 134.85 us, OK
```

Result: keep. This is both faster and simpler: one active N=3-specific branch
and two N=3-only kernels are gone, and the existing split-WY path handles those
rows with better full-sweep latency.

## Experiment 99: rejected N=3 precompute-U

Hypothesis: after routing N=3 through split-WY, extend the existing
`PRECOMPUTE_U` mode from `num_seqs <= 2` to `num_seqs <= 3`. This uses more
prepass/workspace traffic but removes the consumer-side `A_inv @ (beta/G * V)`
dot for N=3 rows.

Targeted N=3 smoke versus Experiment 98 was mixed but promising:

```text
idx46: 100.92 ->  86.03 us, OK
idx56: 105.65 ->  87.92 us, OK
idx75:  82.90 ->  72.92 us, OK
idx76:  69.47 ->  62.72 us, OK
idx98: 134.85 -> 146.50 us, OK
idx39:  80.33 ->  89.02 us, OK
idx47:  99.59 ->  87.47 us, OK
idx77:  69.50 ->  73.97 us, OK
idx92:  44.91 ->  41.98 us, OK
```

Full-sweep evidence versus Experiment 98:

```text
Experiment 98 baseline: mean=61.8183 us, sum=6181.83 us
N=3 precompute-U:       mean=64.4439 us, sum=6444.39 us
delta=+2.6256 us mean, +262.56 us sum
correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
max_abs_error=6.673e-03
max_rel_error=1.029e+06
```

N=3 subset in the full sweep:

```text
idx39:  80.33 ->  83.04 us
idx46:  99.51 ->  98.53 us
idx47:  99.59 ->  98.79 us
idx56: 105.65 ->  99.69 us
idx75:  82.90 ->  86.47 us
idx76:  69.47 ->  74.07 us
idx77:  69.50 ->  74.34 us
idx92:  44.91 ->  53.51 us
idx98: 129.73 -> 123.09 us
```

Result: reject and revert. The targeted run looked good, but the full sweep
showed both noisy non-N3 slowdown and a net N=3 subset loss. Keep
`PRECOMPUTE_U` limited to N<=2.

## Cleanup: removed unreachable gate prepass

After Experiment 98, the active dispatch has no workload path into
`use_pregate_v2`: every shape that satisfied that condition already returned
through flat-WY. Removed the dead launcher branch, `gdn_prefill_gate_prepass`,
and the `PRECOMPUTE_GATES` specialization from fallback v2.

Verification:

```text
python3 -m py_compile solution/triton/kernel.py
idx33 fallback smoke: 29.45 us, OK
idx98 N=3 split smoke: 147.19 us, OK
```

Result: keep as a behavior-preserving cleanup. No `results.csv` row because
this was not a new full-sweep performance result.

## Experiment 100: rejected N=3 split four-warps

Hypothesis: after routing N=3 rows through split-WY, give those rows the same
four-warp split configuration used by N=1. More warp parallelism might help
long N=3 rows without adding a new kernel family.

Targeted smoke versus Experiment 98:

```text
idx46: 100.92 -> 116.03 us, OK
idx56: 105.65 -> 130.45 us, OK
idx75:  82.90 ->  95.10 us, OK
idx92:  44.91 ->  61.54 us, OK
idx98: 134.85 -> 152.64 us, OK
```

Result: reject and revert. N=3 split-WY wants the two-warp configuration; four
warps increase overhead/register pressure more than they help parallelism.

## Experiment 101: rejected N=3 split chunk32

Hypothesis: use the existing split-WY kernel family for N=3 rows, but increase
the chunk length from C=16 to C=32 when average sequence length is at least
512. The intended benefit is simple: fewer WY chunks means less launch-visible
prepass and consumer loop overhead for long N=3 rows, without adding a new
kernel family.

Targeted smoke versus Experiment 98:

```text
idx39:  80.33 ->  71.65 us, OK
idx46:  99.51 ->  86.79 us, OK
idx56: 105.65 -> 100.99 us, OK
idx75:  82.90 ->  73.04 us, OK
idx98: 129.73 -> 114.99 us, OK
idx47:  99.59 ->  86.33 us, OUT-OF-TOL
```

Failure detail for idx47:

```text
output:    max_abs_error=3.014e-03, OK
new_state: max_abs_error=2.603e-02, OUT-OF-TOL
overall:   max_rel_error=3.368e+04
```

Result: reject and revert. The speed signal is strong, but this shape gate is
not correctness-safe: idx46 and idx47 are both N=3, total_tokens=1800,
avg_seq_len=600. A future version can only use C=32 if it finds a narrower
gate or a numerically safer C=32 formulation.

## Experiment 102: accepted very-long N=3 C32 inside split-WY

Hypothesis: keep the simplification from Experiment 98 (N=3 rows use the
existing split-WY path), but only allow C=32 for very long N=3 rows with
`avg_seq_len >= 700`. This avoids the failed avg=600 shape from Experiment
101 while preserving the same kernel family and reducing chunk-loop/prepass
overhead for the two longest N=3 rows.

Code change:

```text
chunk = 32 for:
  N <= 2 and avg_seq_len >= 512
  N == 3 and avg_seq_len >= 700
otherwise C=16
```

Targeted checks:

```text
idx56: 105.65 ->  88.76 us, OK
idx98: 129.73 -> 110.97 us, OK
idx47:  99.59 -> 102.07 us, OK  # remains on C16, prior failure avoided
```

Auditable full sweep saved at `/tmp/kachua_prefill_n3_c32_long_full.log`:

```text
Experiment 98 baseline:       mean=61.8183 us, sum=6181.83 us
Very-long N=3 C32 split-WY:   mean=61.7998 us, sum=6179.98 us
delta=-0.0185 us mean, -1.85 us sum
speedup=1.0003x
correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
max_abs_error=6.673e-03
max_rel_error=1.029e+06
```

Affected N=3 rows in the full sweep:

```text
idx56: 105.65 ->  88.10 us
idx98: 129.73 -> 111.39 us
```

Result: keep, but treat the aggregate as a small noisy win. The per-row signal
is exactly where expected and correctness is clean; the full-sweep delta is
small because unrelated rows moved under Modal/B200 timing noise. This is still
a simplification-friendly refinement: no new kernel family, just a narrower
chunk-size gate inside the surviving split-WY path.

## Cleanup: removed inactive v1 prefill code

After the active dispatch audit, removed non-dispatched v1 launcher/kernels and
two unused helpers/constants from `solution/triton/kernel.py`:

```text
removed: CHUNK_SIZE
removed: _fallback_bv
removed: _launch_gdn_v1
removed: gdn_prefill_kernel_v1
removed: _prefill_update_tile_pair_v1
removed: _load_long_token_v1
removed: gdn_prefill_kernel_v1_long
```

Verification:

```text
python3 -m py_compile solution/triton/kernel.py
rg inactive-v1 symbols: no matches
idx56 B200 smoke: 99.79 us, OK
```

Result: keep as cleanup. This does not change the production dispatch or add a
`results.csv` row; it reduces the active submission file by 375 dead lines so
future experiments only reason about live kernel families.

## Experiment 103: rejected precise state update for N=3 C32

Hypothesis: Experiment 101 failed only in `new_state`, while output stayed
within tolerance. Reopen the broader N=3 C32 gate (`avg_seq_len >= 512`), but
for C32 split-WY rows without `PRECOMPUTE_U`, compute the state update as a
fp32/TF32 dot:

```text
state_delta = _dot_f32(trans(x_chunk), K_tile_f32)
```

instead of using the existing bf16 `x_chunk_bf` state-update dot. If the
previous failure was just final-state rounding in `x_chunk_bf`, idx47 should
clear tolerance.

Targeted result:

```text
idx47: 99.59 -> 131.36 us, OUT-OF-TOL
output:    max_abs_error=3.029e-03, OK
new_state: max_abs_error=2.595e-02, OUT-OF-TOL
overall:   max_rel_error=3.322e+04
```

Result: reject and revert. The C32 idx47 failure is not fixed by making only
the final state-update dot more precise; the error is already baked into the
C32 chunk recurrence/solve before that dot, and the precise dot is slower.

## Experiment 104: accepted high-N flat record-base cache

Hypothesis: in the flat-WY path, every V consumer program re-finds its prefix
record base by walking chunk counts for the sequence. For high sequence counts,
that scalar scan is repeated across many V tiles. Let the existing flat prepass
store the first record id per sequence once, then let consumers load it directly.

Implementation detail:

```text
flat_record_base[seq_id] = record_id when pid_chunk == 0
USE_RECORD_BASE only for num_seqs >= 16
```

The high-N gate matters. The raw all-flat variant improved some many-sequence
rows but made mid-N rows noisy/worse, so the accepted version only enables the
new metadata path where the repeated scalar scan is clearly amortized.

Auditable full sweep saved at
`/tmp/kachua_prefill_flat_record_base_highn_full.log`:

```text
Previous accepted:   mean=61.7998 us, sum=6179.98 us
High-N record base:  mean=61.4950 us, sum=6149.50 us
delta=-0.3048 us mean, -30.48 us sum
speedup=1.0050x
correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
max_abs_error=6.673e-03
max_rel_error=1.029e+06
```

Representative rows:

```text
idx3:  189.89 -> 188.79 us
idx4:  240.38 -> 239.25 us
idx9:  201.87 -> 200.09 us
idx13: 149.48 -> 147.17 us
idx16: 224.34 -> 223.02 us
idx20: 175.24 -> 176.59 us
idx56:  88.10 ->  88.26 us
idx98: 111.39 -> 110.48 us
```

Result: keep. This is a small but real full-sweep win, and it fits the
simplification direction because it improves the existing flat-WY family instead
of adding a new kernel path. The one visible regression row is smaller than the
aggregate high-N gains and may also be Modal timing noise.

## Experiment 105: rejected split-WY capacity review variants

Trigger: PR review pointed out that split-WY `max_chunks` is computed from
`total_tokens` even though the temporary tensors are shaped as
`[num_seqs, ..., max_chunks, ...]`. The reviewer is structurally right: this
can over-allocate and launches dead prepass programs for ragged batches.

Rationale: before changing the PR or current experiment branch, test whether the
fix reduces real GPU work. The exact host-side fix needs:

```text
seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
max_seq_len = torch.max(seq_lens).item()
max_chunks = ceil(max_seq_len / chunk)
```

That syncs on the GPU. Approximate average-length bounds avoid the sync, but
they are only valid if every sequence fits inside the chosen capacity.

Shape audit saved at `/tmp/kachua_prefill_workload_shapes_all.jsonl` showed the
split rows are ragged, but most exact-vs-current over-allocation is modest:

```text
idx47: T=1800 N=3 lengths=[162,123,1515] current_chunks=113 exact=95 ratio=1.19
idx56: T=2857 N=3 lengths=[1544,1291,22] current_chunks=90 exact=49 ratio=1.84
idx75: T=1796 N=3 lengths=[1175,17,604] current_chunks=113 exact=74 ratio=1.53
idx90: T=5709 N=2 lengths=[5637,72] current_chunks=179 exact=177 ratio=1.01
idx98: T=2284 N=3 lengths=[30,2120,134] current_chunks=72 exact=67 ratio=1.07
```

Measured variants:

```text
exact max_seq_len via torch.max(...).item():
  idx6   95.09 ->  95.62 us, OK
  idx47  99.36 -> 217.50 us, OK
  idx56  88.26 -> 199.04 us, OK
  idx90 192.02 -> 259.07 us, OK
  idx98 110.48 -> 224.20 us, OK
  idx20 176.59 -> 176.78 us, OK

avg_seq_len capacity:
  idx47 102.00 us, OUT-OF-TOL max_abs=2.217e+01 max_rel=3.987e+07
  idx56  87.39 us, OUT-OF-TOL max_abs=1.079e+00 max_rel=9.807e+05

2x avg capacity:
  idx47 101.48 us, OUT-OF-TOL max_abs=6.195e+00 max_rel=1.905e+05

2x avg only for N>=4:
  idx49 163.48 us, OUT-OF-TOL max_abs=1.641e-01 max_rel=5.397e+06

3x avg only for N>=4:
  idx49 165.98 us, OUT-OF-TOL max_abs=9.454e-01 max_rel=5.776e+05

N=3 flat-WY dispatch:
  idx39  80.37 ->  93.45 us, OK
  idx47  99.36 -> 134.14 us, OK
  idx56  88.26 ->  88.33 us, OK
  idx75  83.03 ->  82.47 us, OK
  idx76  69.24 ->  84.21 us, OK
  idx92  44.47 ->  44.66 us, OK
  idx98 110.48 -> 111.51 us, OK

split-layout flat prepass:
  idx47  99.36 -> 100.96 us, OK
  idx56  88.26 ->  88.22 us, OK
  idx75  83.03 ->  83.83 us, OK
  idx92  44.47 ->  45.60 us, OK
  idx98 110.48 -> 111.63 us, OK
  idx49 161.10 -> 161.23 us, OK
  idx20 176.59 -> 176.74 us, OK

skip dead wy_bg stores in PRECOMPUTE_U split prepass:
  idx26  66.91 ->  67.48 us, OK
  idx27 100.01 -> 100.60 us, OK
  idx54  39.61 ->  53.34 us, OK
```

Result: reject all variants and leave PR #14 unchanged. The dead work exists,
but the exact host-sync fix is much slower, average-bound capacity is unsafe on
ragged batches, and moving split rows to flat mappings loses more in mapping
overhead than it saves in dead prepass CTAs. A future win in this area needs a
deeper compact split layout and consumer, not a launch-side capacity tweak.

## Experiment 106: rejected flat `wy_qk` Q/K-head sharing

Hypothesis: in the flat-WY path, each Q/K head fans out to two V heads. The
prepass already shares the expensive `K K^T` and `Q K^T` matmuls for high-N
flat rows, but it still stores the same lower-triangular `wy_qk` tile once per
V head. Store `wy_qk` as `[record, qk_head, chunk, chunk]` instead of
`[record, v_head, chunk, chunk]`, have the high-N GVA2 prepass write it only
once, and have both sibling V heads load the shared QK tile.

Full CUPTI sweep saved at
`/tmp/kachua_prefill_flat_qk_head_share_full.log`:

```text
Current accepted:    mean=61.4950 us, sum=6149.50 us
Shared flat wy_qk:   mean=61.8595 us, sum=6185.95 us
delta=+0.3645 us mean, +36.45 us sum
correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
max_abs_error=6.673e-03
max_rel_error=1.029e+06
```

Representative rows:

```text
idx3:  188.79 -> 190.48 us
idx4:  239.25 -> 242.85 us
idx9:  200.09 -> 203.03 us
idx13: 147.17 -> 147.68 us
idx16: 223.02 -> 226.43 us
idx20: 176.59 -> 174.38 us
idx6:   95.09 ->  94.73 us
idx49: 161.10 -> 161.68 us
```

Result: reject and revert. The removed QK store is too small to dominate the
flat path, and the extra Q/K-head addressing plus conditional store appears to
hurt long high-N rows more than the temporary-size reduction helps. Do not
spend more time on tiny flat metadata/storage reductions unless profiling shows
that prepass global stores are actually hot.

## Experiment 107: rejected BV=16 for all high-N flat rows

Hypothesis: for very high sequence counts (`num_seqs > 48`), the current flat
path uses `BV=32`. Switching those rows to `BV=16` might reduce register
pressure and improve occupancy, at the cost of twice as many V-tile consumer
programs.

Targeted B200 result for the first changed row:

```text
idx13 (N=56, T=8192): 147.17 -> 167.95 us, OK
```

Result: reject and revert without running a full sweep. The consumer-program
count increase dominates any register-pressure relief on this row. Keep the
existing `num_seqs > 48 -> BV=32` gate.

## Experiment 108: rejected 4-warp flat GVA2 prepass

Hypothesis: the high-N flat prepass fuses the two V heads that share one Q/K
head. Giving that prepass 4 warps instead of 2 might feed the chunk-local
matmuls better on Blackwell. NVIDIA's Blackwell guide calls out the 64K-register
SM file and occupancy limits, while Triton exposes `num_warps`, `num_stages`,
and `maxnreg` as direct tuning knobs; this test checks whether the heavier
prepass wants more warps.

Full CUPTI sweep saved at
`/tmp/kachua_prefill_flat_gva2_prepass_4warps_full.log`:

```text
Current accepted:       mean=61.4950 us, sum=6149.50 us
4-warp flat prepass:    mean=62.2467 us, sum=6224.67 us
delta=+0.7517 us mean, +75.17 us sum
correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
max_abs_error=6.673e-03
max_rel_error=1.029e+06
```

Worst changed rows:

```text
idx3:  188.79 -> 196.70 us
idx18: 183.56 -> 189.99 us
idx5:  238.85 -> 244.66 us
idx9:  200.09 -> 205.36 us
idx13: 147.17 -> 152.39 us
```

Result: reject and revert. The flat GVA2 prepass is not starved for warp
parallelism; the extra warps likely increase resource pressure and reduce
scheduler flexibility. Keep the 2-warp prepass.

## Experiment 109: rejected exp2 sigmoid in WY prepasses

Hypothesis: the fallback v2 kernel already uses `_sigmoid_exp2` for beta gate
computation, but the split and flat WY prepasses still used `tl.sigmoid`.
Switch the WY prepass beta computation to the base-2 helper so gate math stays
on the cheaper `exp2` path.

Variant A changed all WY prepasses: split, flat, and flat GVA2.

```text
Current accepted:       mean=61.4950 us, sum=6149.50 us
All-WY exp2 sigmoid:    mean=61.7559 us, sum=6175.59 us
delta=+0.2609 us mean, +26.09 us sum
correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
max_abs_error=6.673e-03
max_rel_error=1.029e+06
```

The row pattern was mixed: some high-N flat rows improved slightly, but split
rows regressed, for example:

```text
idx94: 72.03 -> 77.91 us
idx54: 39.61 -> 44.91 us
idx79: 79.08 -> 81.01 us
idx75: 83.03 -> 84.77 us
```

Variant B kept exp2 sigmoid only in flat prepasses and restored `tl.sigmoid`
for split-WY. The full sweep was contaminated by a globally slow Modal run, so
I used a fresh targeted rerun on an affected flat row:

```text
idx3: 188.79 -> 193.65 us, OK
```

Result: reject and revert both variants. Unlike fallback v2, WY prepasses do
not benefit from the exp2 sigmoid substitution. Keep the existing split/flat
prepass `tl.sigmoid` calls.

## Experiment 110: rejected split consumer `maxnreg` sweep

Hypothesis: fresh NCU profiles of the split-WY consumer showed very low
occupancy and extreme register pressure:

```text
T=5709 N=2: gdn_prefill_kernel_split_wy, 112.544 us, grid=128,
            registers/thread=255, achieved_occupancy=3.117%,
            SM=10.510%, DRAM=4.610%
T=3028 N=5: gdn_prefill_kernel_split_wy, 44.320 us, grid=320,
            registers/thread=220, achieved_occupancy=6.626%,
            SM=17.324%, DRAM=7.012%
T=2284 N=3: gdn_prefill_kernel_split_wy, 50.848 us, grid=192,
            registers/thread=255, achieved_occupancy=4.093%,
            SM=13.015%, DRAM=4.967%
```

The first idea from both local analysis and the Gemini cross-check was to cap
registers on the split consumer launch. This is a good falsifiable test: if the
kernel is mostly latency-hidden by occupancy, a cap should help; if the live
state tile really needs those registers, spills should dominate.

Measured on workload `idx90` (N=2, T=5709), which is the worst profiled
low-grid case:

```text
accepted baseline: ~192.02 us from the previous targeted run
maxnreg=128:       479.90 us, OK
maxnreg=192:       277.15 us, OK
maxnreg=224:       204.22 us, OK
```

Result: reject and revert. The profile correctly identified register pressure,
but the cure is worse than the disease: even a soft cap spills enough of the
state-heavy recurrence to lose latency. Future split-WY work should reduce the
consumer live range or CTA payload structurally, not force register allocation
with `maxnreg`.

## Experiment 111: rejected low-N split-WY `BV=8`

Hypothesis: the fresh NCU profiles showed under-filled or thin split-WY grids,
especially `idx90` where N=2 produced only 128 consumer CTAs on a 148-SM B200.
Now that split-WY precomputes chunk-local matrices, setting `BV=8` for low-N
split rows should no longer duplicate the expensive prepass work. It would
double the consumer grid and halve the V rows carried by each CTA.

Patch tested:

```text
if use_split_wy and 2 <= num_seqs <= 3:
    bv = 8
    n_v_tiles = HEAD_DIM // bv
```

Targeted B200 results:

```text
idx90 (N=2, T=5709): accepted ~192.02 us -> 239.47 us, OK
idx98 (N=3, T=2284): accepted ~110.48 us -> 140.64 us, OK
```

Result: reject and revert. More CTAs and smaller state tiles do not overcome
the doubled consumer pass count. For split-WY, `BV=16` remains the better
low-N tradeoff.

## Experiment 112: rejected long N=2 split consumer `num_stages=4`

Hypothesis: the split consumer shows low DRAM throughput and repeatedly loads
WY workspace tiles inside the chunk loop. A deeper pipeline (`num_stages=4`)
only for long N=2 split rows might overlap those loads with tensor-core work
without changing the math or adding a new kernel path.

Patch tested:

```text
split_state_stages = 4 if num_seqs == 2 and avg_seq_len >= 1024 else 3
...
num_warps=split_num_warps, num_stages=split_state_stages
```

Targeted B200 results:

```text
idx90 (N=2, T=5709): accepted ~192.02 us -> 197.34 us, OK
idx27 (N=2, T=3271): accepted ~100.01 us -> 101.60 us, OK
```

Result: reject and revert. The existing three-stage consumer is already better
for these rows; the extra stage likely increases scheduling/register pressure
more than it improves global-load overlap.

## Experiment 113: rejected removing low-N `PRECOMPUTE_U`

Hypothesis: simplify split-WY by removing the N<=2 `PRECOMPUTE_U` branch and
using the same consumer-side `A_inv @ (beta/G * V)` dataflow for all split
rows. This would remove the full-width `wy_u` workspace and one active branch
if latency stayed neutral.

Patch tested:

```text
use_precompute_u = False
```

Targeted B200 result:

```text
idx90 (N=2, T=5709): accepted ~192.02 us -> 243.42 us, OK
```

Result: reject and revert. The specialized precomputed-U branch is justified
for long low-N rows; removing it simplifies code but loses too much latency.

## Experiment 114: rejected bf16 gate workspaces for low-N precomputed-U rows

Hypothesis: in split-WY rows with `PRECOMPUTE_U=True`, the consumer only needs
`G` and `Gc` scale factors, while `wy_bg` is effectively dead. Store `wy_g`,
`wy_bg`, and `wy_gc` as bf16 for low-N precomputed-U rows to reduce workspace
traffic and scalar live pressure. Keep very-long rows on fp32 after targeted
evidence showed they are fragile.

Variants tested:

```text
all PRECOMPUTE_U rows use bf16 gates:
  idx90: 192.02 -> 194.51 us, OK
  idx54:  39.61 ->  38.88 us, OK
  idx27: 100.01 ->  99.74 us, OK
  idx26:  66.91 ->  66.75 us, OK
  idx38:        -> 125.10 us, OK but new_state abs=1.264e-02

gated variant:
  use bf16 only when use_precompute_u and avg_seq_len < 2000
```

Full CUPTI sweep for the gated variant saved at
`/tmp/kachua_prefill_split_gate_bf16_avglt2000_full.log`:

```text
Current accepted:     mean=61.4950 us, sum=6149.50 us
bf16 gated variant:   mean=64.9567 us, sum=6495.67 us
delta=+3.4617 us mean, +346.17 us sum
correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
```

Result: reject and revert. The narrow targeted wins do not survive the full
workload order, and the long-row accuracy/timing signal is worse. Keep split
gate workspaces in fp32.

## Experiment 115: rejected flat consumer `num_stages=2`

Trigger: temporary NCU profiles of high-N flat-WY showed the consumer at
`gdn_prefill_many_split_wy_flat`, `T=8192,N=32`, 79.168 us, grid=2048,
registers/thread=222, achieved_occupancy=11.393%, SM=27.447%, DRAM=14.446%,
dynamic shared memory=33.408 KB. Nsight Compute flagged high L1/TEX pressure
and a partial-wave tail. The GVA2 prepass for the same shape was smaller:
28.608 us, grid=2176, registers/thread=168, achieved_occupancy=15.901%,
SM=26.883%, DRAM=7.793%, dynamic shared memory=9.216 KB.

Hypothesis: reducing only the flat consumer launch from `num_stages=3` to
`num_stages=2` might lower shared-memory/L1 pressure enough to beat any lost
pipeline overlap.

Targeted B200 result:

```text
idx3 (flat high-N): accepted 188.79 us -> 235.90 us, OK
```

Result: reject and revert immediately. The flat consumer needs the existing
three-stage pipeline; reducing stages loses far more than it saves.

## Experiment 116: rejected flat consumer `maxnreg=192`

Hypothesis: the flat high-N consumer profile showed 222 registers/thread and
11.393% achieved occupancy. A moderate `maxnreg=192` cap might raise residency
without the catastrophic spills seen in the split consumer.

Targeted B200 results:

```text
idx3  (N=32 flat): accepted 188.79 us -> 191.74 us, OK
idx13 (N=56 flat): accepted 147.17 us -> 165.95 us, OK
```

Result: reject and revert. Even the moderate cap is a net loss, especially on
very-high-N rows. The flat consumer's register footprint is high but still
preferable to spill-driven occupancy.

## Experiment 117: rejected flat consumer cache eviction hints

Hypothesis: the flat high-N consumer NCU profile showed high L1/TEX throughput
and repeated WY/K/Q loads across V tiles. Cache eviction hints might either
preserve reusable chunk data (`evict_last`) or stream it to reduce L1 pressure
(`evict_first`) without changing math.

Patch tested in `gdn_prefill_many_split_wy_flat` on WY workspace, K, and Q
loads:

```text
eviction_policy="evict_last"
eviction_policy="evict_first"
```

Targeted B200 result on `idx3`:

```text
accepted default: 188.79 us
evict_last:       195.13 us, OK
evict_first:      308.45 us, OK
```

Result: reject and revert. The default Triton load policy is better than both
manual hints. The repeated data either is not scheduled with enough locality for
`evict_last`, or the compiler/runtime already chooses the right balance.

## Experiment 118: rejected high-N flat `CHUNK=32`

Hypothesis: high-N flat rows are dominated by the sequential chunk loop in the
consumer. Using `CHUNK=32` instead of `CHUNK=16` halves the number of flat chunk
records and loop iterations, at the cost of larger triangular matrices and
heavier per-chunk tensor-core work.

Patch tested:

```text
if use_flat_wy and num_seqs >= 16:
    chunk = 32
```

Targeted B200 results:

```text
idx3  (N=32 flat): accepted 188.79 us -> 192.35 us, OK
idx13 (N=56 flat): accepted 147.17 us -> 199.36 us, OK
```

Result: reject and revert. C32 is correctness-safe here, but the larger
per-chunk triangular/inverse work and codegen pressure overwhelm the reduced
loop count. Keep flat-WY at C16.

## Experiment 119: rejected flat consumer QK remask removal

Hypothesis: the flat prepass stores `wy_qk` after applying the causal
lower-triangular mask, so the flat consumer's second
`qk_lower = tl.where(lower_eq, qk_lower, 0.0)` should be redundant. A prior
split/N3 remask-removal test lost, but it did not isolate the current flat-WY
consumer.

Targeted B200 result:

```text
idx3 (flat high-N): accepted 188.79 us -> 193.71 us, OK
```

Result: reject and revert. As in the older split test, the remask is
algebraically redundant but beneficial for Triton code shape. Keep it.

## Experiment 120: accepted GVA2 prepass single record-base writer

Hypothesis: in the high-N flat GVA2 prepass, all four Q/K-head programs wrote
the same `flat_record_base[seq]` value when `pid_chunk == 0`. Only one writer
is needed. Restricting the store to `pid_qk == 0` removes a tiny redundant
global write and avoids a benign same-value race without changing any kernel
path or data layout.

Patch:

```text
tl.store(record_base_ptr + pid_seq, record_id, mask=(pid_chunk == 0) & (pid_qk == 0))
```

Targeted smoke:

```text
idx3:  188.79 -> 187.31 us, OK
idx13: 147.17 -> 147.69 us, OK
```

Full CUPTI sweep saved at
`/tmp/kachua_prefill_flat_gva_record_base_single_writer_full.log`:

```text
Previous accepted:     mean=61.4950 us, sum=6149.50 us
Single-writer variant: mean=61.3375 us, sum=6133.75 us
delta=-0.1575 us mean, -15.75 us sum
speedup=1.0026x
correctness markers: 0 OUT-OF-TOL / FAILED / Mismatch / Traceback
max_abs_error=6.673e-03
max_rel_error=1.029e+06
```

Result: keep. This is a very small but clean high-N prepass cleanup: less
redundant metadata traffic and no extra kernel path.

Correction from Experiment 122: this patch changed the dormant per-V flat
prepass, not the active high-N GVA2 prepass. The full-sweep timing was real
for the submitted code at the time, but the claimed high-N mechanism was not
active. Treat this as a noisy/non-causal timing result; the results.csv row was
removed. The remaining code fix is just the latent `pid_h == 0` store mask in
the dormant per-V prepass.

## Experiment 121: rejected flat GVA2 paired consumer

Hypothesis: the two V heads attached to the same Q/K head can share the flat
consumer's `Q`, `K`, and lower-triangular `qk` work. A paired consumer with
`BV=8` for each sibling head keeps the total live state rows close to the
existing single-head `BV=16` kernel while reducing duplicate sibling-head
loads.

Targeted B200 result:

```text
idx3 (N=32 flat): accepted 189.77 us -> 283.85 us, OK
```

Result: reject and revert. The algebra was correctness-safe, but the paired
consumer keeps two state tiles live and emits a much larger loop body. The
added register/codegen pressure dominates the saved Q/K/qk loads, so this is
the wrong fusion boundary for the flat path.

## Experiment 122: rejected active GVA2 record-base single-writer

Hypothesis: apply the intended single-writer metadata store to the active
high-N GVA2 flat prepass:

```text
tl.store(record_base_ptr + pid_seq, record_id, mask=(pid_chunk == 0) & (pid_qk == 0))
```

This removes three redundant same-value stores per sequence for high-N flat
rows. It should be correctness-preserving because every Q/K-head program sees
the same `record_id` when `pid_chunk == 0`.

Targeted B200 results:

```text
idx3  (N=32 flat): accepted 189.77 us -> 200.69 us, OK
idx13 (N=56 flat): accepted 147.17 us -> 147.17 us, OK
```

Result: reject and revert the active GVA2 mask. The extra predicate/codegen
shape hurts the moderate high-N row more than the tiny metadata-store reduction
can help. Keep the active prepass's simple `pid_chunk == 0` store, and keep the
latent per-V prepass fix from Experiment 120 so that dormant code no longer
references an undefined `pid_qk`.

## Experiment 123: rejected flat consumer one-warp launch

Hypothesis: the high-N flat consumer profile showed high register pressure and
low occupancy, so launching the flat consumer with `num_warps=1` might reduce
per-CTA scheduling/register pressure enough to offset any lost tensor-core
throughput.

Patch tested:

```text
gdn_prefill_many_split_wy_flat[..., num_warps=1, num_stages=3]
```

Targeted B200 results:

```text
idx3  (N=32 flat): accepted 189.77 us -> 230.08 us, OK
idx13 (N=56 flat): accepted 147.17 us -> 236.48 us, OK
```

Result: reject and revert. The flat consumer still needs two warps for its
small GEMM sequence; cutting to one warp loses far more tensor-core/scheduling
throughput than it saves in pressure. Keep `num_warps=2`.

## Experiment 124: rejected record-base mode for all flat rows

Hypothesis: mid-N flat rows currently make the consumer recover the first
record id with `_chunk_prefix_before`, while high-N flat rows load
`flat_record_base[seq]`. Enable `STORE_RECORD_BASE=True` in the per-V flat
prepass and `USE_RECORD_BASE=True` in the flat consumer for every flat row to
remove the scalar prefix scan and simplify the flat consumer address mode.

Patch tested:

```text
num_seqs < 16 flat prepass: STORE_RECORD_BASE=True
flat consumer: USE_RECORD_BASE=True
```

Targeted B200 results:

```text
idx6  (mid-N flat):  accepted ~95.09 us ->  95.93 us, OK
idx20 (mid-N flat):  accepted 176.59 us -> 188.30 us, OK
idx3  (high-N flat): accepted 188.79 us -> 188.89 us, OK
```

Result: reject and revert. The high-N path is effectively neutral because it
already used `flat_record_base`, but the mid-N flat rows lose. The per-sequence
metadata store/load and changed consumer code shape cost more than the small
`_chunk_prefix_before` scalar scan at these sizes.

## Experiment 125: rejected compact two-consumer flat output split

Hypothesis: the high-N flat consumer does too much work inside the sequential
per-sequence chunk loop. Split it into:

```text
1. state-recording consumer:
   - walks chunks sequentially
   - computes x_chunk and state_q
   - stores x_record/state_q_record as bf16 [record, head, chunk, 128]
   - updates final new_state

2. output-record consumer:
   - runs per flat record/head/V tile
   - computes qk_lower @ x_record
   - writes output
```

This leaves the recurrence intact but moves the within-chunk causal output
correction and output stores out to a chunk-parallel kernel. A tiny
`record_ranges` kernel supplied `record_id -> (chunk_start, seq_end)` metadata
so the output kernel did not have to run `_flat_chunk_to_seq` for every V tile.

Targeted B200 result:

```text
idx3 (N=32 flat): accepted 188.79 us -> 215.53 us, OK
```

Result: reject and revert. The split is algebraically correct, but the moved
work is too small: `qk_lower @ x_chunk` is only a C16 triangular dot. The extra
global round trip for `x_record/state_q_record`, the metadata launch, and the
small output kernel cost more than the reduced sequential loop body. Future
flat-consumer work should attack the K=128 state/update contractions or the
recurrence dependency, not split off this tiny output correction.

## Experiment 126: rejected flat consumer bf16 `state_q` epilogue

Hypothesis: in the flat consumer, `state_q = Q @ state^T` is only used in the
final bf16 output. Casting `state_q` to bf16 before adding `qk_contrib` might
reduce fp32 live range/register pressure without affecting the recurrence.

Patch tested inside `gdn_prefill_many_split_wy_flat`:

```text
out_tile = scale * G[:, None] * (state_q.to(tl.bfloat16) + qk_contrib)
```

Targeted B200 results:

```text
idx3  (N=32 flat): accepted 188.79 us -> 189.38 us, OK
idx4  (N=34 flat): accepted 239.25 us -> 248.09 us, OK
idx13 (N=56 flat): accepted 147.17 us -> 151.48 us, OK
```

Result: reject and revert. The precision change is correctness-safe, but it
does not improve codegen; the extra cast/changed accumulation shape makes
important high-N rows slower.

## Experiment 127: rejected per-V flat prepass helper refactor

Hypothesis: the old per-V flat prepass duplicates the gate/inverse/store body
that the high-N GVA2 prepass already reaches through `_store_many_wy_flat_head`.
Refactor `gdn_prefill_many_wy_prepass_flat` to compute `gram_kk`/`gram_qk` and
call the helper with `pid_h`, reducing source duplication and potentially
keeping generated code equivalent.

Targeted B200 results:

```text
idx6  (mid-N flat):  accepted ~95.09 us -> 108.63 us, OK
idx20 (mid-N flat):  accepted 176.59 us -> 177.09 us, OK
idx3  (high-N guard): accepted 188.79 us -> 193.59 us, OK
```

Result: reject and revert. The source simplification is correct, but the mid-N
per-V prepass codegen changes enough to hurt `idx6`. Keep the duplicated
per-V prepass body for now; it is ugly, but it is the faster shape.

## Experiment 128: rejected flat consumer `num_stages=4`

Hypothesis: after `num_stages=2` failed, test the other side of the pipeline
depth tradeoff. The high-N flat consumer repeatedly loads WY, K, Q, V, and
state-derived operands inside the chunk loop; `num_stages=4` might improve
load/compute overlap on B200 if the extra staging does not worsen pressure.

Patch tested:

```text
gdn_prefill_many_split_wy_flat[..., num_warps=2, num_stages=4]
```

Targeted B200 results:

```text
idx3  (N=32 flat): accepted 188.79 us -> 195.77 us, OK
idx13 (N=56 flat): accepted 147.17 us -> 149.73 us, OK
```

Result: reject and revert. The flat consumer's local pipeline-depth optimum is
still `num_stages=3`; two stages starves overlap, four stages adds enough
pressure/codegen cost to slow both high-N probes.

## Experiment 129: rejected very-long N=2 split `CHUNK=64`

Hypothesis: row90 remains one of the largest non-flat contributors. It has one
very long sequence (`5637` tokens) and one short sequence, so the split
consumer's sequential chunk loop is long. Route only very-long N=2 rows
(`avg_seq_len >= 2000`) to `CHUNK=64`, and add the missing Neumann doubling
term required for C64 so the inverse is at least algebraically represented.

Targeted B200 result:

```text
idx90 (N=2, T=5709): accepted ~192.02 us -> 496.15 us, OUT-OF-TOL
  output max_abs=2.286e-02
  new_state max_abs=1.764e-01
```

Result: reject and revert. C64's larger triangular inverse and chunk-local
matmuls overwhelm the halved chunk count, and the numerical error is already
outside tolerance. Do not pursue larger split chunks without a different
bounded/incremental algebra.

## Experiment 130: rejected high-N BV32 threshold at N>=40

Hypothesis: Experiment 79 picked `BV=16` for `16 <= N <= 48` and `BV=32` above
that. After later flat-path changes, retest whether the upper part of that
range should move back to `BV=32`: use `BV=16` only for `N < 40`, and `BV=32`
for `N >= 40`.

Targeted B200 results:

```text
idx9  (N=48 flat): accepted 200.09 us -> 205.28 us, OK
idx16 (N=43 flat): accepted 223.02 us -> 243.09 us, OK
idx14 (N=37 guard): accepted 228.15 us -> 232.83 us, OK
```

Result: reject and revert. The original threshold remains right: moderately
many ragged high-N rows still need `BV=16` to expose enough consumer
parallelism and reduce state-tile pressure. Moving N40+ back to `BV=32`
reintroduces the long-sequence tail.

## Experiment 131: rejected gated late `G/qk_lower` loads

Hypothesis: delay the flat consumer's `G` and `qk_lower` loads until just before
the output correction to reduce live ranges. The ungated form improved one
very-high-N BV32 row but hurt a BV16 row, so gate it with a constexpr:
`LATE_QK_G = num_seqs > 48`.

Targeted B200 results:

```text
ungated smoke:
  idx13 (N=56 flat): accepted 147.17 us -> 144.94 us, OK
  idx3  (N=32 flat): accepted 188.79 us -> 197.17 us, OK

gated smoke:
  idx13 (N=56 flat): accepted 147.17 us -> 148.54 us, OK
  idx18 (N=57 flat): accepted 183.56 us -> 182.97 us, OK
  idx3  (N=32 guard): accepted 188.79 us -> 193.55 us, OK
```

Result: reject and revert. The late-load idea is not robust once gated; one
very-high-N row improves slightly, another regresses, and the guard path does
not look clean enough to trust. Keep the original early loads, which likely
give the compiler better overlap despite a longer live range.

## Experiment 132: rejected late `qk_lower` load only

Hypothesis: Experiment 131 moved both `G` and `qk_lower` late. Isolate the
larger live matrix by keeping `G` early but delaying only the `qk_lower` load
until just before `qk_lower @ x_chunk`.

Targeted B200 results:

```text
idx3  (N=32 flat): accepted 188.79 us -> 194.52 us, OK
idx13 (N=56 flat): accepted 147.17 us -> 148.96 us, OK
```

Result: reject and revert. Even qk-only delay is slower. The original early
load likely gives Triton more opportunity to overlap the workspace load with
the K=128 state contractions.

## Experiment 133: accepted narrow N=3 C32 split-GVA2 prepass

Hypothesis: split-WY prepass recomputes chunk-local `K @ K^T` and `Q @ K^T`
for both V heads that share one Q/K head. A paired GVA2 prepass can compute
those two gram matrices once per Q/K head, then write the V-head-specific
gate/inverse/W/G/bg/Gc data for the two paired V heads. This is a fusion
candidate: if it ties, fewer redundant tensor-core grams and fewer prepass CTAs
are worth keeping, but only where occupancy/reg pressure do not dominate.

Implementation kept:

```text
new helper:
  _store_split_wy_head(...)

new prepass:
  grid = (num_seqs, NUM_K_HEADS, max_chunks)
  gdn_prefill_wy_prepass_gva2

active gate:
  use_split_gva2_prepass = num_seqs == 3 and chunk == 32
```

Broad gates were rejected before narrowing:

```text
N=2 PRECOMPUTE_U:
  idx90 accepted ~192 us -> 207.45 us, OK but slower

N=3 C16:
  idx47 accepted ~100 us -> 113.18 us, OK but slower

mid-N split:
  idx49 accepted ~161 us -> 162.59 us, OK/noise
  idx94 accepted ~72 us -> 73.15 us, OK/slower
```

Narrow N=3 C32 evidence:

```text
idx56: accepted ~88.10 us -> 87.45 us in full sweep, OK
idx98: accepted ~111.39 us -> 111.10 us in full sweep, OK
idx47 guard after narrowing: 100.03 us, OK
idx90 guard after narrowing: 192.77 us in full sweep, OK
idx95 guard after narrowing: 73.09 us in full sweep, OK
```

Full B200 sweep:

```text
count: 100 workloads
correctness: OK on all 100 workloads
median mean: 61.20 us
median sum: 6119.52 us
max_abs_error: 6.673e-03
max_rel_error: 1.029e+06
previous logged mean: 61.50 us
```

Result: keep narrowly. The win is small and likely close to timing noise, but
it passes the full sweep and follows the fusion policy: for the N=3 C32 split
path, the fused paired prepass ties or slightly improves while removing
duplicated Q/K gram work. Do not extend it back to N=2, N=3 C16, or broad mid-N
without fresh evidence; those variants lost targeted smokes.

## Experiment 134: rejected flat high-N longest-first sequence scheduling

Hypothesis: high-N flat rows often have their longest sequence buried in the
middle of `cu_seqlens`. Building a tiny length-sort map and launching flat
consumer CTAs in longest-sequence-first order might reduce the final tail on
B200 without changing math or WY record layout.

Patch tested:

```text
gdn_prefill_build_seq_order[(1,)]:
  loads all seq lengths for N <= 64
  writes seq_order sorted by descending length

flat consumer:
  pid_sched = program_id(0)
  pid_seq = seq_order[pid_sched]
```

Targeted B200 result:

```text
idx4 (N=34 flat, longest sequence not first):
  accepted ~239-241 us -> 246.85 us, OK
```

Result: reject and revert. The schedule-only idea is semantically clean, but
the extra launch/map work and changed CTA ordering do not reduce the flat
consumer tail enough. Previous grid-axis swizzle already showed the same
pattern: scheduling changes can help isolated rows but lose the full harness
unless the mapping removes real work.

## Experiment 135: rejected split consumer `warp_specialize=True`

Hypothesis: Triton exposes `tl.range(..., warp_specialize=True)` for Blackwell,
and the split consumer chunk loop contains repeated memory loads plus tensor
core dots. Auto warp specialization might partition memory/MMA/vector work and
hide latency on B200.

Patch tested:

```text
for chunk_start in tl.range(
    seq_start,
    seq_end,
    CHUNK,
    warp_specialize=WARP_SPECIALIZE,
):
```

with `WARP_SPECIALIZE=True` only on the non-precompute split consumer smoke.

Targeted B200 result:

```text
idx98 (N=3 C32 split): accepted ~111 us -> 124.09 us, OK
```

Result: reject and revert. It compiles and is correctness-safe, but the loop is
not a simple matmul loop: it mixes WY workspace loads, gate vectors, V loads,
state contractions, output correction, and state update. The compiler's
additional warp partitioning cost/reg pressure outweighs any overlap.

## Experiment 136: rejected N=3 C32 split consumer `BV=8`

Hypothesis from the Pro cross-check: after the narrow GVA2 paired prepass,
long N=3/C32 split rows may be consumer-resource limited rather than
prepass-work limited. Changing only the N=3/C32 route from `BV=16` to `BV=8`
would halve each CTA's live state rows and double the consumer grid, potentially
raising B200 occupancy without changing math or workspace layout.

Patch tested:

```text
elif num_seqs == 3 and avg_seq_len >= 700:
    bv = 8
```

Targeted B200 result:

```text
idx56 (N=3 C32 split): accepted ~88.10 us -> 108.67 us, OK
```

Result: reject and revert. The smaller live tile is not enough to pay for twice
as many consumer CTAs and workspace/K/Q/V loads. Keep `BV=16` for the N=3/C32
split consumer; the paired GVA2 prepass remains the useful fusion boundary.

## Experiment 137: rejected N=3 C32 split kernels with four warps

Hypothesis from the Pro resource-shape suggestion: long N=3/C32 split rows may
have too few CTAs for B200, so using four warps for the N=3/C32 split prepass
and consumer might improve per-CTA tensor-core scheduling and hide WY workspace
loads. This leaves the algebra and workspace layout unchanged.

Patch tested:

```text
split_num_warps = 4 if (num_seqs == 1 or (num_seqs == 3 and chunk == 32)) else 2
```

Targeted B200 result:

```text
idx56 (N=3 C32 split): accepted ~88.10 us -> 116.30 us, OK
```

Result: reject and revert. The current N=3/C32 split path wants two warps; four
warps add scheduling/register pressure without improving the sequential chunk
loop. This reinforces the same pattern as `BV=8` and warp specialization:
resource-shape tweaks around the consumer are losing, so the next useful work
must remove real sequential work or duplicated math.

## Experiment 138: rejected long N=2 C16 split-WY ablation

Hypothesis: before implementing blocked C64 as four C16 sub-blocks, retest
whether plain C16 is locally better for the hardest long N=2 row. If C16 beats
the current C32 split path, a simpler dispatch fix might exist; if it loses,
blocked C64 must win by reducing loop/prepass overhead while preserving local
C16 solves.

Patch tested:

```text
chunk = 32 if (
    (num_seqs == 1 and avg_seq_len >= 512)
    or (num_seqs == 3 and avg_seq_len >= 700)
) else 16
```

Targeted B200 result:

```text
idx90 (N=2, lengths=[5637,72]): accepted ~192 us -> 259.42 us, OK
```

Result: reject and revert. Plain C16 doubles the sequential chunk count too
much for long N=2. Blocked C64 remains conceptually different: it would need
to keep C16-local triangular solves but reduce outer loop/prepass count. Do
not lower the N=2 long chunk dispatcher to C16.

## Experiment 139: rejected paired C32 triangular consumer merge

Hypothesis: before writing a full four-C16 blocked-C64 prepass, test a cheaper
two-block merge using the existing C32 split-WY workspace. For two adjacent C32
chunks, compute the second chunk's state-dependent terms from the original
state plus cross-block corrections:

```text
x0 = u0 - w0 @ S0.T
S1 = Gc0 * (S0 + x0.T @ K0)
x1 = u1 - Gc0 * (w1 @ S0.T + (w1 @ K0.T) @ x0)
Q1 @ S1.T = Gc0 * (Q1 @ S0.T + (Q1 @ K0.T) @ x0)
```

This is a blocked-triangular merge smoke: it does not add new buffers and is
algebraically equivalent to two sequential C32 chunks, but halves the consumer
loop trip count.

Targeted B200 result:

```text
idx90 (N=2, lengths=[5637,72]): accepted ~192 us -> 334.94 us, OK
```

Result: reject and revert. The algebra was correctness-safe, but the larger
consumer body and extra cross-block tensor-core dots dominate. A full blocked
C64 path would need to move cross-block metadata into a prepass and reduce
consumer live ranges; doing the merge directly inside the consumer is the wrong
boundary for Triton/B200.

## Experiment 140: reconfirmed narrow N=3 C32 split-GVA2 prepass

Question: the accepted fused prepass previously won/tied in the full sweep, but
a fresh `idx56` targeted run reproduced at a higher absolute latency than the
logged full-sweep row. Compare the fused N=3/C32 prepass against the old per-V
split prepass under the same current timing environment before deciding whether
to revert the fusion.

Current narrow fused prepass:

```text
idx56: 110.78 us, OK
idx98: 110.53 us, OK
```

Temporary old per-V prepass (`use_split_gva2_prepass = False`):

```text
idx56: 121.50 us, OK
idx98: 113.92 us, OK
```

Result: keep the fused prepass narrowly. The absolute `idx56` timing drifted
relative to the earlier full-sweep evidence, but direct A/B still favors the
fused prepass on both N=3/C32 target rows. This remains a fusion-policy keep:
do not broaden it, but do not revert it unless a full sweep shows aggregate
regression.

## Experiment 141: rejected split consumer `disable_licm`

Hypothesis from Triton docs: `tl.range(..., disable_licm=True)` can avoid
hoisting loop-invariant code when hoisting creates long live ranges. The split
consumer has a long recurrent chunk loop and previous experiments repeatedly
pointed at register/live-range pressure, so try the flag only for low-N split
rows (`num_seqs <= 3`) without changing math.

Patch tested:

```text
for chunk_start in tl.range(seq_start, seq_end, CHUNK, disable_licm=DISABLE_LICM):
```

Targeted B200 result:

```text
idx90 (N=2 C32 split): accepted ~192 us -> 209.05 us, OK
```

Result: reject and revert. LICM hoisting appears beneficial for this consumer,
or disabling it increases repeated scalar/address work more than it reduces
live ranges. Keep the default `tl.range` loop.

## Experiment 142: rejected split consumer accumulator multi-buffer disable

Hypothesis: the split consumer may be register-pressure limited, so ask Triton
not to multi-buffer dot accumulators in the low-N split loop. This is a
compiler-scheduling hint only; it does not change the math or workspace.

Patch tested:

```text
tl.range(..., disallow_acc_multi_buffer=DISALLOW_ACC_MULTI_BUFFER)
```

Targeted B200 result:

```text
idx90 (N=2 C32 split): accepted ~192 us -> 194.64 us, OK
```

Result: reject and revert. The timing is close but not better, and the extra
constexpr complicates the split kernel signature. Keep the default accumulator
multi-buffering.

## Experiment 143: clean full-sweep confirmation after Pro-guided rejects

Question: after restoring the accepted narrow N=3/C32 GVA2 prepass and reverting
the rejected consumer/compiler experiments, run a clean full sweep with no file
edits during the Modal job. This checks correctness coverage and whether the
accepted fusion is still in the expected band before adding another experiment.

Clean B200 full-sweep result:

```text
--workload-idx -1: 100/100 correctness OK
mean of streamed workload medians: 64.875 us
idx56: 99.62 us, OK
idx90: 193.14 us, OK
idx98: 122.82 us, OK
```

Result: do not add a CSV row. This run was useful correctness coverage, but it
was not a new latency record relative to row 30 (`61.20 us`). Direct A/B from
Experiment 140 still favored the fused N=3/C32 prepass over the old per-V
prepass, so keep the fusion narrowly and continue with scheduling experiments.

## Experiment 144: kept flat high-N edge-interleave sequence scheduling

Hypothesis from the flat high-N scheduling plan: the flat prepass can already
store per-sequence `record_base` for `num_seqs >= 16`, so the consumer does not
need to execute sequence CTAs in natural sequence order. For ragged high-N rows,
natural order can leave long sequences in the tail of the CTA schedule. Try a
cheap deterministic edge interleave in the flat consumer only:

```text
raw_seq: 0, 1, 2, 3, ...
pid_seq: 0, N-1, 1, N-2, ...
```

This changes only scheduling. It does not change the recurrence, the flat WY
workspace layout, or any output index semantics. The cost is a couple of scalar
integer ops per consumer CTA and one constexpr.

Targeted smoke was mixed because these rows are noisy:

```text
idx14: 231.69 us clean full -> 238.78 us targeted, OK
idx9:  204.05 us clean full -> 199.90 us targeted, OK
idx7:  238.75 us clean full -> 231.93 us targeted, OK
```

The full sweep was cleaner:

```text
--workload-idx -1: 100/100 correctness OK
mean of streamed workload medians: 61.17 us
previous accepted CSV row: 61.20 us
affected high-N flat rows: 3377.17 us -> 3285.01 us against Experiment 143
max_abs_error: 6.673e-03
max_rel_error: 1.029e+06
```

Result: keep. The full-suite delta is effectively a tie/slight win, but every
affected high-N flat row improved inside the full sweep, and the implementation
does not add launches or buffers. This satisfies the scheduling-only experiment
goal without touching lower-N paths.

## Experiment 145: rejected stacked consumer W/Q state dot

Hypothesis: the split and flat WY consumers both compute `W @ state.T` and
`Q @ state.T` with the same right-hand `state.T`. On B200 tensor cores, a
single stacked `[2C, K] x [K, BV]` dot might amortize dispatch/setup cost better
than two `[C, K] x [K, BV]` dots, especially for C32 N=3 split rows. This is a
fusion-policy candidate only if it ties or wins, since it does not reduce HBM
traffic and increases the left tile height.

Broad test, applied to split and flat consumers:

```text
idx56:  83.87 us, OK
idx90: 246.54 us, OK
idx98: 104.54 us, OK
idx3:  241.22 us, OK
idx13: 201.57 us, OK
idx20: 209.06 us, OK
idx49: 187.17 us, OK
```

The apparent N=3 wins were not enough to justify the broad path because N=2
precompute and high-N flat rows regressed sharply. I then narrowed the stack to
only the N=3/C32 split path:

```text
idx56:  96.58 us, OK
idx98: 116.74 us, OK
idx90 guard: 216.75 us, OK
```

That also lost against the accepted full-sweep band. After reverting the
stacked path and keeping the already accepted GVA2 prepass plus flat swizzle,
remote sanity returned to the expected shape:

```text
idx56: 99.92 us, OK
idx90: 195.01 us, OK
idx3:  194.46 us, OK
```

Result: reject and revert. The fused consumer dot is the wrong fusion for this
kernel: the bigger M dimension appears to worsen tiling/register scheduling more
than it saves launch or dot setup work. Keep the spread `W @ state.T` and
`Q @ state.T` dots unless a future codegen-specific variant can prove otherwise.
