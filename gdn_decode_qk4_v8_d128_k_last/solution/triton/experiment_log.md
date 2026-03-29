# Triton Decode Experiment Log

## 2026-03-29 — Warp-1 launch experiment

### Change

- file: `solution/triton/kernel.py`
- change: `num_warps=8 -> num_warps=1`
- everything else left unchanged

### Why

The earlier Triton-vs-CUDA profile showed that Triton already had much higher
occupancy and much higher SM throughput than CUDA, yet it was still slower at
`b16` and `b64`. That suggested the old Triton launch was overspending on a
small decode tile instead of lacking parallelism.

### Profile commands

```bash
uv run modal run scripts/profile_modal.py --decode-batches 1,16,64 --ncu-set basic
```

### Profile result

| Batch | Old Triton us | New Triton us | Old/New |
| --- | ---: | ---: | ---: |
| 1 | 4.544 | 5.088 | 0.893x |
| 16 | 8.768 | 7.264 | 1.207x |
| 64 | 21.120 | 13.376 | 1.579x |

### Interpretation

- `b1` got slightly worse
- `b16` got meaningfully better
- `b64` got much better
- new Triton is now within about `10-14%` of CUDA at `b16` and `b64`

The warp-1 launch should become the new baseline for the next Triton decode
iteration.

### Benchmark command

```bash
uv run modal run scripts/run_modal.py
```

### Benchmark note

The benchmark passed all decode workloads. Representative large-batch workload
`eaf0a285...` ran at `60.526 us` with `636.72x` speedup, `abs_err=1.91e-06`,
and `rel_err=3.19e-02`.

For this experiment, the profile result is more important than the benchmark
result because the profile isolates kernel behavior while the benchmark includes
wrapper and harness overhead.

## Experiment 2: CUDA-shaped Triton rewrite (discarded)

### Rationale

The warp-1 launch closed most of the gap to CUDA without touching the math. The
next idea was to make the Triton inner structure look more like the CUDA kernel:

- explicit `32 x 4` K slicing
- compute `kq` once per program
- process V rows in pairs
- keep the warp-1 launch

This seemed worth testing because the remaining gap looked more like per-tile
work inefficiency than launch-width mismatch.

### Files changed

- `solution/triton/kernel.py`

### Profile commands

```bash
uv run modal run scripts/profile_modal.py --decode-batches 1,16,64 --ncu-set basic
```

### Profile result

| Batch | Warp-1 us | Rewrite us | Warp-1/Rewrite |
| --- | ---: | ---: | ---: |
| 1 | 5.088 | 6.752 | 0.754x |
| 16 | 7.264 | 7.840 | 0.926x |
| 64 | 13.376 | 13.280 | 1.007x |

### Interpretation

- `b1` regressed badly
- `b16` regressed modestly
- `b64` improved only marginally, about `0.7%`
- the rewrite is not a better baseline than the simpler warp-1 kernel

The likely issue is that the more CUDA-shaped Triton indexing added instruction
and bookkeeping cost without recovering enough useful memory or reduction
efficiency to pay for it.

### Conclusion

Do not promote this rewrite over the warp-1 baseline.
