# Prefill Experiment Log

## 2026-03-29

- Added a first Triton prefill kernel based on the existing decode tile shape.
- Kept the execution model simple: one program owns one `(sequence, value-head, value-tile)`
  slice and loops over the tokens in that sequence.
- Chose correctness and submission isolation over aggressive parallel scan work for v1.
- Verification is currently limited to static checks in this worktree because local `torch`
  is unavailable and no CUDA runtime is attached to this branch checkout.

### Modal profile: v1 baseline

Rationale:

- before changing launch width or kernel structure, measure the current prefill submission on
  the representative dataset shapes
- profile matters more than benchmark here because it isolates the kernel, while the benchmark
  harness currently gives no incremental progress output

Commands:

```bash
uv run modal run gdn_prefill_qk4_v8_d128_k_last/scripts/profile_modal.py --ncu-set basic
uv run modal run gdn_prefill_qk4_v8_d128_k_last/scripts/run_modal.py
```

Notes:

- the subdir `scripts/run_modal.py` and `scripts/profile_modal.py` are currently byte-for-byte
  synced with the root copies
- local static check passed:

```bash
python -m py_compile gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py
```

Profile summary:

| Shape | Duration | Block | Grid | Regs/thread | Occupancy % | SM % | DRAM % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `t6_n1_c2` | `11.712 us` | `256` | `128` | `32` | `12.49` | `8.33` | `0.62` |
| `t134_n1_c2` | `151.040 us` | `256` | `128` | `32` | `12.57` | `13.89` | `0.09` |
| `t82_n3_c4` | `36.160 us` | `256` | `384` | `32` | `29.14` | `36.08` | `0.69` |
| `t401_n4_c5` | `285.184 us` | `256` | `512` | `32` | `35.83` | `21.93` | `0.17` |
| `t4124_n15_c16` | `1.626048 ms` | `256` | `1920` | `32` | `53.13` | `44.08` | `0.20` |
| `t8192_n20_c21` | `3.586560 ms` | `256` | `2560` | `32` | `35.44` | `29.51` | `0.17` |
| `t8192_n32_c33` | `2.902592 ms` | `256` | `4096` | `32` | `49.21` | `47.26` | `0.25` |
| `t8192_n57_c58` | `2.653440 ms` | `256` | `7296` | `32` | `61.16` | `49.26` | `0.39` |

Interpretation:

- the kernel is using a fixed `256`-thread launch across all prefill shapes
- that launch is likely too wide for the smallest sequence-count cases; Nsight repeatedly flags
  underfilled hardware on the low-grid-count shapes
- even on the multi-millisecond shapes, DRAM throughput stays very low, which suggests the
  kernel is latency/dependency limited rather than bandwidth limited
- the simple per-sequence sequential loop is probably correctness-safe, but it is not obviously
  exploiting the hardware well yet

Benchmark note:

- the Modal benchmark packed successfully as `gdn-prefill-triton-v1`
- after `Packed:` it remained silent in the observation window, so there is no reliable
  benchmark summary to record yet
- treat the benchmark result as operationally inconclusive for now

Suggested next step:

- keep this as the measured prefill baseline
- investigate a narrower launch policy first, because the fixed `num_warps=8` shape is the
  most obvious mismatch exposed by the profile
