# Kernel.cu

We need to read:

1. State (KxV)
2. Key (1xK) and Value (1xV) vectors
3. Query (1xK)
4. Scalars (a, dt_bias, A_log, scale)

and write:

1. Updated state (KxV)
2. Output (1xK)

## Benchmark Summary (Modal B200)

| Kernel | Median Speedup | Median Latency | Per-run Medians | Range |
|--------|---------------|----------------|-----------------|-------|
| v1     | 75x           | 15 µs          | 75x, 82x       | 66-88x |
| v2     | 82x           | 10 µs          | 78x, 84x       | 72-94x |
| v3     | 93x           | 12 µs          | 91x, 97x, 94x  | 86-110x |

Modal has ~20% run-to-run variance due to different B200 instances.

## Kernel 1

- Across number of tokens (B*T)
- T == 1
- Each block:
  - Loads a single tile of state
  - Updates a single tile of state
  - Writes a single row of output
- Design choices:
  - Processing 2D tile of state
  - Stride over K dimension inside the kernel

Results

```bash
gdn_decode_qk4_v8_d128_k_last:
  Workload 6700a748...: PASSED | 0.015 ms | 80.57x speedup | abs_err=1.56e-02, rel_err=5.33e-01
  Workload d66ae544...: PASSED | 0.015 ms | 74.63x speedup | abs_err=1.22e-04, rel_err=1.29e-02
  Workload bf115ff9...: PASSED | 0.015 ms | 74.73x speedup | abs_err=1.53e-05, rel_err=1.06e-02
  Workload 30bb1856...: PASSED | 0.015 ms | 76.33x speedup | abs_err=1.81e-05, rel_err=4.92e-02
  Workload 48a5be68...: PASSED | 0.017 ms | 68.37x speedup | abs_err=1.53e-05, rel_err=1.48e-01
  Workload 6f09252c...: PASSED | 0.015 ms | 73.04x speedup | abs_err=4.88e-04, rel_err=1.59e-02
  Workload 1e4e5a87...: PASSED | 0.015 ms | 72.58x speedup | abs_err=1.53e-05, rel_err=1.06e-01
  Workload 8e3e1aa6...: PASSED | 0.015 ms | 74.42x speedup | abs_err=2.44e-04, rel_err=2.81e-02
  Workload 741eb2c4...: PASSED | 0.015 ms | 76.36x speedup | abs_err=3.05e-05, rel_err=4.03e-02
  Workload 562d6431...: PASSED | 0.015 ms | 75.10x speedup | abs_err=1.56e-02, rel_err=1.06e-02
  Workload cf76ded3...: PASSED | 0.015 ms | 74.91x speedup | abs_err=2.29e-05, rel_err=1.33e-02
  Workload 55b3b0a4...: PASSED | 0.015 ms | 73.62x speedup | abs_err=2.29e-05, rel_err=4.41e-02
  Workload 8add0e58...: PASSED | 0.015 ms | 87.70x speedup | abs_err=1.72e-05, rel_err=1.38e-02
  Workload 5727c2b1...: PASSED | 0.015 ms | 74.20x speedup | abs_err=7.81e-03, rel_err=1.21e-01
  Workload cc8d77b2...: PASSED | 0.015 ms | 75.59x speedup | abs_err=1.14e-05, rel_err=2.87e-02
  Workload 3bd97bb8...: PASSED | 0.015 ms | 74.17x speedup | abs_err=1.91e-05, rel_err=3.08e-01
  Workload f1b6043f...: PASSED | 0.015 ms | 74.16x speedup | abs_err=1.91e-05, rel_err=6.57e-02
  Workload 8038c14e...: PASSED | 0.015 ms | 76.24x speedup | abs_err=3.05e-05, rel_err=7.28e-02
  Workload 03436065...: PASSED | 0.015 ms | 72.92x speedup | abs_err=3.91e-03, rel_err=2.43e-01
  Workload c40fb468...: PASSED | 0.017 ms | 65.94x speedup | abs_err=1.95e-03, rel_err=7.97e-02
Stopping app - local entrypoint completed.
✓ App completed. View run at https://modal.com/apps/mlsys-flashinfer-26/main/ap-BBj55dpwTlJzEmbvfte07B
```

```bash
gdn_decode_qk4_v8_d128_k_last:
  Workload 6700a748...: PASSED | 0.016 ms | 83.51x speedup | abs_err=2.44e-04, rel_err=1.25e-02
  Workload d66ae544...: PASSED | 0.016 ms | 80.61x speedup | abs_err=1.53e-05, rel_err=1.25e-01
  Workload bf115ff9...: PASSED | 0.016 ms | 80.73x speedup | abs_err=9.77e-04, rel_err=9.15e-02
  Workload 30bb1856...: PASSED | 0.016 ms | 81.52x speedup | abs_err=1.72e-05, rel_err=1.37e-01
  Workload 48a5be68...: PASSED | 0.016 ms | 82.87x speedup | abs_err=1.56e-02, rel_err=4.26e-02
  Workload 6f09252c...: PASSED | 0.016 ms | 83.16x speedup | abs_err=6.10e-05, rel_err=1.38e-02
  Workload 1e4e5a87...: PASSED | 0.016 ms | 81.84x speedup | abs_err=1.95e-03, rel_err=5.29e-01
  Workload 8e3e1aa6...: PASSED | 0.016 ms | 82.53x speedup | abs_err=9.77e-04, rel_err=8.88e-01
  Workload 741eb2c4...: PASSED | 0.016 ms | 82.89x speedup | abs_err=2.67e-05, rel_err=1.11e-01
  Workload 562d6431...: PASSED | 0.016 ms | 83.48x speedup | abs_err=1.95e-03, rel_err=2.88e-02
  Workload cf76ded3...: PASSED | 0.016 ms | 81.20x speedup | abs_err=1.53e-05, rel_err=2.22e-02
  Workload 55b3b0a4...: PASSED | 0.016 ms | 81.32x speedup | abs_err=2.44e-04, rel_err=2.83e-01
  Workload 8add0e58...: PASSED | 0.016 ms | 82.44x speedup | abs_err=6.10e-05, rel_err=7.31e+00
  Workload 5727c2b1...: PASSED | 0.016 ms | 79.83x speedup | abs_err=1.72e-05, rel_err=4.11e-02
  Workload cc8d77b2...: PASSED | 0.016 ms | 81.47x speedup | abs_err=1.56e-02, rel_err=7.36e-02
  Workload 3bd97bb8...: PASSED | 0.016 ms | 80.79x speedup | abs_err=2.29e-05, rel_err=2.33e-01
  Workload f1b6043f...: PASSED | 0.016 ms | 82.21x speedup | abs_err=6.10e-05, rel_err=2.20e-01
  Workload 8038c14e...: PASSED | 0.016 ms | 80.74x speedup | abs_err=7.81e-03, rel_err=2.09e-01
  Workload 03436065...: PASSED | 0.016 ms | 80.60x speedup | abs_err=6.10e-05, rel_err=8.58e-02
  Workload c40fb468...: PASSED | 0.016 ms | 81.37x speedup | abs_err=3.91e-03, rel_err=5.58e-02
Stopping app - local entrypoint completed.
✓ App completed. View run at https://modal.com/apps/mlsys-flashinfer-26/main/ap-8jSGaiCjdorEMJIwwobEXS
```

## Kernel 2

- Remove HMEM<>SMEM load for state
- SMEM for V and K
- float4 loads
- Remove the slowdowns I introduced in v1 (OLD__V and NEW_V in SMEM, beta and gate values stored in SMEM instead of registers)

```bash
gdn_decode_qk4_v8_d128_k_last:
  Workload 6700a748...: PASSED | 10.238 µs | 72.69x speedup | abs_err=1.53e-05, rel_err=3.19e-02
  Workload d66ae544...: PASSED | 10.052 µs | 78.93x speedup | abs_err=1.72e-05, rel_err=1.90e-01
  Workload bf115ff9...: PASSED | 10.089 µs | 78.55x speedup | abs_err=6.25e-02, rel_err=1.39e-01
  Workload 30bb1856...: PASSED | 10.067 µs | 78.54x speedup | abs_err=1.53e-05, rel_err=3.01e-02
  Workload 48a5be68...: PASSED | 8.425 µs | 93.90x speedup | abs_err=3.91e-03, rel_err=1.66e-01
  Workload 6f09252c...: PASSED | 9.992 µs | 78.85x speedup | abs_err=1.72e-05, rel_err=1.25e-01
  Workload 1e4e5a87...: PASSED | 8.681 µs | 90.42x speedup | abs_err=1.72e-05, rel_err=2.54e-02
  Workload 8e3e1aa6...: PASSED | 10.097 µs | 78.43x speedup | abs_err=4.88e-04, rel_err=9.78e-02
  Workload 741eb2c4...: PASSED | 10.114 µs | 78.39x speedup | abs_err=2.29e-05, rel_err=9.06e-02
  Workload 562d6431...: PASSED | 10.148 µs | 77.99x speedup | abs_err=1.95e-03, rel_err=1.22e-01
  Workload cf76ded3...: PASSED | 10.016 µs | 78.94x speedup | abs_err=1.95e-03, rel_err=8.76e-03
  Workload 55b3b0a4...: PASSED | 10.268 µs | 77.21x speedup | abs_err=7.81e-03, rel_err=7.12e-02
  Workload 8add0e58...: PASSED | 10.204 µs | 77.08x speedup | abs_err=1.53e-05, rel_err=5.60e-02
  Workload 5727c2b1...: PASSED | 10.033 µs | 78.26x speedup | abs_err=3.05e-05, rel_err=3.11e-02
  Workload cc8d77b2...: PASSED | 10.027 µs | 78.70x speedup | abs_err=2.44e-04, rel_err=3.89e-02
  Workload 3bd97bb8...: PASSED | 10.144 µs | 77.49x speedup | abs_err=2.44e-04, rel_err=2.64e-02
  Workload f1b6043f...: PASSED | 10.109 µs | 77.74x speedup | abs_err=1.62e-05, rel_err=5.53e-02
  Workload 8038c14e...: PASSED | 9.965 µs | 78.85x speedup | abs_err=1.53e-05, rel_err=2.37e-02
  Workload 03436065...: PASSED | 9.919 µs | 79.40x speedup | abs_err=2.44e-04, rel_err=2.91e-01
  Workload c40fb468...: PASSED | 10.041 µs | 78.57x speedup | abs_err=1.22e-04, rel_err=3.11e-02
```

```bash
gdn_decode_qk4_v8_d128_k_last:
  Workload 6700a748...: PASSED | 9.177 µs | 79.17x speedup | abs_err=9.77e-04, rel_err=2.90e-02
  Workload d66ae544...: PASSED | 9.152 µs | 86.06x speedup | abs_err=1.22e-04, rel_err=2.28e-01
  Workload bf115ff9...: PASSED | 9.183 µs | 84.04x speedup | abs_err=7.81e-03, rel_err=1.86e-01
  Workload 30bb1856...: PASSED | 9.215 µs | 82.35x speedup | abs_err=3.05e-05, rel_err=6.34e-02
  Workload 48a5be68...: PASSED | 9.151 µs | 82.30x speedup | abs_err=2.44e-04, rel_err=1.11e-02
  Workload 6f09252c...: PASSED | 9.109 µs | 83.44x speedup | abs_err=1.22e-04, rel_err=4.94e-02
  Workload 1e4e5a87...: PASSED | 9.134 µs | 86.51x speedup | abs_err=2.29e-05, rel_err=2.31e-01
  Workload 8e3e1aa6...: PASSED | 9.139 µs | 85.89x speedup | abs_err=1.22e-04, rel_err=2.96e-01
  Workload 741eb2c4...: PASSED | 9.127 µs | 85.98x speedup | abs_err=2.29e-05, rel_err=3.59e-02
  Workload 562d6431...: PASSED | 9.142 µs | 85.60x speedup | abs_err=2.29e-05, rel_err=5.82e-02
  Workload cf76ded3...: PASSED | 9.179 µs | 85.64x speedup | abs_err=6.25e-02, rel_err=4.24e-02
  Workload 55b3b0a4...: PASSED | 9.158 µs | 82.19x speedup | abs_err=4.88e-04, rel_err=6.10e-02
  Workload 8add0e58...: PASSED | 9.202 µs | 81.41x speedup | abs_err=1.56e-02, rel_err=2.72e-02
  Workload 5727c2b1...: PASSED | 9.319 µs | 84.18x speedup | abs_err=1.95e-03, rel_err=1.44e-01
  Workload cc8d77b2...: PASSED | 9.104 µs | 86.32x speedup | abs_err=3.05e-05, rel_err=6.52e-02
  Workload 3bd97bb8...: PASSED | 9.169 µs | 86.27x speedup | abs_err=1.53e-05, rel_err=4.42e-02
  Workload f1b6043f...: PASSED | 9.118 µs | 83.18x speedup | abs_err=3.91e-03, rel_err=9.59e-01
  Workload 8038c14e...: PASSED | 9.168 µs | 82.74x speedup | abs_err=1.95e-03, rel_err=3.70e-02
  Workload 03436065...: PASSED | 9.147 µs | 86.89x speedup | abs_err=1.72e-05, rel_err=8.03e-02
  Workload c40fb468...: PASSED | 9.164 µs | 82.53x speedup | abs_err=1.95e-03, rel_err=3.15e-01
Stopping app - local entrypoint completed.
✓ App completed. View run at https://modal.com/apps/mlsys-flashinfer-26/main/ap-wTKDa8tqQRbrTvSQSGeB28
```

## Kernel 3

Complete rewrite based on 46-experiment optimization campaign (30 CuTeDSL + 16 CUDA).

Architecture: BV=8, 1 warp (32 threads) per block, 128 CTAs for B=1. Direct TVM FFI C binding via `TVM_FFI_DLL_EXPORT_TYPED_FUNC` (zero Python dispatch overhead).

Key optimizations:
1. float4 vectorized state loads via `__ldg` (128-bit coalesced, read-only cache)
2. Separated load issue from data extraction (maximizes memory-level parallelism)
3. uint2/uint4 vectorized bf16 q/k/v loads
4. `__stcs` streaming stores (bypass L2 on write-back)
5. Paired warp reductions via inline PTX `shfl.sync.bfly` (overlaps 2 shuffle latencies)
6. Fused decay + delta rule (gate decay folded into dot-product FMA chain)
7. Interleaved output computation + state store (overlaps write latency with compute)
8. `__expf`/`__logf`/`__frcp_rn` fast math intrinsics for gates
9. All dimensions hardcoded as `constexpr` (compiler constant-folding)
10. `__launch_bounds__(32, 1)` for maximum register budget per thread

Key insights from campaign:
- **TVM FFI C binding** (`TVM_FFI_DLL_EXPORT_TYPED_FUNC`) is the #1 speedup — eliminates ALL Python dispatch
- **L2 persistence HURTS on B200** — the 126MB L2 naturally caches the 524KB state
- **Shared memory staging HURTS** for this latency-bound kernel
- Kernel is **instruction-latency-limited** at ~25% occupancy, NOT bandwidth-limited (<2% HBM BW at B=1)

<details>
<summary>Run 1 (median 91.2x, 8.3 µs)</summary>

```
gdn_decode_qk4_v8_d128_k_last:
  Workload 6700a748...: PASSED | 8.294 µs | 86.13x speedup | abs_err=5.34e-05, rel_err=2.20e+00
  Workload d66ae544...: PASSED | 8.242 µs | 86.65x speedup | abs_err=3.05e-05, rel_err=1.59e-02
  Workload bf115ff9...: PASSED | 8.182 µs | 94.09x speedup | abs_err=5.34e-05, rel_err=1.37e-01
  Workload 30bb1856...: PASSED | 8.294 µs | 92.05x speedup | abs_err=4.88e-04, rel_err=3.22e-02
  Workload 48a5be68...: PASSED | 8.212 µs | 91.96x speedup | abs_err=1.91e-05, rel_err=6.91e-02
  Workload 6f09252c...: PASSED | 8.320 µs | 90.63x speedup | abs_err=3.81e-05, rel_err=8.86e-02
  Workload 1e4e5a87...: PASSED | 8.246 µs | 91.22x speedup | abs_err=3.91e-03, rel_err=6.71e-02
  Workload 8e3e1aa6...: PASSED | 8.205 µs | 92.52x speedup | abs_err=8.39e-05, rel_err=9.63e-02
  Workload 741eb2c4...: PASSED | 8.315 µs | 91.82x speedup | abs_err=9.77e-04, rel_err=1.16e-02
  Workload 562d6431...: PASSED | 8.268 µs | 91.38x speedup | abs_err=3.12e-02, rel_err=3.67e+01
  Workload cf76ded3...: PASSED | 8.216 µs | 91.93x speedup | abs_err=4.88e-04, rel_err=9.53e-02
  Workload 55b3b0a4...: PASSED | 8.259 µs | 91.35x speedup | abs_err=9.77e-04, rel_err=1.33e-01
  Workload 8add0e58...: PASSED | 8.244 µs | 92.37x speedup | abs_err=2.10e-05, rel_err=3.50e+00
  Workload 5727c2b1...: PASSED | 8.304 µs | 85.42x speedup | abs_err=3.81e-05, rel_err=6.33e-02
  Workload cc8d77b2...: PASSED | 8.299 µs | 86.58x speedup | abs_err=6.10e-05, rel_err=2.68e-01
  Workload 3bd97bb8...: PASSED | 8.332 µs | 85.30x speedup | abs_err=3.91e-03, rel_err=4.34e-02
  Workload f1b6043f...: PASSED | 8.407 µs | 91.68x speedup | abs_err=3.43e-05, rel_err=3.39e-02
  Workload 8038c14e...: PASSED | 8.231 µs | 92.37x speedup | abs_err=6.10e-05, rel_err=3.16e-02
  Workload 03436065...: PASSED | 8.280 µs | 91.44x speedup | abs_err=1.30e-04, rel_err=1.47e-01
  Workload c40fb468...: PASSED | 8.295 µs | 91.00x speedup | abs_err=7.81e-03, rel_err=1.70e-02
```
</details>

<details>
<summary>Run 2 (median 96.5x, 14.1 µs)</summary>

```
gdn_decode_qk4_v8_d128_k_last:
  Workload 6700a748...: PASSED | 14.104 µs | 106.52x speedup | abs_err=7.63e-05, rel_err=4.27e-02
  Workload d66ae544...: PASSED | 13.847 µs | 96.23x speedup | abs_err=3.05e-05, rel_err=1.45e-01
  Workload bf115ff9...: PASSED | 13.972 µs | 96.42x speedup | abs_err=4.88e-04, rel_err=6.16e-02
  Workload 30bb1856...: PASSED | 14.655 µs | 92.91x speedup | abs_err=2.44e-04, rel_err=3.55e-02
  Workload 48a5be68...: PASSED | 14.644 µs | 97.14x speedup | abs_err=9.77e-04, rel_err=1.69e-02
  Workload 6f09252c...: PASSED | 14.535 µs | 97.43x speedup | abs_err=1.22e-04, rel_err=1.75e-02
  Workload 1e4e5a87...: PASSED | 16.771 µs | 83.37x speedup | abs_err=3.05e-05, rel_err=8.38e-01
  Workload 8e3e1aa6...: PASSED | 15.040 µs | 96.02x speedup | abs_err=2.44e-04, rel_err=1.45e-01
  Workload 741eb2c4...: PASSED | 15.651 µs | 88.38x speedup | abs_err=2.67e-05, rel_err=2.98e-02
  Workload 562d6431...: PASSED | 12.870 µs | 106.78x speedup | abs_err=9.77e-04, rel_err=8.51e-02
  Workload cf76ded3...: PASSED | 12.892 µs | 107.65x speedup | abs_err=2.67e-05, rel_err=1.19e-02
  Workload 55b3b0a4...: PASSED | 12.770 µs | 108.51x speedup | abs_err=4.58e-05, rel_err=7.23e-03
  Workload 8add0e58...: PASSED | 14.586 µs | 92.96x speedup | abs_err=3.05e-05, rel_err=5.21e-02
  Workload 5727c2b1...: PASSED | 15.664 µs | 80.29x speedup | abs_err=5.34e-05, rel_err=2.39e-01
  Workload cc8d77b2...: PASSED | 15.391 µs | 86.31x speedup | abs_err=6.10e-05, rel_err=1.47e-02
  Workload 3bd97bb8...: PASSED | 12.618 µs | 107.00x speedup | abs_err=3.12e-02, rel_err=6.34e-02
  Workload f1b6043f...: PASSED | 12.896 µs | 109.20x speedup | abs_err=4.88e-04, rel_err=2.20e-02
  Workload 8038c14e...: PASSED | 12.735 µs | 109.82x speedup | abs_err=3.05e-05, rel_err=1.28e-01
  Workload 03436065...: PASSED | 13.870 µs | 96.64x speedup | abs_err=1.56e-02, rel_err=1.52e-01
  Workload c40fb468...: PASSED | 17.996 µs | 69.76x speedup | abs_err=3.12e-02, rel_err=2.00e-02
```
</details>

<details>
<summary>Run 3 (median 93.8x, 12.4 µs)</summary>

```
gdn_decode_qk4_v8_d128_k_last:
  Workload 6700a748...: PASSED | 12.410 µs | 95.15x speedup | abs_err=3.91e-03, rel_err=6.66e-02
  Workload d66ae544...: PASSED | 12.236 µs | 94.02x speedup | abs_err=3.43e-05, rel_err=3.14e-02
  Workload bf115ff9...: PASSED | 12.459 µs | 92.95x speedup | abs_err=1.22e-04, rel_err=9.36e-02
  Workload 30bb1856...: PASSED | 12.350 µs | 94.78x speedup | abs_err=2.67e-05, rel_err=8.10e+00
  Workload 48a5be68...: PASSED | 12.468 µs | 93.86x speedup | abs_err=1.53e-05, rel_err=3.51e-02
  Workload 6f09252c...: PASSED | 12.471 µs | 93.95x speedup | abs_err=2.29e-05, rel_err=2.02e-02
  Workload 1e4e5a87...: PASSED | 12.444 µs | 95.08x speedup | abs_err=3.12e-02, rel_err=9.26e+01
  Workload 8e3e1aa6...: PASSED | 12.659 µs | 92.25x speedup | abs_err=1.95e-03, rel_err=1.64e+01
  Workload 741eb2c4...: PASSED | 12.593 µs | 92.89x speedup | abs_err=3.81e-05, rel_err=1.36e-02
  Workload 562d6431...: PASSED | 12.373 µs | 93.80x speedup | abs_err=1.22e-04, rel_err=3.67e-02
  Workload cf76ded3...: PASSED | 12.517 µs | 93.33x speedup | abs_err=9.77e-04, rel_err=5.77e-02
  Workload 55b3b0a4...: PASSED | 12.401 µs | 93.86x speedup | abs_err=3.05e-05, rel_err=3.25e-02
  Workload 8add0e58...: PASSED | 12.577 µs | 93.04x speedup | abs_err=6.10e-05, rel_err=1.43e-02
  Workload 5727c2b1...: PASSED | 12.678 µs | 92.10x speedup | abs_err=2.44e-04, rel_err=4.23e-01
  Workload cc8d77b2...: PASSED | 12.548 µs | 93.71x speedup | abs_err=2.10e-05, rel_err=1.14e-01
  Workload 3bd97bb8...: PASSED | 12.468 µs | 94.11x speedup | abs_err=3.05e-05, rel_err=1.80e-01
  Workload f1b6043f...: PASSED | 12.292 µs | 94.94x speedup | abs_err=3.24e-05, rel_err=2.07e-02
  Workload 8038c14e...: PASSED | 12.304 µs | 93.53x speedup | abs_err=9.77e-04, rel_err=1.25e-02
  Workload 03436065...: PASSED | 12.351 µs | 93.40x speedup | abs_err=9.77e-04, rel_err=1.87e-01
  Workload c40fb468...: PASSED | 12.589 µs | 92.69x speedup | abs_err=1.22e-04, rel_err=1.48e-01
```
</details>

Learnings on why v3 is better:

v3 kernel issues a single warp and process 8 corresponding output rows.

1. 8 rows per block - each kernel is doing more work. Even though for v2 a single kernel needed to do less work, and occupancy was higher, v3 still wins because of the block size (32, 1). Each block has 1 warp working on 8 output rows.
2. The major win is that per row, v3 kernel is doing less work. For eg: gate, beta calculations are done once per warp and used for all 8 rows. In v2 (block dims = (32, 8)), a single thread per row calculated it and broadcasted it within the warp.
3. For 8 rows of the same head, q, k, gate scalars, indexing math, and reduction setup are mostly shared. v3 amortizes the invariant work.
