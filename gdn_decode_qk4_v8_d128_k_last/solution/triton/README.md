# Triton GDN Decode Kernel

We need to read:

1. State (VxK) — k-last layout
2. Key (1xK) and Value (1xV) vectors
3. Query (1xK)
4. Scalars (a, dt_bias, A_log, b, scale)

and write:

1. Updated state (VxK)
2. Output (1xV)

## Benchmark Summary (Modal B200)

| Kernel | Benchmark Latency | Kernel Latency (NCU) | Speedup | Regs/Thread |
|--------|-------------------|---------------------|---------|-------------|
| v1     | ~104 µs           | —                   | ~12x    | —           |
| v2     | ~88 µs            | 4.96 µs             | ~14x    | 32          |
| v3     | ~55 µs            | 4.74 µs             | ~12x    | 30          |


## Kernel v1

- Naive baseline
- 1D grid: one Triton program per (batch, v_head)
- Sequential V-tile loop: state [128, 128] processed in [BLOCK_V=16, K=128] tiles
- GVA head expansion: 8 V-heads map to 4 Q/K-heads (ratio 2:1)
- All computation in f32, output stored as bf16

<details>
<summary>Run 1 (median 12.2x, 96.7 µs)</summary>

```
gdn_decode_qk4_v8_d128_k_last:
  Workload 6700a748...: PASSED | 97.381 µs | 12.01x speedup | abs_err=9.77e-04, rel_err=9.94e-02
  Workload d66ae544...: PASSED | 92.589 µs | 12.63x speedup | abs_err=2.29e-05, rel_err=1.35e-02
  Workload bf115ff9...: PASSED | 92.069 µs | 12.70x speedup | abs_err=6.25e-02, rel_err=2.16e-01
  Workload 30bb1856...: PASSED | 92.386 µs | 12.88x speedup | abs_err=1.22e-04, rel_err=2.51e-02
  Workload 48a5be68...: PASSED | 91.197 µs | 13.16x speedup | abs_err=2.29e-05, rel_err=1.06e-02
  Workload 6f09252c...: PASSED | 90.179 µs | 13.25x speedup | abs_err=3.05e-05, rel_err=2.80e-02
  Workload 1e4e5a87...: PASSED | 92.861 µs | 12.54x speedup | abs_err=7.81e-03, rel_err=1.52e-02
  Workload 8e3e1aa6...: PASSED | 90.688 µs | 12.91x speedup | abs_err=3.12e-02, rel_err=8.53e-02
  Workload 741eb2c4...: PASSED | 91.227 µs | 12.74x speedup | abs_err=3.05e-05, rel_err=1.14e-01
  Workload 562d6431...: PASSED | 105.658 µs | 11.29x speedup | abs_err=3.81e-05, rel_err=1.54e+00
  Workload cf76ded3...: PASSED | 99.671 µs | 11.94x speedup | abs_err=1.91e-05, rel_err=8.40e-02
  Workload 55b3b0a4...: PASSED | 109.235 µs | 10.91x speedup | abs_err=3.12e-02, rel_err=1.46e-01
  Workload 8add0e58...: PASSED | 106.393 µs | 11.02x speedup | abs_err=6.10e-05, rel_err=4.84e-02
  Workload 5727c2b1...: PASSED | 89.007 µs | 12.94x speedup | abs_err=3.81e-05, rel_err=6.95e-02
  Workload cc8d77b2...: PASSED | 109.496 µs | 10.91x speedup | abs_err=6.48e-05, rel_err=4.42e-01
  Workload 3bd97bb8...: PASSED | 96.157 µs | 12.04x speedup | abs_err=1.91e-05, rel_err=9.33e-02
  Workload f1b6043f...: PASSED | 97.229 µs | 12.26x speedup | abs_err=3.05e-05, rel_err=2.36e-02
  Workload 8038c14e...: PASSED | 108.021 µs | 10.88x speedup | abs_err=1.95e-03, rel_err=1.27e-01
  Workload 03436065...: PASSED | 106.439 µs | 11.18x speedup | abs_err=7.25e-05, rel_err=1.68e-01
  Workload c40fb468...: PASSED | 101.858 µs | 11.59x speedup | abs_err=2.29e-05, rel_err=4.33e-02
```
</details>

<details>
<summary>Run 2 (median 14.5x, 91.0 µs)</summary>

```
gdn_decode_qk4_v8_d128_k_last:
  Workload 6700a748...: PASSED | 99.021 µs | 13.76x speedup | abs_err=3.43e-05, rel_err=1.32e-01
  Workload d66ae544...: PASSED | 92.958 µs | 13.80x speedup | abs_err=1.91e-05, rel_err=2.17e-02
  Workload bf115ff9...: PASSED | 89.686 µs | 14.79x speedup | abs_err=9.77e-04, rel_err=1.16e-02
  Workload 30bb1856...: PASSED | 88.456 µs | 15.05x speedup | abs_err=1.72e-05, rel_err=2.49e-02
  Workload 48a5be68...: PASSED | 88.365 µs | 14.87x speedup | abs_err=9.77e-04, rel_err=3.14e-02
  Workload 6f09252c...: PASSED | 104.162 µs | 12.15x speedup | abs_err=7.81e-03, rel_err=2.19e-01
  Workload 1e4e5a87...: PASSED | 111.642 µs | 10.32x speedup | abs_err=9.77e-04, rel_err=1.13e-01
  Workload 8e3e1aa6...: PASSED | 94.514 µs | 13.09x speedup | abs_err=1.56e-02, rel_err=2.77e-01
  Workload 741eb2c4...: PASSED | 88.676 µs | 14.94x speedup | abs_err=9.77e-04, rel_err=1.86e-01
  Workload 562d6431...: PASSED | 91.520 µs | 14.31x speedup | abs_err=2.29e-05, rel_err=1.36e-01
  Workload cf76ded3...: PASSED | 89.479 µs | 14.82x speedup | abs_err=3.12e-02, rel_err=1.73e-02
  Workload 55b3b0a4...: PASSED | 89.106 µs | 14.68x speedup | abs_err=6.10e-05, rel_err=2.55e-02
  Workload 8add0e58...: PASSED | 109.204 µs | 11.10x speedup | abs_err=3.05e-05, rel_err=2.32e-01
  Workload 5727c2b1...: PASSED | 104.600 µs | 11.46x speedup | abs_err=2.67e-05, rel_err=4.07e-02
  Workload cc8d77b2...: PASSED | 88.978 µs | 14.64x speedup | abs_err=5.34e-05, rel_err=3.21e-02
  Workload 3bd97bb8...: PASSED | 88.427 µs | 14.97x speedup | abs_err=3.05e-05, rel_err=4.81e-02
  Workload f1b6043f...: PASSED | 90.425 µs | 14.59x speedup | abs_err=2.38e-05, rel_err=5.03e-02
  Workload 8038c14e...: PASSED | 89.885 µs | 14.78x speedup | abs_err=2.29e-05, rel_err=4.46e-02
  Workload 03436065...: PASSED | 96.211 µs | 13.47x speedup | abs_err=8.01e-05, rel_err=6.57e-02
  Workload c40fb468...: PASSED | 109.217 µs | 10.91x speedup | abs_err=4.58e-05, rel_err=1.19e-01
```
</details>

<details>
<summary>Run 3 (median 11.6x, 110.2 µs)</summary>

```
gdn_decode_qk4_v8_d128_k_last:
  Workload 6700a748...: PASSED | 113.103 µs | 11.39x speedup | abs_err=3.81e-05, rel_err=4.28e-02
  Workload d66ae544...: PASSED | 109.136 µs | 11.73x speedup | abs_err=2.29e-05, rel_err=6.31e-01
  Workload bf115ff9...: PASSED | 110.783 µs | 11.56x speedup | abs_err=1.91e-05, rel_err=3.26e-02
  Workload 30bb1856...: PASSED | 110.160 µs | 11.64x speedup | abs_err=1.56e-02, rel_err=2.31e-02
  Workload 48a5be68...: PASSED | 109.451 µs | 11.73x speedup | abs_err=9.77e-04, rel_err=1.99e-01
  Workload 6f09252c...: PASSED | 110.703 µs | 11.56x speedup | abs_err=3.05e-05, rel_err=2.33e-02
  Workload 1e4e5a87...: PASSED | 110.183 µs | 11.59x speedup | abs_err=1.95e-03, rel_err=5.26e-02
  Workload 8e3e1aa6...: PASSED | 109.978 µs | 11.61x speedup | abs_err=1.95e-03, rel_err=6.74e-02
  Workload 741eb2c4...: PASSED | 110.675 µs | 11.57x speedup | abs_err=1.95e-03, rel_err=2.32e-02
  Workload 562d6431...: PASSED | 109.501 µs | 11.68x speedup | abs_err=3.12e-02, rel_err=1.64e-01
  Workload cf76ded3...: PASSED | 109.804 µs | 11.64x speedup | abs_err=2.29e-05, rel_err=9.85e-03
  Workload 55b3b0a4...: PASSED | 109.685 µs | 11.66x speedup | abs_err=2.44e-04, rel_err=3.89e-01
  Workload 8add0e58...: PASSED | 109.318 µs | 11.69x speedup | abs_err=6.25e-02, rel_err=2.97e-01
  Workload 5727c2b1...: PASSED | 111.045 µs | 11.53x speedup | abs_err=3.05e-05, rel_err=9.91e-02
  Workload cc8d77b2...: PASSED | 109.454 µs | 11.67x speedup | abs_err=1.56e-02, rel_err=1.82e-01
  Workload 3bd97bb8...: PASSED | 109.685 µs | 11.65x speedup | abs_err=1.22e-04, rel_err=4.57e-02
  Workload f1b6043f...: PASSED | 110.852 µs | 11.54x speedup | abs_err=1.95e-03, rel_err=1.89e-02
  Workload 8038c14e...: PASSED | 110.414 µs | 11.59x speedup | abs_err=3.05e-05, rel_err=2.35e-02
  Workload 03436065...: PASSED | 110.376 µs | 11.61x speedup | abs_err=9.77e-04, rel_err=7.18e-02
  Workload c40fb468...: PASSED | 111.471 µs | 11.48x speedup | abs_err=9.77e-04, rel_err=3.33e-02
```
</details>

## Kernel v2

Evolved from v1 through 30 experiments on B200. Key changes:

- 3D grid `(B, NUM_V_HEADS, V_DIM//BV)` — parallelizes V-tile dimension across CTAs
- Block pointers via `tl.make_block_ptr` — TMA-eligible state loads/stores on Blackwell
- Tuned: `num_warps=8, num_stages=4, BLOCK_V=16`

<details>
<summary>Run 1 (median 13.7x, 88.2 µs)</summary>

```
gdn_decode_qk4_v8_d128_k_last:
  Workload 6700a748...: PASSED | 87.602 µs | 13.78x speedup | abs_err=3.81e-05, rel_err=2.48e-01
  Workload d66ae544...: PASSED | 87.421 µs | 13.78x speedup | abs_err=1.95e-03, rel_err=2.87e-02
  Workload bf115ff9...: PASSED | 87.709 µs | 13.74x speedup | abs_err=2.67e-05, rel_err=5.82e-02
  Workload 30bb1856...: PASSED | 88.763 µs | 13.50x speedup | abs_err=3.05e-05, rel_err=3.74e-02
  Workload 48a5be68...: PASSED | 87.431 µs | 13.80x speedup | abs_err=1.22e-04, rel_err=2.07e-02
  Workload 6f09252c...: PASSED | 87.357 µs | 13.82x speedup | abs_err=2.67e-05, rel_err=1.39e-01
  Workload 1e4e5a87...: PASSED | 87.386 µs | 13.81x speedup | abs_err=1.53e-05, rel_err=3.58e-01
  Workload 8e3e1aa6...: PASSED | 88.886 µs | 13.73x speedup | abs_err=1.95e-03, rel_err=3.47e-02
  Workload 741eb2c4...: PASSED | 88.710 µs | 13.54x speedup | abs_err=3.91e-03, rel_err=1.91e-02
  Workload 562d6431...: PASSED | 95.923 µs | 12.49x speedup | abs_err=1.91e-05, rel_err=9.31e-01
  Workload cf76ded3...: PASSED | 88.273 µs | 13.61x speedup | abs_err=3.91e-03, rel_err=4.13e-01
  Workload 55b3b0a4...: PASSED | 92.138 µs | 13.10x speedup | abs_err=3.91e-03, rel_err=5.42e-02
  Workload 8add0e58...: PASSED | 87.558 µs | 13.67x speedup | abs_err=4.88e-04, rel_err=6.73e-02
  Workload 5727c2b1...: PASSED | 88.832 µs | 13.54x speedup | abs_err=3.43e-05, rel_err=1.77e-02
  Workload cc8d77b2...: PASSED | 87.186 µs | 13.83x speedup | abs_err=6.25e-02, rel_err=1.19e-01
  Workload 3bd97bb8...: PASSED | 91.514 µs | 13.13x speedup | abs_err=1.91e-05, rel_err=3.58e-01
  Workload f1b6043f...: PASSED | 87.656 µs | 13.77x speedup | abs_err=2.48e-05, rel_err=2.64e-01
  Workload 8038c14e...: PASSED | 88.770 µs | 13.52x speedup | abs_err=9.77e-04, rel_err=3.20e-01
  Workload 03436065...: PASSED | 88.710 µs | 13.56x speedup | abs_err=3.12e-02, rel_err=1.68e+00
  Workload c40fb468...: PASSED | 88.146 µs | 13.68x speedup | abs_err=2.44e-04, rel_err=3.23e-02
```
</details>

<details>
<summary>Run 2 (median 14.2x, 106.8 µs)</summary>

```
gdn_decode_qk4_v8_d128_k_last:
  Workload 6700a748...: PASSED | 107.166 µs | 12.62x speedup | abs_err=3.05e-05, rel_err=3.52e-01
  Workload d66ae544...: PASSED | 105.317 µs | 14.43x speedup | abs_err=3.05e-05, rel_err=7.15e-03
  Workload bf115ff9...: PASSED | 105.870 µs | 14.38x speedup | abs_err=2.67e-05, rel_err=3.72e-02
  Workload 30bb1856...: PASSED | 106.402 µs | 14.28x speedup | abs_err=1.91e-05, rel_err=2.38e-02
  Workload 48a5be68...: PASSED | 106.638 µs | 14.27x speedup | abs_err=1.91e-05, rel_err=3.73e-02
  Workload 6f09252c...: PASSED | 111.447 µs | 13.59x speedup | abs_err=3.05e-05, rel_err=4.42e-02
  Workload 1e4e5a87...: PASSED | 107.185 µs | 14.22x speedup | abs_err=1.53e-05, rel_err=4.34e-02
  Workload 8e3e1aa6...: PASSED | 106.744 µs | 14.24x speedup | abs_err=1.95e-03, rel_err=1.06e-02
  Workload 741eb2c4...: PASSED | 107.801 µs | 14.12x speedup | abs_err=1.22e-04, rel_err=5.33e-02
  Workload 562d6431...: PASSED | 106.495 µs | 14.31x speedup | abs_err=2.44e-04, rel_err=1.25e-01
  Workload cf76ded3...: PASSED | 108.494 µs | 14.00x speedup | abs_err=2.29e-05, rel_err=1.05e-02
  Workload 55b3b0a4...: PASSED | 106.667 µs | 14.22x speedup | abs_err=3.05e-05, rel_err=2.11e-02
  Workload 8add0e58...: PASSED | 105.746 µs | 14.33x speedup | abs_err=9.77e-04, rel_err=8.80e-02
  Workload 5727c2b1...: PASSED | 108.074 µs | 14.07x speedup | abs_err=7.81e-03, rel_err=1.02e-01
  Workload cc8d77b2...: PASSED | 106.037 µs | 14.37x speedup | abs_err=6.87e-05, rel_err=2.20e-02
  Workload 3bd97bb8...: PASSED | 107.244 µs | 14.15x speedup | abs_err=1.95e-03, rel_err=8.70e-02
  Workload f1b6043f...: PASSED | 107.850 µs | 14.09x speedup | abs_err=3.05e-05, rel_err=1.46e-01
  Workload 8038c14e...: PASSED | 106.862 µs | 14.19x speedup | abs_err=2.29e-05, rel_err=3.63e-02
  Workload 03436065...: PASSED | 108.206 µs | 14.06x speedup | abs_err=4.88e-04, rel_err=2.04e-01
  Workload c40fb468...: PASSED | 106.774 µs | 14.26x speedup | abs_err=1.91e-05, rel_err=1.26e-01
```
</details>

<details>
<summary>Run 3 (median 13.3x, 87.8 µs)</summary>

```
gdn_decode_qk4_v8_d128_k_last:
  Workload 6700a748...: PASSED | 87.885 µs | 13.37x speedup | abs_err=3.12e-02, rel_err=7.39e+00
  Workload d66ae544...: PASSED | 87.399 µs | 14.02x speedup | abs_err=3.12e-02, rel_err=1.99e-02
  Workload bf115ff9...: PASSED | 87.109 µs | 14.86x speedup | abs_err=2.29e-05, rel_err=7.44e-02
  Workload 30bb1856...: PASSED | 87.360 µs | 13.29x speedup | abs_err=3.05e-05, rel_err=1.02e-01
  Workload 48a5be68...: PASSED | 87.841 µs | 13.20x speedup | abs_err=1.91e-05, rel_err=8.81e-02
  Workload 6f09252c...: PASSED | 87.278 µs | 13.27x speedup | abs_err=3.91e-03, rel_err=1.30e-01
  Workload 1e4e5a87...: PASSED | 88.039 µs | 13.25x speedup | abs_err=1.91e-05, rel_err=1.00e-01
  Workload 8e3e1aa6...: PASSED | 87.650 µs | 13.22x speedup | abs_err=4.58e-05, rel_err=4.02e-01
  Workload 741eb2c4...: PASSED | 88.283 µs | 14.18x speedup | abs_err=3.05e-05, rel_err=1.90e-02
  Workload 562d6431...: PASSED | 87.799 µs | 13.27x speedup | abs_err=1.91e-05, rel_err=2.64e-02
  Workload cf76ded3...: PASSED | 87.767 µs | 13.25x speedup | abs_err=2.29e-05, rel_err=1.85e-02
  Workload 55b3b0a4...: PASSED | 88.050 µs | 13.21x speedup | abs_err=2.29e-05, rel_err=8.67e-02
  Workload 8add0e58...: PASSED | 87.079 µs | 13.44x speedup | abs_err=3.05e-05, rel_err=1.16e-01
  Workload 5727c2b1...: PASSED | 91.313 µs | 12.74x speedup | abs_err=3.43e-05, rel_err=1.25e-01
  Workload cc8d77b2...: PASSED | 87.056 µs | 13.39x speedup | abs_err=6.87e-05, rel_err=4.90e-01
  Workload 3bd97bb8...: PASSED | 88.529 µs | 13.13x speedup | abs_err=7.81e-03, rel_err=9.10e-02
  Workload f1b6043f...: PASSED | 86.913 µs | 14.49x speedup | abs_err=2.48e-05, rel_err=6.96e-02
  Workload 8038c14e...: PASSED | 87.625 µs | 13.35x speedup | abs_err=3.12e-02, rel_err=2.42e-02
  Workload 03436065...: PASSED | 89.750 µs | 12.90x speedup | abs_err=9.54e-05, rel_err=7.59e-02
  Workload c40fb468...: PASSED | 88.186 µs | 13.27x speedup | abs_err=1.72e-05, rel_err=1.99e-02
```
</details>

## Kernel v3 (current)

Evolved from v2 through 24 experiments on B200. Key changes:

- **BV=8** (128 blocks) — 86% SM coverage on 148 SMs vs v2's 43% with BV=16
- **Plain pointer arithmetic** — generates tighter PTX than `make_block_ptr` descriptors
- **Live range optimization** — output computed via algebraic identity `output = scale * (old_state@q + delta_v * dot(k,q))` BEFORE materializing `state_out`. Decouples output from state store, improving IPC 25%
- **Compact delta** — `delta_v = beta*(v - old_v)` eliminates intermediate `new_v`

**NCU Profile (B=1):**

| Metric | v2 | v3 |
|--------|----|----|
| Kernel time | 4.96 µs | 4.74 µs |
| IPC Active | 0.54 | 0.72 |
| SM Active Cycles | 1497 | 2425 |
| Compute Throughput | 1.72% | 4.75% |
| L1/TEX Throughput | 12.19% | 13.22% |
| Elapsed Cycles | 9748 | 8796 |

<details>
<summary>Benchmark run (median ~12x, ~55 µs)</summary>

```
gdn_decode_qk4_v8_d128_k_last:
  Workload 6700a748...: PASSED | 55.441 µs | 11.97x speedup | abs_err=3.43e-05, rel_err=1.35e-02
  Workload d66ae544...: PASSED | 55.385 µs | 12.03x speedup | abs_err=2.29e-05, rel_err=1.14e+00
  Workload bf115ff9...: PASSED | 55.517 µs | 11.98x speedup | abs_err=2.29e-05, rel_err=1.24e-02
  Workload 30bb1856...: PASSED | 55.201 µs | 12.04x speedup | abs_err=6.10e-05, rel_err=2.22e-01
  Workload 48a5be68...: PASSED | 54.707 µs | 12.22x speedup | abs_err=6.10e-05, rel_err=5.69e-02
  Workload 6f09252c...: PASSED | 55.221 µs | 12.04x speedup | abs_err=3.05e-05, rel_err=1.10e-01
  Workload 1e4e5a87...: PASSED | 55.215 µs | 12.15x speedup | abs_err=2.44e-04, rel_err=1.04e-01
  Workload 8e3e1aa6...: PASSED | 55.114 µs | 12.12x speedup | abs_err=4.88e-04, rel_err=5.22e-02
  Workload 741eb2c4...: PASSED | 55.120 µs | 12.13x speedup | abs_err=1.56e-02, rel_err=2.57e-01
  Workload 562d6431...: PASSED | 55.623 µs | 11.97x speedup | abs_err=1.95e-03, rel_err=2.26e-01
  Workload cf76ded3...: PASSED | 55.321 µs | 12.07x speedup | abs_err=6.10e-05, rel_err=2.06e-02
  Workload 55b3b0a4...: PASSED | 55.099 µs | 12.19x speedup | abs_err=9.77e-04, rel_err=2.09e-01
  Workload 8add0e58...: PASSED | 55.276 µs | 12.07x speedup | abs_err=1.56e-02, rel_err=1.52e-02
  Workload 5727c2b1...: PASSED | 55.573 µs | 11.99x speedup | abs_err=1.56e-02, rel_err=3.48e-02
  Workload cc8d77b2...: PASSED | 55.520 µs | 12.04x speedup | abs_err=4.88e-04, rel_err=5.70e-02
  Workload 3bd97bb8...: PASSED | 55.706 µs | 11.98x speedup | abs_err=1.22e-04, rel_err=1.07e-02
  Workload f1b6043f...: PASSED | 55.153 µs | 12.08x speedup | abs_err=2.48e-05, rel_err=3.33e-01
  Workload 8038c14e...: PASSED | 55.272 µs | 12.11x speedup | abs_err=1.95e-03, rel_err=1.87e-01
  Workload 03436065...: PASSED | 55.715 µs | 12.03x speedup | abs_err=4.88e-04, rel_err=2.97e-01
  Workload c40fb468...: PASSED | 55.267 µs | 12.15x speedup | abs_err=1.56e-02, rel_err=1.92e-01
```
</details>
