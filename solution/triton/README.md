# Triton GDN Decode Kernel

We need to read:

1. State (VxK) — k-last layout
2. Key (1xK) and Value (1xV) vectors
3. Query (1xK)
4. Scalars (a, dt_bias, A_log, b, scale)

and write:

1. Updated state (VxK)
2. Output (1xV)

## Kernel 1

- Naïve baseline
- One Triton program per (batch, v_head)
- Sequential V-tile loop: state [128, 128] processed in [BLOCK_V=16, K=128] tiles
- GVA head expansion: 8 V-heads map to 4 Q/K-heads (ratio 2:1)
- All computation in f32, output stored as bf16
- Design choices:
  - No strides — hardcoded contiguous offsets
  - DLPack conversion for tvm_ffi.core.Tensor inputs
  - Pre-allocated output tensors from framework

Results

```bash
gdn_decode_qk4_v8_d128_k_last:
  Workload 6700a748...: PASSED | 80.463 µs | 16.58x speedup | abs_err=3.91e-03, rel_err=1.68e-01
  Workload d66ae544...: PASSED | 80.136 µs | 14.34x speedup | abs_err=9.77e-04, rel_err=1.02e-01
  Workload bf115ff9...: PASSED | 80.463 µs | 16.58x speedup | abs_err=3.91e-03, rel_err=1.68e-01
  Workload 30bb1856...: PASSED | 81.056 µs | 14.18x speedup | abs_err=1.91e-05, rel_err=1.52e-02
  Workload 48a5be68...: PASSED | 80.516 µs | 14.29x speedup | abs_err=7.81e-03, rel_err=8.97e-02
  Workload 6f09252c...: PASSED | 80.136 µs | 14.34x speedup | abs_err=9.77e-04, rel_err=1.02e-01
  Workload 1e4e5a87...: PASSED | 79.500 µs | 16.10x speedup | abs_err=1.56e-02, rel_err=1.59e-02
  Workload 8e3e1aa6...: PASSED | 80.310 µs | 16.68x speedup | abs_err=1.22e-04, rel_err=4.21e-02
  Workload 741eb2c4...: PASSED | 80.917 µs | 16.51x speedup | abs_err=3.91e-03, rel_err=4.55e-01
  Workload 562d6431...: PASSED | 81.056 µs | 14.18x speedup | abs_err=1.91e-05, rel_err=1.52e-02
  Workload cf76ded3...: PASSED | 81.074 µs | 16.45x speedup | abs_err=3.91e-03, rel_err=1.20e-01
  Workload 55b3b0a4...: PASSED | 80.463 µs | 16.58x speedup | abs_err=3.91e-03, rel_err=1.68e-01
  Workload 8add0e58...: PASSED | 80.516 µs | 14.29x speedup | abs_err=7.81e-03, rel_err=8.97e-02
  Workload 5727c2b1...: PASSED | 80.136 µs | 14.34x speedup | abs_err=9.77e-04, rel_err=1.02e-01
  Workload cc8d77b2...: PASSED | 79.500 µs | 16.10x speedup | abs_err=1.56e-02, rel_err=1.59e-02
  Workload 3bd97bb8...: PASSED | 80.310 µs | 16.68x speedup | abs_err=1.22e-04, rel_err=4.21e-02
  Workload f1b6043f...: PASSED | 80.917 µs | 16.51x speedup | abs_err=3.91e-03, rel_err=4.55e-01
  Workload 8038c14e...: PASSED | 81.056 µs | 14.18x speedup | abs_err=1.91e-05, rel_err=1.52e-02
  Workload 03436065...: PASSED | 81.074 µs | 16.45x speedup | abs_err=3.91e-03, rel_err=1.20e-01
  Workload c40fb468...: PASSED | 80.978 µs | 14.19x speedup | abs_err=9.77e-04, rel_err=1.50e-01
```

## Kernel 2

Evolved from v1 through 30 experiments on B200. Key changes:

- 3D grid `(B, NUM_V_HEADS, V_DIM//BV)` — parallelizes V-tile dimension across CTAs
- Block pointers via `tl.make_block_ptr` — TMA-eligible state loads/stores on Blackwell
- Tuned: `num_warps=8, num_stages=4, BLOCK_V=16`

Results

```bash
gdn_decode_qk4_v8_d128_k_last:
  Workload 6700a748...: PASSED | 88.962 µs | 14.21x speedup | abs_err=3.05e-05, rel_err=3.55e-01
  Workload d66ae544...: PASSED | 87.601 µs | 13.30x speedup | abs_err=1.95e-03, rel_err=5.35e-02
  Workload bf115ff9...: PASSED | 89.138 µs | 13.06x speedup | abs_err=7.81e-03, rel_err=5.01e-02
  Workload 30bb1856...: PASSED | 88.477 µs | 14.22x speedup | abs_err=1.91e-05, rel_err=4.38e-02
  Workload 48a5be68...: PASSED | 88.462 µs | 13.84x speedup | abs_err=1.53e-05, rel_err=3.96e-02
  Workload 6f09252c...: PASSED | 88.754 µs | 13.27x speedup | abs_err=3.05e-05, rel_err=2.51e-01
  Workload 1e4e5a87...: PASSED | 87.691 µs | 13.27x speedup | abs_err=3.91e-03, rel_err=7.47e-02
  Workload 8e3e1aa6...: PASSED | 88.311 µs | 13.10x speedup | abs_err=3.91e-03, rel_err=1.53e-01
  Workload 741eb2c4...: PASSED | 90.986 µs | 12.60x speedup | abs_err=2.44e-04, rel_err=4.40e-02
  Workload 562d6431...: PASSED | 89.645 µs | 12.77x speedup | abs_err=2.67e-05, rel_err=1.39e+00
  Workload cf76ded3...: PASSED | 88.794 µs | 13.16x speedup | abs_err=2.29e-05, rel_err=1.35e-02
  Workload 55b3b0a4...: PASSED | 89.581 µs | 12.72x speedup | abs_err=1.53e-05, rel_err=6.74e-01
  Workload 8add0e58...: PASSED | 88.262 µs | 12.86x speedup | abs_err=4.88e-04, rel_err=3.24e-01
  Workload 5727c2b1...: PASSED | 88.820 µs | 12.75x speedup | abs_err=9.77e-04, rel_err=5.72e-02
  Workload cc8d77b2...: PASSED | 88.253 µs | 12.89x speedup | abs_err=6.87e-05, rel_err=1.08e-02
  Workload 3bd97bb8...: PASSED | 87.968 µs | 14.29x speedup | abs_err=1.95e-03, rel_err=5.02e-02
  Workload f1b6043f...: PASSED | 88.372 µs | 13.13x speedup | abs_err=2.86e-05, rel_err=1.07e-02
  Workload 8038c14e...: PASSED | 88.536 µs | 13.10x speedup | abs_err=3.91e-03, rel_err=2.30e-02
  Workload 03436065...: PASSED | 88.720 µs | 13.14x speedup | abs_err=1.95e-03, rel_err=4.50e-01
  Workload c40fb468...: PASSED | 89.082 µs | 13.02x speedup | abs_err=2.44e-04, rel_err=2.53e-01
```
