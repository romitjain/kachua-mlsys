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
