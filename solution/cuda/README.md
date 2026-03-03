# Kernel.cu

We need to read:

1. State (KxV)
2. Key (1xK) and Value (1xV) vectors
3. Query (1xK)
4. Scalars (a, dt_bias, A_log, scale)

and write:

1. Updated state (KxV)
2. Output (1xK)

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
