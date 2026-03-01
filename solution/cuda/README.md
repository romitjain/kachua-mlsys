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
