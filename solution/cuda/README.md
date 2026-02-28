# Kernel.cu

Every kernel needs to read:

1. State (KxV)
2. Key (1xK) and Value (1xV) vectors
3. Query (1xK)

and write:

1. Updated state (KxV)
2. Output (1xK)

## Kernel 1

- Across number of tokens (B*T)
- T == 1
- Each block for one individual thread
