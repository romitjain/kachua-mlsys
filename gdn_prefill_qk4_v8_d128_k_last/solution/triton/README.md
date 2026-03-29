# Triton GDN Prefill Kernel

This submission holds the correctness-first Triton prefill kernel for
`gdn_prefill_qk4_v8_d128_k_last`.

Design choices:

- One Triton program per `(sequence, value-head, value-tile)`
- Sequential token loop inside each program to preserve the recurrent dependency
- Shared constants with the decode kernel: `qk4`, `v8`, `head_dim=128`, `BV=8`

This is intentionally conservative. It matches the recurrence directly and is a
clean baseline for later optimization.
