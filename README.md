# Kachua MLSys GDN Kernels

This repository contains optimized Gated Delta Network (GDN) kernels for the MLSys 2026 FlashInfer AI Kernel Generation Contest, targeting NVIDIA B200.

There are two active definitions:

| Definition | Active backend | Entry point | Main kernel |
|---|---:|---|---|
| `gdn_prefill_qk4_v8_d128_k_last` | Triton | `kernel.py::launch_gdn` | `gdn_prefill_qk4_v8_d128_k_last/solution/triton/kernel.py` |
| `gdn_decode_qk4_v8_d128_k_last` | CUDA | `kernel.cu::launch_gdn` | `gdn_decode_qk4_v8_d128_k_last/solution/cuda/kernel.cu` |

## Repository Structure

```text
.
├── gdn_prefill_qk4_v8_d128_k_last/
│   ├── config.toml
│   ├── scripts/
│   └── solution/
│       ├── reference_torch_impl.py
│       └── triton/
│           └── kernel.py
├── gdn_decode_qk4_v8_d128_k_last/
│   ├── config.toml
│   ├── scripts/
│   └── solution/
│       ├── reference_torch_impl.py
│       ├── cuda/
│       │   ├── kernel.cu
│       │   └── binding.py
│       └── triton/
│           └── kernel.py
├── docs/
│   ├── program.md
│   ├── submission-report.pdf
│   └── kachua_mlsys26_presentation.pdf
├── EVALUATION.md
├── FAQ.md
└── ORIGINAL_README.md
```

Each definition directory is self-contained. Its `config.toml` selects the backend, entry point, solution name, and contest definition.

## Kernel Summary

### Prefill Triton

The prefill kernel processes multiple prompt tokens using chunked recurrence blocks. It dispatches between three live paths:

- `Direct v2`: one Triton launch; each BV state-row tile recomputes token-side chunk work.
- `Split-WY`: prepass computes reusable chunk metadata, then a consumer applies it across BV state-row tiles.
- `Flat-WY`: many-sequence variant that flattens chunks into records for denser prepass parallelism.

The two key tiling axes are:

```text
CHUNK -> number of tokens grouped into one local recurrence solve
BV    -> number of state/output rows updated by one Triton program
```

### Decode CUDA

The active decode submission is CUDA. Decode handles one token at a time, so the state transition is a rank-1-style update from single-token `q`, `k`, and `v` vectors. The current CUDA path focuses on low-latency single-token execution with vectorized state loads, compact gate/update math, and direct binding through the configured CUDA entry point.

Historical Triton decode kernels and notes remain under `gdn_decode_qk4_v8_d128_k_last/solution/triton/`.

## Setup

Install the FlashInfer benchmark tools and Modal:

```bash
conda create -n fi-bench python=3.12
conda activate fi-bench
pip install flashinfer-bench modal
```

Download the contest dataset:

```bash
git lfs install
# Download the contest traces and definitions
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
export FIB_DATASET_PATH=/path/to/mlsys26-contest
```

For Modal runs:

```bash
modal setup
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/mlsys26-contest
```

## Running

Run commands from the definition directory you want to evaluate.

Prefill:

```bash
cd gdn_prefill_qk4_v8_d128_k_last

# Run the FlashInfer-Bench path on a local GPU
python scripts/run_local.py
# Run the preferred Modal GPU-timing benchmark
modal run scripts/bench_fi_timing_modal.py

# Run the standard Modal benchmark
modal run scripts/run_modal.py
# Run the Modal profiling workflow
modal run scripts/profile_modal.py
```

Decode:

```bash
cd gdn_decode_qk4_v8_d128_k_last

# Run the FlashInfer-Bench path on a local GPU
python scripts/run_local.py
# Run the preferred Modal GPU-timing benchmark
modal run scripts/bench_fi_timing_modal.py

# Run the standard Modal benchmark
modal run scripts/run_modal.py
# Run the Modal profiling workflow
modal run scripts/profile_modal.py
```

`bench_fi_timing_modal.py` is the preferred quick benchmark path because it uses GPU kernel timing that better matches the contest evaluation signal.

## Packing

Pack the active solution for a definition:

```bash
cd gdn_prefill_qk4_v8_d128_k_last
python scripts/pack_solution.py
```

or:

```bash
cd gdn_decode_qk4_v8_d128_k_last
python scripts/pack_solution.py
```

This writes `solution.json` inside that definition directory using its local `config.toml`.

## Useful References

- [docs/program.md](docs/program.md): Auto research workflow
- [docs/submission-report.pdf](docs/submission-report.pdf): Technical writeup.
- [docs/kachua_mlsys26_presentation.pdf](docs/kachua_mlsys26_presentation.pdf): Presentation for MLSys 26.
- [ORIGINAL_README.md](ORIGINAL_README.md): original FlashInfer contest starter README.
