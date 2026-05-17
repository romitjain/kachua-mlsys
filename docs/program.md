# Optimizing Gated Delta Network (GDN) decode kernel for NVIDIA B200 GPUs

You are a senior GPU kernel optimization researcher working on writing the fastest kernel for the Gated Delta Network (GDN) for prefill and decode operations on NVIDIA B200 GPUs in the FlashInfer AI Kernel Generation Contest @ MLSys 2026. You are working in autonomous research mode.

## Setup

1. **Start with a tag for the run**: Get the current date and the idea for the experiment campaign you are going to run and this will be the branch tag for the current run in the format `experiment/date/<run_tag>`, like `experiment/14-03/init` or `experiment/14-03/remove-shared-memory`. We will be running multiple experiments on this in succession, learning from the previous runs and improving the kernel.
2. **Create a new branch for the run**: `git checkout -b experiment/14-03/<run_tag>` from the `triton-optimization-speedrun-baseline` branch.
3. **Read the in-scope files**: The repo is small. Read these files for full context (gdn_decode_qk4_v8_d128_k_last contains the decode kernels and gdn_prefill_qk4_v8_d128_k_last contains the prefill kernels)
  1. `scripts/run_modal.py` - Modal script to run the benchmark on cloud B200 GPU
  2. `scripts/profile_modal.py` - Modal script to profile the kernel on cloud B200 GPU
  3. `scripts/bench_fi_timing_modal.py` - Modal script to benchmark the kernel on cloud B200 GPU
  4. `solution/triton/experiment_log.md` - Log of previous experiments, approaches and their results
  5. `solution/triton/results.csv` - Results of previous experiments
  6. `solution/triton/kernel.py` - The kernel implementation
4. **Verify data exists**: Check that `mlsys26-contest/definitions/gdn/gdn_decode_qk4_v8_d128_k_last.json` and `mlsys26-contest/definitions/gdn/gdn_prefill_qk4_v8_d128_k_last.json` exists and has the correct data.
5. **Check results csv is initialized**: Check that `solution/triton/results.csv` has the header and data of previous runs. If it is empty, then this is first run.
6. **Confirm and go**: The setup looks good. You are ready to go.

Once you are ready to go, kickoff the experiment.

## Experimentation

Each experiment runs on a remote B200 GPU via Modal. You will need to write the kernel code, and run `uv run modal run scripts/run_modal.py` to run the benchmark and `uv run modal run scripts/profile_modal.py` to profile the kernel.

TIP - You can run `uv run modal run scripts/bench_fi_timing_modal.py` to benchmark the kernel on cloud B200 GPU along with correctness checks. The Bench Fi Timings is a pure GPU timer that matches the reference repo's Modal CUPTI path rather than the benchmark latency path used by scripts/run_modal.py, so prefer using just this than the profiling or run modal scripts. In the end the raw CUPTI timing results are what we care about.

Each benchmark and profiling run will take not more than a few minutes to complete. You may run both in parallel. After each run, ensure that the modal run we kicked off is completed.

What you CAN do:

- Modify the kernel code in `solution/triton/kernel.py` - This is the only file you edit. Every optimization inside is a fair game as long as we are not reward hacking.
- Modify the `config.toml` file to set the entry point to the new kernel function you are writing, if needed.
- Modify the experiment log in `solution/triton/experiment_log.md` - This is where you document your experiments, approaches and their results.
- Modify the results csv in `solution/triton/results.csv` - This is where you log the results of your experiments.
- Spawn parallel research agents to explore and research and report back to you with their findings.

What you CANNOT do:

- Modify anything under `scripts/` directory. - It is read-only. It contains the fixed packing and benchmarking and other utility scripts.
- Create worktrees. Just use the current directory with a new branch.
- Use Graph Capture as a technique to optimize the kernel because Python dispatch overhead is not a real issue. All we care about is the actual kernel latency.

The goal is simple: get the fastest kernel speedup compared to the reference implementation by understanding the benchmarking and profiling results so that we can win the competition. Everything is fair game: modify the kernel code, look up online on the internet on the best practices, reduce bottleneck, come up with novel kernel maths, use Triton tricks and anything else you can think of to get a speedup utilizing the Triton kernel primitives and the understanding of the B200 GPU architecture and other references online. The only constraint is that the code runs without crashing and gets a speedup compared to the reference implementation and our previous runs.
The first run: Your very first run should always be to establish the Triton baseline, so you will run the current reference Triton kernel as is. Feel free to modify the `config.toml` file to run the baseline.

When comparing, kernel_latency is the more important metric to look at because it is the actual kernel latency, compared to the benchmark_latency or speedup.

## Logging Results

When an experiment is done and you have the benchmark results, you will need to log the benchmark results in the `solution/triton/results.csv` file and in the experiment approach and results in the `solution/triton/experiment_log.md` file. The results should be logged in the following format:

- s_no: serial number of the experiment
- speedup: speedup compared to the reference implementation
- benchmark_latency: latency of the kernel from the benchmark results (in microseconds)
- kernel_latency: latency of the kernel from the profiling results (in microseconds)
- max_abs_error: maximum absolute error
- max_rel_error: maximum relative error
- timestamp: timestamp of the experiment
- short_approach_title: short title of the approach

### The experiment loop

The experiment runs on a dedicated campaign branch (e.g. `experiment/14-03/init` or `experiment/14-03/remove-shared-memory`).

LOOP FOREVER. NEVER STOP. NEVER ASK THE HUMAN.

1. Look at the git state: the current branch/commit we're on
2. Hypothesize an experimental idea based on the experiment log, looking up online, reading papers and other researchers' ideas and your own intuition. Current bottlenecks? What tier of optimization to explore? What worked or failed in the past? Are there combinations of successful approaches to try? Write a brief hypothesis for what to try. Look up online to find more ideas and inspiration for the hypothesis.
3. Tune `solution/triton/kernel.py` with an experimental idea by directly hacking the code.
4. `git commit` to save the changes.
5. Run the experiment: `uv run modal run scripts/run_modal.py` and `uv run modal run scripts/profile_modal.py` (can run in parallel)
6. Read out the results of the benchmark and profiling from the run output.
7. If the benchmark or profiling results are empty or have error messages, the run crashed. Read the modal run output to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
8. Record the results in the `solution/triton/results.csv` file and in the `solution/triton/experiment_log.md` file.
9. If speedup improved (higher), you "advance" the branch, keeping the git commit and move to the next experiment. Log the approach and the results in the log file and update the results.csv file.
10. If speedup is equal or worse, you git reset the kernel code back to where you started, taking a note in the log file of the approach and the results. Do not update the results.csv for equal or worse results.
11. The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).
12. Move to the next experiment (step 1).

Crashes: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

NEVER STOP: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

Also, remember that because of infra instantiation and other runtime randomness, the results of the same experiment may vary slightly within a few percentage points. Do not be too strict with the results.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Optimization Playbook

Work through these tiers roughly in order. Earlier tiers give larger gains with less risk.
Later tiers require more expertise but can unlock the final 10-20%.

### Tier 1: Block Size Tuning

The single most impactful change for most kernels. Block sizes control tile dimensions and
directly affect occupancy, register pressure, and shared memory usage.

**What to try:**

- Sweep BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K through powers of 2: 16, 32, 64, 128, 256.
- For matmul-like kernels, try rectangular tiles (e.g., 128x64 instead of 64x64).
- Larger blocks = more work per thread block = better arithmetic intensity, but higher register pressure.
- Use `num_warps` and `num_stages` as secondary tuning knobs alongside block sizes.

**Typical gains**: 10-50% from finding the right block size vs the default.

### Tier 2: Memory Access Optimization

Once block sizes are tuned, memory is usually the bottleneck.

**Coalescing:**

- Ensure threads in the same warp access consecutive memory addresses.
- For matmul, this means loading along the contiguous dimension (stride-1).
- Transpose one operand if needed to make both loads coalesced.

**Prefetching:**

- Use `tl.prefetch` or software pipelining to overlap memory loads with computation.
- Add `num_stages` to the kernel to enable Triton's built-in software pipelining.
- Typical: `num_stages=3` or `num_stages=4` for matmul.

**L2 Cache Swizzling:**

- Reorder tile indices so neighboring thread blocks access nearby memory.
- Group tiles along the K dimension to maximize L2 cache reuse.

**Shared Memory Bank Conflicts:**

- 32 banks, 4 bytes wide on NVIDIA GPUs. Add 1 element of padding per row.

**Typical gains**: 10-30% from memory optimizations on top of tuned block sizes.

### Tier 3: Compute Optimization

**TF32 and Mixed Precision:**

- Use `tl.dot(a, b, allow_tf32=True)` for matmul accumulation with TF32 inputs.
- Keep accumulators in fp32 for numerical stability.
- Cast results to output dtype only at the end.

**Fused Operations:**

- Fuse elementwise operations (bias add, activation, scaling) into the kernel epilogue.
- Avoid writing intermediate results to global memory.

**Instruction-Level Optimization:**

- Minimize operations in the inner loop. Hoist invariant computations outside.
- Use `tl.where` instead of branches where possible.

**Typical gains**: 5-15% from compute optimizations.

### Tier 4: Advanced Techniques

**Split-K:**

- Decompose the K dimension across multiple thread blocks.
- Helps when M and N are small (not enough parallelism from spatial tiles alone).

**Persistent Kernels:**

- Launch exactly as many thread blocks as there are SMs on the GPU.
- Each block loops over multiple tiles instead of processing just one.
- Eliminates launch overhead and improves L2 cache utilization.

**Autotune:**

- Use `@triton.autotune` with multiple `triton.Config` configurations.
- Let Triton search over block sizes, num_warps, and num_stages.

**Warp Specialization:**

- Assign different warps to different roles (producers vs consumers).

**Register Tiling:**

- Manually control register allocation via constexpr tile sizes.
- Larger register tiles increase ILP but can cause register spilling.

**Typical gains**: 5-20% from advanced techniques, but higher risk.

### Tier 5: Architecture-Specific Optimizations

**H100 (Hopper, SM90):**

- TMA (Tensor Memory Accelerator): hardware-accelerated bulk copies.
- WGMMA (Warp Group Matrix Multiply Accumulate): next-gen tensor core instructions.
- Cluster-level shared memory.

**A100 (Ampere, SM80):**

- Async global-to-shared memory copies (`cp.async`).
- TF32 tensor cores (19.5 TFLOPS).
- Fine-grained structured sparsity (2:4).

**L40S / L4 / RTX (Ada Lovelace / Ampere consumer):**

- Smaller shared memory, fewer SMs. Use smaller block sizes, fewer stages.
- L40S: 142 SMs, good FP16 throughput.
- L4: very memory-bandwidth limited.
- RTX 4090: 128 SMs but consumer-grade memory bandwidth.

**Typical gains**: 5-15% from architecture-specific tuning.

**Blackwell (B200 / GB200, SM100 / CC 10.0):**

- **TCGen05 tensor cores:** new `tcgen05.mma` instructions replace Hopper-style WGMMA as the key tensor-core primitive. For GEMM-heavy kernels, Blackwell SM100 can deliver roughly **2× Hopper throughput** for TF32 / FP16 / BF16 / INT8 tensor-core paths, and **up to 4× Hopper FP8 throughput** for some block-scaled low-precision modes.

- **Extended TMA + new tensor memory:** Blackwell adds Tensor Memory Accelerator extensions plus a dedicated **tensor memory (`tmem`)** space and explicit data-movement paths such as `smem → tmem` and `tmem → rmem`. High-performance kernels should redesign pipelines around bulk asynchronous copies and deeper staging rather than Hopper-style memory pipelines.

- **Cluster Launch Control:** hardware support for **work stealing** by allowing cancellation of not-yet-started thread blocks or clusters. This helps load balance irregular workloads such as grouped GEMM, MoE routing, variable-length sequence inference, and persistent-kernel schedulers.

- **Larger cluster/shared-memory tuning space:** Blackwell keeps distributed shared memory and supports portable **cluster size 8**, with **cluster size 16 available on B200**. CC 10.0 parts expose **228 KB shared memory per SM** and up to **227 KB per thread block**, enabling larger tiles, deeper pipelining, and more aggressive shared-memory staging.

- **Bigger memory hierarchy:** B200 supports **HBM3/HBM3e up to ~180 GB**, and GB200 increases **L2 cache to ~126 MB**. Larger L2 residency improves attention kernels, KV-cache-heavy decoding, and fused pipelines.

- **Multi-GPU scale-up improvements:** fifth-generation NVLink significantly increases peer bandwidth and system scaling. Architecture-specific tuning should include topology-aware partitioning and communication/computation overlap for tensor parallelism and expert parallel inference.

### Tier 6: Kernel-Specific Tricks

**Matrix Multiplication (matmul):**

- Swizzle tile ordering for L2 reuse.
- Epilogue fusion (bias, activation, scaling).
- Split-K for tall-skinny matrices.

**Softmax:**

- Two-pass online softmax (track running max and sum in one pass).
- Multi-row processing: process multiple rows per thread block.

**LayerNorm / RMSNorm:**

- Welford's online algorithm for numerically stable variance.
- Fuse weight and bias application into the kernel.
- Multi-row processing for better occupancy.

**Flash Attention:**

- Online softmax with running statistics.
- Block-sparse patterns for long sequences.
- Causal masking with early termination.

**Cross Entropy:**

- Online log-sum-exp for numerical stability.
- Fuse with label indexing to avoid materializing the full logit tensor.

**Rotary Embeddings (RoPE):**

- Fuse with Q/K projection.
- Vectorized sin/cos computation.
- Precompute and cache frequency tables.


## References

### GDN / DeltaNet Theory

- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) — official paper, combines gating + delta rule into unified architecture
- [GDN Decode math walkthrough](https://veitner.bearblog.dev/gated-delta-net-decoding/) — breaks down the decode state-update math (g, beta, state update) aimed at FlashInfer competition participants
- [DeltaNet pt.1 — Delta rule for error-correcting memory](https://sustcsonglin.github.io/blog/2024/deltanet-1/) — why linear attention fails at retrieval, how delta rule fixes it by erasing old associations before writing new ones
- [DeltaNet pt.2 — Chunkwise parallel form via WY representation](https://sustcsonglin.github.io/blog/2024/deltanet-2/) — how to parallelize the recurrent update across sequence length for GPU training using chunk-level matrix ops
- [DeltaNet pt.3 — Architecture modernization](https://sustcsonglin.github.io/blog/2024/deltanet-3/) — L2 normalization, SiLU activations, short convolutions, hybrid attention variants, scaling to 3B params

### FlashAttention & IO-Aware Kernel Design for specific Hardware

- [FlashAttention repo (v1–v4)](https://github.com/Dao-AILab/flash-attention) — reference implementations across GPU generations; FA4 targets Blackwell with warp-specialized CuTeDSL kernels
- [Flash Attention 4 Blog](https://www.together.ai/blog/flashattention-4) and [Paper](https://arxiv.org/html/2603.05451v1) - Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling
- [Reverse-engineering Flash Attention 4](https://modal.com/blog/reverse-engineer-flash-attention-4) — Modal's deep dive into FA4 internals: warp specialization (5 dedicated warp roles), fast polynomial exp approximations, lazy softmax rescaling (10x fewer corrections). Key insight: Blackwell kernels look like async microservice pipelines
- [FlashAttention from first principles: IO-aware tiling](https://mbrenndoerfer.com/writing/flashattention-algorithm-memory-efficient-gpu-tiling) — interactive walkthrough of the core tiling strategy and why IO-awareness (minimizing HBM round-trips) matters more than FLOP count
- [Flash Attention V4](https://www.together.ai/blog/flashattention-4) — Together's blog post on Flash Attention V4, a state-of-the-art attention mechanism for long-context inference on NVIDIA Blackwell GPUs

### B200 / Blackwell Architecture

- [Microbenchmarking Blackwell Architecture](https://arxiv.org/abs/2512.02189) — in-depth B200 benchmarks: memory subsystem, tensor core throughput, FP precision support, ~1.85x over H200 for training workloads
- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) — official NVIDIA guide for tuning CUDA applications on Blackwell (SM 100)
- [Matrix Multiplication on Blackwell (Modular, 4-part series)](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction) — GPU fundamentals → tensor cores → Blackwell's 5th-gen tcgen05 + tensor memory. Shows the gap from naive (5 TFLOPS) to cuBLAS (1763 TFLOPS) and how to close it

### Kernel Optimization

- [From 11% to 88% peak bandwidth: Custom Triton kernels for LLM inference](https://subhadipmitra.com/blog/2025/triton-kernels-llm-inference/) — practical guide: fused RMSNorm (8x speedup), SwiGLU, INT8 GEMM. Key lesson: measure memory bandwidth not FLOPS for inference kernels, most LLM ops are memory-bound
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) - The official NVIDIA guide index for CUDA programming.
- [Getting Memory-bound Kernels to Speed-of-Light](https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md) - A guide on how to optimize memory-bound kernels for speed-of-light.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) - The official NVIDIA guide for best practices in CUDA C++ programming.
- [Warp Specialization in Triton: Design and Roadmap](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/)
- [Blackwell Programming for the Masses With OpenAI Triton](https://semianalysis.com/wp-content/uploads/2025/03/Blackwell-Programming-for-the-Masses-With-OpenAI-Triton-Phil-Tillet.pdf)
- [Intro to PTX Optimization](https://dhmnr.sh/posts/intro-to-ptx-optimization/)

### GDN / DeltaNet Optimization

- [Official FlashInfer CuteDSL Implementation](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gdn_kernels/gdn_decode_bf16_state.py) - The official implementation of the GDN decode kernel in CuteDSL.
- [A fast submission for the FlashInfer Competition](https://github.com/tomasruizt/flashinfer-competition-codebase/) - A fast submission for the FlashInfer Competition by Tomás Ruiz.
- [FlashKDA implementation](https://github.com/MoonshotAI/FlashKDA) and [cuLA KDA implementation](https://github.com/inclusionAI/cuLA/tree/main/cula/kda) - Flash Kimi Delta Attention — high-performance KDA kernels built on CUTLASS (we can get some ideas from here)
- [Flash Linear Attention implementation](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_rule) - Flash Linear Attention — high-performance Linear Attention kernels built on CUTLASS (we can get some ideas from here)
