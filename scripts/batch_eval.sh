#!/usr/bin/env bash
set -euo pipefail

# Batch evaluation: benchmark all kernel versions on B200 via Modal.
# Copies archived kernels into solution/ temporarily; restores via git checkout on exit.
#
# Usage: bash scripts/batch_eval.sh

cd "$(git rev-parse --show-toplevel)"

RESULTS_DIR="results/batch_eval"
BENCH_DIR="$RESULTS_DIR/benchmarks"

mkdir -p "$BENCH_DIR"

cleanup() {
    echo ""
    echo "==> Restoring original files..."
    git checkout -- config.toml solution/cuda/kernel.cu solution/cuda/binding.py solution/triton/kernel.py 2>/dev/null || true
    echo "==> Done."
}
trap cleanup EXIT

# ---------- helpers ----------

write_config() {
    local language="$1" entry_point="$2" name="$3" binding="${4:-}"
    local binding_line=""
    if [[ -n "$binding" ]]; then
        binding_line="binding = \"${binding}\""
    fi
    cat > config.toml <<EOF
[solution]
name = "${name}"
definition = "gdn_decode_qk4_v8_d128_k_last"
author = "kachua"

[build]
language = "${language}"
${binding_line}
entry_point = "${entry_point}"
EOF
}

run_kernel() {
    local kernel_id="$1"
    echo ""
    echo "============================================"
    echo "  [$kernel_id] Benchmark"
    echo "============================================"

    echo "--- [$kernel_id] benchmark ---"
    uv run modal run scripts/run_modal.py 2>&1 | tee "$BENCH_DIR/${kernel_id}.txt" || true
}

# ---------- main ----------

# --- CUDA v1 ---
cp archive/cuda/gdn_v1.cu solution/cuda/kernel.cu
write_config "cuda" "kernel.cu::launch_gdn" "gdn-decode-cuda-v1" "torch"
run_kernel "cuda_v1"

# --- CUDA v2 ---
cp archive/cuda/gdn_v2.cu solution/cuda/kernel.cu
write_config "cuda" "kernel.cu::launch_gdn" "gdn-decode-cuda-v2" "torch"
run_kernel "cuda_v2"

# --- CUDA v3 ---
git checkout -- solution/cuda/kernel.cu
write_config "cuda" "kernel.cu::launch_gdn" "gdn-decode-cuda-v3" "torch"
run_kernel "cuda_v3"

# --- Triton v1 ---
cp archive/triton/kernel_v1.py solution/triton/kernel.py
write_config "triton" "kernel.py::launch_gdn" "gdn-decode-triton-v1"
run_kernel "triton_v1"

# --- Triton current ---
git checkout -- solution/triton/kernel.py
write_config "triton" "kernel.py::launch_gdn" "gdn-decode-triton-current"
run_kernel "triton_current"

# ---------- summary ----------
echo ""
echo "============================================"
echo "  Batch evaluation complete"
echo "============================================"
echo ""
echo "Benchmarks:"
ls -1 "$BENCH_DIR/" 2>/dev/null || echo "  (none)"
