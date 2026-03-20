#!/usr/bin/env bash
set -euo pipefail

# Batch evaluation: benchmark all kernel versions on B200 via Modal.
# Auto-discovers archived kernels from archive/{cuda,triton}/ and
# benchmarks the current solution/ versions last.
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

# Archived CUDA kernels
for f in archive/cuda/gdn_*.cu; do
    [[ -f "$f" ]] || continue
    stem=$(basename "$f" .cu)
    label="cuda_${stem#gdn_}"
    cp "$f" solution/cuda/kernel.cu
    write_config "cuda" "kernel.cu::launch_gdn" "gdn-decode-${label}" "torch"
    run_kernel "$label"
done

# Current CUDA kernel
git checkout -- solution/cuda/kernel.cu
write_config "cuda" "kernel.cu::launch_gdn" "gdn-decode-cuda-current" "torch"
run_kernel "cuda_current"

# Archived Triton kernels
for f in archive/triton/kernel_*.py; do
    [[ -f "$f" ]] || continue
    stem=$(basename "$f" .py)
    label="triton_${stem#kernel_}"
    cp "$f" solution/triton/kernel.py
    write_config "triton" "kernel.py::launch_gdn" "gdn-decode-${label}"
    run_kernel "$label"
done

# Current Triton kernel
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
