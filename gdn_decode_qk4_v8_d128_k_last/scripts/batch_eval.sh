#!/usr/bin/env bash
set -euo pipefail

# Batch evaluation: benchmark + profile all kernel versions on B200 via Modal.
# Auto-discovers archived kernels from archive/{cuda,triton}/ and
# runs the current solution/ versions last.
#
# Usage: bash scripts/batch_eval.sh [--no-profile]

cd "$(git rev-parse --show-toplevel)"

PROFILE=true
if [[ "${1:-}" == "--no-profile" ]]; then
    PROFILE=false
fi

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
    echo "  [$kernel_id] Benchmark + Profile"
    echo "============================================"

    uv run modal run scripts/run_modal.py 2>&1 | tee "$BENCH_DIR/${kernel_id}.txt" &
    local bench_pid=$!

    if [[ "$PROFILE" == true ]]; then
        uv run modal run scripts/profile_modal.py 2>&1 | tee "$BENCH_DIR/${kernel_id}_profile.txt" &
        local profile_pid=$!
    fi

    wait "$bench_pid" || true
    if [[ "$PROFILE" == true ]]; then
        wait "$profile_pid" || true
    fi
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
for f in "$BENCH_DIR"/*.txt; do
    [[ -f "$f" ]] && [[ "$f" != *_profile.txt ]] && echo "  $(basename "$f")"
done
echo ""
if [[ "$PROFILE" == true ]]; then
    echo "Profiles:"
    for f in "$BENCH_DIR"/*_profile.txt; do
        [[ -f "$f" ]] && echo "  $(basename "$f")"
    done
fi
