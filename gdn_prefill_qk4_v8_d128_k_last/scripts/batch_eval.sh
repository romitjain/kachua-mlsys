#!/usr/bin/env bash
set -euo pipefail

# Batch evaluation: benchmark + profile the prefill Triton kernel on B200 via Modal.
#
# Usage: bash scripts/batch_eval.sh [--no-profile]

cd "$(dirname "$0")/.."

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
    git checkout -- config.toml solution/triton/kernel.py 2>/dev/null || true
    echo "==> Done."
}
trap cleanup EXIT

# ---------- helpers ----------

write_config() {
    local entry_point="$1" name="$2"
    cat > config.toml <<EOF
[solution]
name = "${name}"
definition = "gdn_prefill_qk4_v8_d128_k_last"
author = "kachua"

[build]
language = "triton"
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

# Current Triton prefill kernel
write_config "kernel.py::launch_gdn" "gdn-prefill-triton-current"
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
