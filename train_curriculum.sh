#!/usr/bin/env bash
# ==============================================================================
# OceanGo V2 — Automated 3-Stage Curriculum Runner for Linux & macOS
# Stages: Stage 1 (9x9) -> Stage 2 (13x13) -> Stage 3 (19x19)
#
# Defaults (identical to train_curriculum.bat):
#   - Preset  : flagship (18 blocks x 256 channels | 40 -> 45 -> 60 iters)
#   - Compile : --compile enabled (OceanGo.py auto-enables Triton on CUDA and
#               falls back cleanly to fast eager bfloat16 on MPS / non-Triton setups)
#
# Usage:
#   chmod +x train_curriculum.sh
#   ./train_curriculum.sh                     # Flagship (18x256) + --compile
#   ./train_curriculum.sh --preset balanced   # Balanced (15x192 | 30/35/50 iters)
#   ./train_curriculum.sh --no-compile        # Force eager execution
#
# Run detached in background (survives closing terminal / SSH):
#   nohup ./train_curriculum.sh > curriculum_console.log 2>&1 &
#   tail -f curriculum_console.log
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Auto-activate local virtual environment if present and not already active
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi

OS_KERNEL="$(uname -s)"
SAVE_DIR="${SAVE_DIR:-checkpoints}"
COMPILE_FLAG="--compile"
PRESET="flagship"

EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --no-compile)
            COMPILE_FLAG=""
            shift
            ;;
        --compile)
            COMPILE_FLAG="--compile"
            shift
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --preset)
            PRESET="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ "$PRESET" = "balanced" ]; then
    S1_BLOCKS=15; S1_CHANNELS=192; S1_GAMES=500; S1_SIMS=600;  S1_ITERS=30
    S2_GAMES=400; S2_SIMS=600;  S2_ITERS=35
    S3_GAMES=250; S3_SIMS=800;  S3_ITERS=50
else
    S1_BLOCKS=18; S1_CHANNELS=256; S1_GAMES=600; S1_SIMS=800;  S1_ITERS=40
    S2_GAMES=500; S2_SIMS=800;  S2_ITERS=45
    S3_GAMES=300; S3_SIMS=1000; S3_ITERS=60
fi

refresh_ckpt_state() {
    local raw_line=""
    raw_line="$("$PYTHON_BIN" -c '
import contextlib, os, sys
with contextlib.redirect_stdout(sys.stderr):
    from OceanGo import resolve_latest_v2_checkpoint, load_checkpoint_auto, is_state_dict_finite
    save_dir = sys.argv[1]
    p = resolve_latest_v2_checkpoint(save_dir)
    b, it, st = 0, 0, 0
    if p and os.path.isfile(p):
        meta = load_checkpoint_auto(p)
        if meta.get("version") == 2 and is_state_dict_finite(meta.get("state_dict")):
            b = int(meta.get("board_size", 0))
            it = int(meta.get("iteration", 0))
            st = int(meta.get("stage_iter", it))
print(f"CKPT_STATE {b} {it} {st}")
' "$SAVE_DIR" | awk '/^CKPT_STATE /{print $2, $3, $4}' | tail -n 1)" || true

    CKPT_BOARD=0
    CKPT_ITER=0
    CKPT_STAGE_ITER=0
    if [ -n "$raw_line" ]; then
        read -r CKPT_BOARD CKPT_ITER CKPT_STAGE_ITER <<<"$raw_line" || true
    fi
    [[ "${CKPT_BOARD:-}" =~ ^[0-9]+$ ]] || CKPT_BOARD=0
    [[ "${CKPT_ITER:-}" =~ ^[0-9]+$ ]] || CKPT_ITER=0
    [[ "${CKPT_STAGE_ITER:-}" =~ ^[0-9]+$ ]] || CKPT_STAGE_ITER=0
}

echo "======================================================================"
echo "OceanGo V2 Automated 3-Stage Curriculum ($OS_KERNEL)"
echo "Preset      : $PRESET (${S1_BLOCKS} blocks x ${S1_CHANNELS} channels)"
echo "Python      : $(command -v "$PYTHON_BIN")"
echo "Save Dir    : $SAVE_DIR"
echo "Compile     : ${COMPILE_FLAG:-disabled (eager bfloat16)}"
echo "Started At  : $(date)"
echo "======================================================================"

# ------------------------------------------------------------------------------
# Stage 1: 9x9 — Tactical Reading, Captures, Ko & Life-and-Death
# ------------------------------------------------------------------------------
refresh_ckpt_state
if [ "$CKPT_BOARD" -eq 0 ] || { [ "$CKPT_BOARD" -eq 9 ] && [ "$CKPT_STAGE_ITER" -lt "$S1_ITERS" ]; }; then
    echo -e "\n>>> [Stage 1/3] Launching 9x9 Curriculum Stage (Target: $S1_ITERS stage iters)..."
    "$PYTHON_BIN" -u OceanGo.py --train \
        --save-dir "$SAVE_DIR" \
        --board-size 9 \
        --blocks "$S1_BLOCKS" \
        --channels "$S1_CHANNELS" \
        --games "$S1_GAMES" \
        --sims "$S1_SIMS" \
        --iters "$S1_ITERS" \
        --lr 0.01 \
        --replay-ratio 1.5 \
        --eval-interval 5 \
        --eval-games 20 \
        ${COMPILE_FLAG:+$COMPILE_FLAG} \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
else
    echo -e "\n>>> [Stage 1/3] Skipping 9x9 (already at ${CKPT_BOARD}x${CKPT_BOARD}, iter=${CKPT_ITER}, stage_iter=${CKPT_STAGE_ITER})."
fi

# ------------------------------------------------------------------------------
# Stage 2: 13x13 — Joseki, Ladders & Whole-Board Influence
# ------------------------------------------------------------------------------
refresh_ckpt_state
if [ "$CKPT_BOARD" -eq 9 ] || { [ "$CKPT_BOARD" -eq 13 ] && [ "$CKPT_STAGE_ITER" -lt "$S2_ITERS" ]; }; then
    echo -e "\n>>> [Stage 2/3] Launching 13x13 Curriculum Stage (Target: $S2_ITERS stage iters)..."
    "$PYTHON_BIN" -u OceanGo.py --train \
        --save-dir "$SAVE_DIR" \
        --board-size 13 \
        --games "$S2_GAMES" \
        --sims "$S2_SIMS" \
        --iters "$S2_ITERS" \
        --lr 0.01 \
        --replay-ratio 1.5 \
        --eval-interval 5 \
        --eval-games 20 \
        ${COMPILE_FLAG:+$COMPILE_FLAG} \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
else
    echo -e "\n>>> [Stage 2/3] Skipping 13x13 (already at ${CKPT_BOARD}x${CKPT_BOARD}, iter=${CKPT_ITER}, stage_iter=${CKPT_STAGE_ITER})."
fi

# ------------------------------------------------------------------------------
# Stage 3: 19x19 — Full-Board Fuseki, Fighting & Endgame Yose
# ------------------------------------------------------------------------------
refresh_ckpt_state
if [ "$CKPT_BOARD" -eq 13 ] || { [ "$CKPT_BOARD" -eq 19 ] && [ "$CKPT_STAGE_ITER" -lt "$S3_ITERS" ]; }; then
    echo -e "\n>>> [Stage 3/3] Launching 19x19 Curriculum Stage (Target: $S3_ITERS stage iters)..."
    "$PYTHON_BIN" -u OceanGo.py --train \
        --save-dir "$SAVE_DIR" \
        --board-size 19 \
        --games "$S3_GAMES" \
        --sims "$S3_SIMS" \
        --iters "$S3_ITERS" \
        --lr 0.005 \
        --replay-ratio 1.8 \
        --eval-interval 5 \
        --eval-games 20 \
        ${COMPILE_FLAG:+$COMPILE_FLAG} \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
else
    echo -e "\n>>> [Stage 3/3] Skipping 19x19 (already completed: ${CKPT_BOARD}x${CKPT_BOARD}, iter=${CKPT_ITER}, stage_iter=${CKPT_STAGE_ITER})."
fi

echo -e "\n======================================================================"
echo "All 3 Curriculum Stages Complete at $(date)!"
echo "Inspect checkpoints : $PYTHON_BIN policy_print.py --save-dir $SAVE_DIR"
echo "Play vs 19x19 AI    : $PYTHON_BIN OceanGo.py --play --sims 1200 --play-symmetries 8"
echo "======================================================================"
