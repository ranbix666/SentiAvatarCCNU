#!/usr/bin/env bash
# ============================================================
# SentiAvatarCCNU — unified setup + smoke-test script.
#
# Performs, in order:
#   1. Sanity check (Python version, CUDA driver).
#   2. pip install -r requirements.txt          (skip with SKIP_PIP=1)
#   3. Download checkpoints from HuggingFace    (skip with SKIP_DOWNLOAD=1)
#   4. Verify all 7 checkpoint sub-trees exist.
#   5. Start the vLLM Qwen2-0.5B planner on :8095 in the background,
#      and wait until it is ready.
#   6. Run the built-in single-case demo (examples/demo.wav).
#
# Env knobs:
#   SKIP_PIP=1        skip pip install
#   SKIP_DOWNLOAD=1   skip HuggingFace checkpoint download
#   ONLY_SERVER=1     bring up vLLM only, no inference
#   GPU_ID=0          CUDA device for vLLM (default 0)
#   VLLM_PORT=8095    port for vLLM Flask server
#
# Special args:
#   --stop            kill the background vLLM started by an earlier run
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

VLLM_PORT="${VLLM_PORT:-8095}"
GPU_ID="${GPU_ID:-0}"
PID_FILE="/tmp/sentiavatar_vllm.pid"
LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/vllm.log"

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn ]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[fail ]\033[0m %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------
# --stop: tear down a previously-launched background server
# ------------------------------------------------------------
if [[ "${1:-}" == "--stop" ]]; then
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        log "Stopping vLLM (pid $(cat "${PID_FILE}"))"
        kill "$(cat "${PID_FILE}")" || true
        rm -f "${PID_FILE}"
    else
        warn "No running vLLM found (pid file missing or stale)."
        rm -f "${PID_FILE}"
    fi
    exit 0
fi

# ------------------------------------------------------------
# 1. Sanity checks
# ------------------------------------------------------------
log "Checking Python and CUDA driver…"
PY_VER="$(python -c 'import sys; print("%d.%d"%sys.version_info[:2])')"
case "${PY_VER}" in
    3.10|3.11|3.12) log "Python ${PY_VER} OK" ;;
    *) die "Python ${PY_VER} not supported. Use 3.10–3.12 (vllm 0.11 / torch 2.8 wheels)." ;;
esac

if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi not found. A CUDA-12 capable NVIDIA GPU is required."
fi
DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
if (( DRV < 545 )); then
    warn "NVIDIA driver ${DRV}.x detected; vllm 0.11 / cu12 wheels expect >= 545. Continuing anyway."
else
    log "NVIDIA driver ${DRV}.x OK"
fi

# ------------------------------------------------------------
# 2. pip install
# ------------------------------------------------------------
if [[ "${SKIP_PIP:-0}" != "1" ]]; then
    log "Installing pip requirements (set SKIP_PIP=1 to skip)…"
    pip install -r requirements.txt
    pip install -U "huggingface_hub[cli]"
else
    log "SKIP_PIP=1 — not installing requirements."
fi

# ------------------------------------------------------------
# 3. Download checkpoints
# ------------------------------------------------------------
if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
    log "Downloading checkpoints from HuggingFace into ./checkpoints …"
    # Using huggingface-cli (NOT git clone) because checkpoints/ is non-empty
    # in-tree (README + sub-folder placeholders) and `git clone` would refuse.
    huggingface-cli download Chuhaojin/SentiAvatar --local-dir checkpoints/
else
    log "SKIP_DOWNLOAD=1 — not downloading checkpoints."
fi

# ------------------------------------------------------------
# 4. Verify checkpoint tree
# ------------------------------------------------------------
log "Verifying checkpoint layout…"
required=(
    "llm/model.safetensors"
    "mask_transformer/model.safetensors"
    "rvqvae/model/epoch_30.pth"
    "face_vqvae/pytorch_model_face_fad2cl_260116_codesize2048_codelength512.bin"
    "chinese-hubert-base/pytorch_model.bin"
    "hubert_kmeans/model.mdl"
    "eval_model/best_model.pt"
)
missing=0
for f in "${required[@]}"; do
    if [[ -s "checkpoints/${f}" ]]; then
        log "  OK   checkpoints/${f}"
    else
        warn "  MISS checkpoints/${f}"
        missing=$((missing + 1))
    fi
done
(( missing == 0 )) || die "${missing} checkpoint file(s) missing. Re-run without SKIP_DOWNLOAD=1."

# ------------------------------------------------------------
# 5. vLLM server (background) with readiness wait
# ------------------------------------------------------------
mkdir -p "${LOG_DIR}"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    log "Re-using existing vLLM server (pid $(cat "${PID_FILE}"))."
else
    log "Starting vLLM on port ${VLLM_PORT}, GPU ${GPU_ID}, log → ${LOG_FILE}"
    nohup bash scripts/start_vllm_server.sh \
        "${PROJECT_DIR}/checkpoints/llm" "${VLLM_PORT}" "${GPU_ID}" \
        > "${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}"
    log "vLLM pid $(cat "${PID_FILE}") — waiting for readiness (this can take 60–120 s)…"
fi

# Readiness: poll the Flask /health endpoint that vllm_server.py exposes.
# Cap the wait at ~5 minutes so a broken start aborts cleanly.
deadline=$(( $(date +%s) + 300 ))
until curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        die "vLLM process died during startup. See ${LOG_FILE}."
    fi
    if (( $(date +%s) > deadline )); then
        die "vLLM did not become ready within 5 minutes. See ${LOG_FILE}."
    fi
    sleep 2
done
log "vLLM is ready on :${VLLM_PORT}."

# ------------------------------------------------------------
# 6. Smoke-test inference
# ------------------------------------------------------------
if [[ "${ONLY_SERVER:-0}" == "1" ]]; then
    log "ONLY_SERVER=1 — leaving vLLM running, skipping demo. Stop later with: bash $0 --stop"
    exit 0
fi

log "Running single-case demo (examples/demo.wav)…"
CUDA_VISIBLE_DEVICES="${GPU_ID}" bash scripts/run_single_infer.sh

log "Done. Demo outputs are in: ${PROJECT_DIR}/output_demo"
log "vLLM is still running (pid $(cat "${PID_FILE}")). Stop it with: bash $0 --stop"
