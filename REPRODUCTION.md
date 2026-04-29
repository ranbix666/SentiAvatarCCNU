# SentiAvatarCCNU — Reproduction Guide

This document is an *operational* companion to `README.md`. It records the
roadblocks we hit while trying to run inference out-of-the-box on a single
consumer GPU (RTX 4090, 24 GB), and gives a step-by-step recipe plus a
unified setup script (`scripts/setup_and_run_demo.sh`) that fixes them.

---

## 1. Known issues / undocumented gotchas

| # | Area | Problem | Fix / Workaround |
|---|------|---------|------------------|
| 1 | `README.md` §Inference | Tells you to run `python scripts/preprocess_data.py …`, but the file actually lives at `motion_generation/scripts/preprocess_data.py`. The top-level `scripts/` directory only contains the four `*.sh` wrappers. | Always invoke `python motion_generation/scripts/preprocess_data.py` (or `cd motion_generation && python scripts/preprocess_data.py`). |
| 2 | Checkpoints | README points to `https://huggingface.co/Chuhaojin/SentiAvatar` but uses `git clone … checkpoints/`. The repo ships a non-empty `checkpoints/` (with `README.md` + sub-folder placeholders), so `git clone` aborts with *"destination path already exists and is not an empty directory"*. | Use `huggingface-cli download Chuhaojin/SentiAvatar --local-dir checkpoints/` (overlays into the existing tree). The setup script does exactly this. |
| 3 | vLLM server | `scripts/start_vllm_server.sh` runs vLLM **in the foreground**. Every downstream command (`run_test.sh`, `run_single_infer.sh`) silently assumes a server is listening on `:8095`. New users start it, then run inference in the *same* shell and get a connection refused. | Launch with `nohup … &` (or `tmux`). The setup script writes the PID to `/tmp/sentiavatar_vllm.pid` and waits for `/health` (or first successful Flask response) before continuing. |
| 4 | vLLM port mismatch | `motion_generation/vllm_server.py` defaults to `--port 8081`. Every shell wrapper and the README example use `8095`. If anyone launches `python vllm_server.py` directly without `--port`, all clients hang. | Always pass `--port 8095` explicitly, or use `scripts/start_vllm_server.sh` which sets it. |
| 5 | `gpu_memory_utilization` | `vllm_server.py` hard-codes `gpu_memory_utilization=0.2`. On a 24 GB card this caps vLLM at ~5 GB, which is plenty for Qwen2-0.5B but co-loading Mask Transformer + RVQVAE + HuBERT in the same GPU still leaves ~15 GB free. Just be aware: the value is **not** a knob you can pass on the CLI. | Single-GPU users can leave it. If you want to put vLLM on a separate GPU, set `GPU_ID` arg #3 of `start_vllm_server.sh` (it sets `CUDA_VISIBLE_DEVICES`). |
| 6 | Offline mode | `scripts/run_single_infer.sh` exports `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1`. If `checkpoints/chinese-hubert-base/` is incomplete (missing `pytorch_model.bin`, ~361 MB) the script will throw a confusing *"Can't load tokenizer"* error rather than re-download. | Verify all 7 checkpoint subfolders are populated before running. The setup script does this with a checklist. |
| 7 | Batch mode dataset | `scripts/run_test.sh` reads from `data/motion_token_data`, `data/audio_*`, `data/text_data/motion2text.json`, `data/wav_data/`, `data/split/test_file_list.txt`. None of this ships with the repo — you must download the **SuSuInterActs** dataset from `https://huggingface.co/datasets/Chuhaojin/SuSuInterActs` and then run `motion_generation/scripts/preprocess_data.py --all`. | For a quick "does it work?" run, use **single-case demo mode** (`scripts/run_single_infer.sh`, no args) — it only needs `examples/demo.wav` which is in-tree. |
| 8 | `data/` directory | The repo has no `data/` folder at all. Running batch evaluation without first creating it produces a path error inside hydra. | The setup script creates the empty skeleton if you opt into batch mode. |
| 9 | Python / CUDA pinning | `requirements.txt` pins `vllm==0.11.0`, `torch==2.8.0`, `xformers==0.0.32.post1`, `nvidia-cu*-cu12==12.8.*`. These wheels need Python 3.10–3.12 and a CUDA-12 driver (`nvidia-smi` should show driver ≥ 545). On older drivers, vLLM segfaults on import. | The setup script checks `nvidia-smi` driver version and Python version up front. |
| 10 | Total disk footprint | All seven checkpoints together are **< 3 GB** (1.1 + 0.28 + 0.75 + 0.05 + 0.36 + 0.0015 + 0.43 GB). Easily fits on one GPU. README hints at this but does not state it explicitly, leading users to assume they need an A100. | Documented here for the record. |

---

## 2. Hardware / software baseline

* GPU: any NVIDIA card with ≥ 8 GB VRAM and CUDA 12 driver (RTX 4090 verified).
* OS: Linux x86_64 (we have not tested macOS/Windows; vLLM does not support them).
* Python: 3.10 (matches the README; 3.11/3.12 also work for the pinned wheels).
* Disk: ~10 GB free (3 GB checkpoints + ~5 GB pip wheels + scratch).

---

## 3. Step-by-step reproduction (manual)

```bash
# 0. Clone
git clone https://github.com/ranbix666/sentiavatarccnu.git
cd sentiavatarccnu

# 1. Environment
conda create -n sentiavatar python=3.10 -y
conda activate sentiavatar
pip install -r requirements.txt

# 2. Download checkpoints into the existing (non-empty) checkpoints/ folder.
#    Do NOT use `git clone … checkpoints/` (will fail — see issue #2).
pip install -U "huggingface_hub[cli]"
huggingface-cli download Chuhaojin/SentiAvatar --local-dir checkpoints/

# 3. Sanity-check the checkpoint tree (all 7 sub-trees must exist & be non-empty).
for d in llm mask_transformer rvqvae/model face_vqvae chinese-hubert-base hubert_kmeans eval_model; do
    test -n "$(ls -A checkpoints/$d 2>/dev/null)" \
      && echo "OK   checkpoints/$d" \
      || echo "MISS checkpoints/$d"
done

# 4. Start the vLLM planner in the background (issue #3).
mkdir -p logs
nohup bash scripts/start_vllm_server.sh checkpoints/llm 8095 0 \
    > logs/vllm.log 2>&1 &
echo $! > /tmp/sentiavatar_vllm.pid

# 5. Wait until the server is reachable. The Flask endpoint is POST /v1/predict;
#    the simplest readiness check is to grep the log for "Running on".
until grep -q "Running on http" logs/vllm.log 2>/dev/null; do sleep 2; done
echo "vLLM ready."

# 6a. Quick demo (no SuSuInterActs needed).
bash scripts/run_single_infer.sh
# → output_demo/<name>.bvh, .json, .wav

# 6b. Batch / test-set mode (requires the SuSuInterActs dataset + preprocessing).
huggingface-cli download Chuhaojin/SuSuInterActs --repo-type dataset --local-dir data/
python motion_generation/scripts/preprocess_data.py --all --device cuda:0
bash scripts/run_test.sh 8095 0
bash scripts/run_eval.sh ./output/reconstructed 0

# 7. Tear down.
kill "$(cat /tmp/sentiavatar_vllm.pid)" && rm -f /tmp/sentiavatar_vllm.pid
```

---

## 4. Unified setup script

A single bash script that performs steps 1–6a (env check, checkpoint
download + verify, vLLM bring-up with readiness wait, demo inference) is
provided at:

```
scripts/setup_and_run_demo.sh
```

Usage:

```bash
# Full path: install deps, download ckpts, start vLLM, run demo.
bash scripts/setup_and_run_demo.sh

# Skip pip install (env already prepared).
SKIP_PIP=1 bash scripts/setup_and_run_demo.sh

# Skip checkpoint download (already on disk).
SKIP_DOWNLOAD=1 bash scripts/setup_and_run_demo.sh

# Just bring up vLLM, do not run inference.
ONLY_SERVER=1 bash scripts/setup_and_run_demo.sh

# Stop the background vLLM started by an earlier run.
bash scripts/setup_and_run_demo.sh --stop
```

The script is idempotent: re-running it will re-use an already-running
vLLM server (detected via `/tmp/sentiavatar_vllm.pid`).
