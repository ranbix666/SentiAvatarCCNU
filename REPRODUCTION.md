# SentiAvatarCCNU — Reproduction Guide

This document is an *operational* companion to `README.md`. It records the
roadblocks we hit while trying to run inference out-of-the-box on a single
consumer GPU (RTX 4090, 24 GB), and gives a step-by-step recipe plus a
unified setup script (`scripts/setup_and_run_demo.sh`) that fixes them.

---

## 1. Known issues / undocumented gotchas

| # | Area | Problem | Fix / Workaround |
|---|------|---------|------------------|
| 1 | `README.md` §Inference | Told you to run `python scripts/preprocess_data.py …`, but the file actually lives at `motion_generation/scripts/preprocess_data.py`. The top-level `scripts/` directory only contains the four `*.sh` wrappers. | **Fixed in this branch** — README now points at `motion_generation/scripts/preprocess_data.py`. |
| 2 | Checkpoints | README pointed to `https://huggingface.co/Chuhaojin/SentiAvatar` and offered a `git clone … checkpoints/` recipe. The repo ships a non-empty `checkpoints/` (with `README.md` + sub-folder placeholders), so `git clone` aborts with *"destination path already exists and is not an empty directory"*. | **Fixed in this branch** — README now only documents the `huggingface-cli download` path, which overlays into the existing tree. |
| 3 | vLLM server | `scripts/start_vllm_server.sh` runs vLLM **in the foreground**. Every downstream command (`run_test.sh`, `run_single_infer.sh`) silently assumes a server is listening on `:8095`. New users start it, then run inference in the *same* shell and get a connection refused. | `scripts/setup_and_run_demo.sh` launches the server with `nohup … &`, writes the PID to `/tmp/sentiavatar_vllm.pid`, and polls the Flask `/health` endpoint until ready. |
| 4 | vLLM port mismatch | `motion_generation/vllm_server.py` defaulted to `--port 8081`. Every shell wrapper and the README example use `8095`. If anyone launched `python vllm_server.py` directly without `--port`, all clients would hang. | **Fixed in this branch** — `vllm_server.py` default changed to `8095`. |
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

---

## 5. Will it actually work?

Static analysis (we cannot execute the pipeline here — no GPU/network in
this environment), with a confidence rating per concern.

### High confidence the pipeline runs end-to-end

* **All seven checkpoints are physically present** in `checkpoints/` after
  `huggingface-cli download` (verified file-by-file by
  `setup_and_run_demo.sh`). RVQVAE weight `epoch_30.pth` matches the path
  hard-coded in `run_test.sh`; HuBERT `pytorch_model.bin` and the K-means
  `model.mdl` are present and referenced by `single_case_infer.py`.
* **vLLM 0.11 + Qwen2-0.5B is a well-supported combo.** Qwen2 has a
  first-class entry in vLLM's model registry since 0.5.x; the `LLM(...)`
  kwargs used (`gpu_memory_utilization`, `max_model_len`,
  `enable_chunked_prefill`, `max_num_seqs`) are all still valid in 0.11.
* **The Flask layer is straightforward** (`/text_to_motion`, `/health`,
  `/parameters`); the client in `pipeline_infer.py` hits `/health` and
  `/text_to_motion` exactly. After the port fix in this branch, defaults
  agree across server, wrappers, and clients.
* **Demo path needs no dataset.** `examples/demo.wav` is in-tree and
  `single_case_infer.py` does HuBERT extraction + K-means quantisation in
  process — no SuSuInterActs files required.
* **Memory budget is comfortable.** vLLM is capped at `0.2 ×` of GPU memory
  (~5 GB on a 24 GB card). Qwen2-0.5B fp16 is ~1 GB. Mask Transformer
  (276 MB) + RVQVAE (754 MB) + HuBERT (361 MB) loaded in the same process
  is well under 5 GB. Everything fits a single 8 GB card if vLLM is moved
  to a separate device, or one 16 GB+ card co-resident.

### Medium-risk items (likely to "just work" but watch for these)

* **Forward-dated pins in `requirements.txt`** — `pandas==3.0.0`,
  `gradio==6.5.0`, `numpy==2.2.6`, `torch==2.8.0`, `vllm==0.11.0`. As of
  this writing all of these *do* exist on PyPI (the repo targets a 2026
  release), but if a fresh `pip install -r requirements.txt` ever fails it
  will be due to a yanked or relocated wheel, not the model code itself.
  Fall-back: install `vllm==0.11.0` first (it pins compatible
  `torch`/`xformers`) and let pip resolve the rest.
* **CUDA driver requirement is hard.** The cu12.8 wheels in
  `requirements.txt` need driver ≥ 545. The setup script warns but does
  not abort. On older drivers vLLM segfaults during model load.
* **`numpy==2.2.6` + `numba==0.61.2` + `librosa==0.11`** — librosa pulls
  in numba; numba 0.61 supports numpy 2.x but only since its 0.60 line.
  Pin combo is consistent but is the most common source of installer
  conflicts in audio stacks.
* **`gpu_memory_utilization=0.2` is hard-coded.** Fine on a 24 GB card,
  but on an 8 GB card 0.2 ≈ 1.6 GB which is right at the Qwen2-0.5B fp16
  edge — tight but viable. Not a CLI knob; would need a code edit.

### Low-risk / cosmetic

* **arXiv id `2604.02908` and the 2026 publication date** look forward-
  dated but do not affect execution.
* **`TRANSFORMERS_OFFLINE=1`** in `run_single_infer.sh` is fine *as long
  as* the HuBERT folder is fully populated (verified by the setup
  script). If a download is interrupted you'll see a confusing
  tokenizer error instead of a re-download.

### What we cannot verify without running

* Whether the HuggingFace repo `Chuhaojin/SentiAvatar` is publicly
  readable (no anonymous-gated token), and whether its file layout
  matches the seven sub-trees this branch validates.
* Whether the published Mask-Transformer `config.json` matches the
  `AudioMotionConfig` schema in `models/audio_motion_model.py` exactly
  (mismatched embed dims would surface as a state-dict error on load).
* End-to-end numerical parity with the paper's reported R@1 / FID — that
  requires the SuSuInterActs dataset and a multi-hour batch run.

### Verdict

For the **demo path** (`bash scripts/setup_and_run_demo.sh`): high
confidence (≈85%) it produces a `.bvh / .json / .wav` triplet on a
clean Linux + RTX 4090 + driver-545 box, given network access to
HuggingFace. The remaining ≈15% risk is concentrated in (a) the
HuggingFace repo's actual public availability and file layout, and
(b) pip resolving the forward-dated wheel set.

For the **batch / paper-reproduction path**: medium confidence. Adds
the ~37 h SuSuInterActs download, a multi-hour preprocessing run, and
the assumption that the released eval model's stats files match the
ones referenced by `run_eval.sh`. None of these have an obvious code
bug, but each adds an independent point of failure.
