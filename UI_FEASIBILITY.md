# SentiAvatarCCNU — Interactive Web-UI Feasibility Assessment

**Audience:** advisor / project lead
**Question:** the project page advertises a polished demo video, but the
public repo does not behave like
[OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat)
(no `bat` / shell launcher that opens a browser tab and lets you talk to
the avatar over `localhost`). Is that true, and how much work is it to
close the gap?

---

## 1. What the repo actually ships

A static survey of the codebase (`grep -r` over `*.py / *.html / *.sh /
*.md`) confirms that no browser-facing or streaming-conversation code
exists in this repository.

| Layer | Status in repo | Evidence |
|---|---|---|
| Motion-token planner LLM service | **Present** (internal-only Flask) | `motion_generation/vllm_server.py` exposes `POST /text_to_motion` for the *motion-token* head of Qwen2-0.5B. It is not a chat model. |
| File-in / file-out inference | **Present** | `single_case_infer.py`, `pipeline_infer.py`, `reconstruct_from_tokens.py` consume a `.wav` + an action tag string and write `.bvh` + `.json` + `.wav` to disk. |
| BVH viewer for Blender | **Present** | `tools/visualize_motion.py`. |
| Browser front-end | **Absent** | no `*.html`, no `static/`, no `templates/`. `gradio==6.5.0` is in `requirements.txt` but `grep -r gradio` returns zero usages — it is a vestigial pin. |
| ASR (speech-to-text) | **Absent** | no FunASR / Whisper / SenseVoice import anywhere. |
| Dialogue LLM | **Absent** | the in-tree Qwen2-0.5B is fine-tuned to emit motion tokens, *not* dialogue. |
| TTS (text-to-speech) | **Absent** | no CosyVoice / EdgeTTS / GPT-SoVITS import. |
| Streaming pipeline | **Absent** | HuBERT extraction, planning, and Mask-Transformer infill all run on a complete audio clip end-to-end (offline). |
| WebRTC / WebSocket transport | **Absent** | only a Flask `POST` endpoint exists. |
| 3D character mesh / rig (the "SuSu" avatar in the demo video) | **Absent** | only motion data is released; the art assets are non-commercial-licensed and not in the repo. |

So the released code is the **motion-generation slice only** — the
"plan-then-infill" body of the paper. The end-to-end *interactive
avatar* shown in the demo video is built from additional, unreleased,
glue.

## 2. Comparison to OpenAvatarChat

OpenAvatarChat is a full duplex voice → avatar pipeline with these
ingredients:

```
mic ──► VAD ──► ASR ──► chat-LLM ──► TTS ──► lip-sync + body motion
                                                       │
                                              browser renderer
                                                (Three.js / VRM,
                                                 WebRTC transport)
```

Of those eight components, SentiAvatarCCNU provides exactly **one**:
the body-motion model (and a partial face-animation model via
`face_vqvae` + `susu_face_speech_align.py`). Everything else has to be
brought in.

## 3. Effort estimate to close the gap

We split the work into three tiers ordered by ambition. Estimates are
for one engineer who is comfortable with Python + Three.js, starting
from a working `scripts/setup_and_run_demo.sh` install.

### Tier 1 — Gradio "drop a wav, get a BVH" demo
**Effort: ½ – 1 day, ~150 LOC.**

Wrap `single_case_infer.run()` in a Gradio page with two inputs (audio
upload + action-tag textbox) and one file-download output. The
`gradio` dependency is already installed. Useful for quick eyeballing
in a browser; **not** an "interactive avatar."

**Risk:** essentially none.

### Tier 2 — Static motion preview in the browser
**Effort: 3 – 5 days.**

Add a Three.js (or `@pixiv/three-vrm`) page that loads the generated
BVH and plays it side-by-side with the audio file. Friction:

* The skeleton is non-standard: 63 joints (25 body + 20 + 20 hands) in
  6D rotation. The stock `THREE.BVHLoader` won't accept it directly;
  you must bake it into a Mixamo-compatible BVH first. The Python
  helper in `tools/visualize_motion.py` already does most of this
  conversion — extending it is a few hours of work.
* The accompanying `*.json` is described in code comments as "UE
  engine format" but is otherwise undocumented; inspect before
  relying on it.
* Without the SuSu mesh you'll be driving a generic stick figure or
  a free VRM avatar. Visually that gets you ~80% of the demo video's
  impact.

**Risk:** low. Worst case you ship a generic-avatar viewer.

### Tier 3 — OpenAvatarChat-style realtime conversational avatar
**Effort: 3 – 6 weeks.**

This is the genuinely large item. The missing pieces are:

| # | Component | Difficulty | Notes |
|---|---|---|---|
| 1 | ASR | Easy | FunASR or Whisper-large-v3-zh, off-the-shelf. |
| 2 | Dialogue LLM | Easy | Any chat model (e.g. Qwen2-7B-Instruct) — its text output is what you feed into the existing motion planner. The in-tree Qwen2-0.5B is **not** a chat model. |
| 3 | TTS | Easy | CosyVoice / GPT-SoVITS / EdgeTTS, 16 kHz mono Chinese to match HuBERT's expected sample rate. |
| 4 | **Streaming pipeline** | **Hard** | The current pipeline is *offline*: HuBERT runs over the full clip, then planning + infill happen end-to-end. Realtime requires chunked HuBERT, sliding-window planning, and a lookahead strategy for the Mask Transformer. This is the engineering core of the work and has no shortcut. |
| 5 | Face animation glue | Medium | `face_vqvae` + `susu_face_speech_align.py` emit 51-dim ARKit blendshapes; the renderer must consume them. |
| 6 | Browser renderer + transport | Medium | Three.js scene + WebRTC (LiveKit is the OpenAvatarChat reference). Several open-source skeletons exist. |
| 7 | **3D character asset** | **Blocker** | The SuSu mesh / rig / textures are not in the repo and are licensed non-commercial. Without them, even a flawless renderer drives a generic avatar. |

**Risk:** medium-high. Items 4 and 7 dominate. Item 4 is solvable but
will take an iteration cycle to validate (latency / smoothness
trade-offs); item 7 is a sourcing problem, not an engineering one.

## 4. Recommendation for the advisor

* If the goal is a **classroom / paper-defense demo**: do Tier 2.
  ~1 week of work, browser-renderable, visually convincing with a
  free VRM character. No new ML training required.
* If the goal is **product parity with OpenAvatarChat**: budget Tier 3
  (3–6 weeks) and either (a) license the SuSu art assets from the
  authors, or (b) use a generic VRM character and accept the visual
  delta. The streaming refactor (item 4) is the only piece that
  requires touching the model code itself.
* The fact that the released repo is "motion generation only" is
  consistent with the paper's scope — the paper claims a method for
  motion generation, not an end-to-end product. The demo video
  conflates the research artifact with the surrounding (unreleased)
  product stack.
