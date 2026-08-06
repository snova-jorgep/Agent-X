# Agent-X — Setup & Run (FC-SO Testing Suite integration)

Agent-X (https://github.com/mbzuai-oryx/Agent-X) is a **vision-centric agentic
tool-use** benchmark: 828 multimodal tasks across 6 domains (web, surveillance,
driving, sports, math, data). It has two stages — OpenCompass inference against
an AgentLego tool server, then an LLM-as-judge that emits 12 metrics per task.

This directory adds the suite wrappers: [config_agentx.yaml](config_agentx.yaml),
[run_agentx.py](run_agentx.py), [agentx_report.py](agentx_report.py). They inject
provider endpoints into `opencompass/configs/eval_gta_bench.py`, run inference,
run the judge, and aggregate results to a CSV → S3 (`agentx_results` Athena table).

> **Models must be multimodal.** Text-only models cannot solve the tasks — mark
> them `not_available` in `model_mappings`.

## 1. One-time environment (GCP VM `benchmarking-client-dev`)

Agent-X uses conda (not the suite's `uv`/venv), two envs:

```bash
# Tool server env
conda create -n agentlego python=3.11.9 -y && conda activate agentlego
cd tests/agentx_snova/agentlego
pip install -r requirements_all.txt && pip install agentlego && pip install -e .
mim install mmengine && mim install mmcv==2.1.0
# Edit transformers/modeling_utils.py line ~1279: _supports_sdpa = False -> True

# Inference env
conda create -n opencompass python=3.10 -y && conda activate opencompass
cd tests/agentx_snova/agentlego && pip install -e .
cd ../opencompass && pip install -e .
```

We do **NOT** need LMDeploy — models under test are hosted provider APIs
(OpenAI-compatible), not locally-served HF weights.

## 1b. Local setup without conda (macOS / uv)

Verified on macOS (Apple Silicon). Uses `uv` instead of conda — three venvs, all
under `tests/agentx_snova/`. Pins live in [requirements-local/](requirements-local/).

```bash
cd tests/agentx_snova
uv python install 3.10 3.11          # fetch interpreters (no conda/pyenv)

# --- Env 1: tool server (the only env with the heavy vision deps) ---
uv venv --python 3.11 .venv_agentlego
source .venv_agentlego/bin/activate
cd agentlego
uv pip install .                     # NON-editable: agentlego pins setuptools<64, so `-e` fails (no PEP660)
uv pip install -r requirements/optional.txt   # torch + easyocr (arm64 wheels OK)
uv pip install -r requirements/server.txt     # fastapi/typer/uvicorn/makefun — needed to run the server
deactivate; cd ..
# NOTE: skip requirements_all.txt (frozen x86-Linux dump: mkl-*/triton have no arm64 wheels).
# Skip mim/mmcv/mmpretrain too — only needed for VQA tools; OCR uses easyocr. See the tools caveat in §4.

# --- Env 2: inference ---
uv venv --python 3.10 .venv_opencompass
source .venv_opencompass/bin/activate
uv pip install ./agentlego ./opencompass       # both NON-editable
# (download the dataset now — see §2 — then apply the local pins LAST:)
uv pip install -r requirements-local/inference.txt
deactivate

# --- Env 3: judge (openai 0.28, isolated) ---
uv venv --python 3.10 .venv_agentx_judge
source .venv_agentx_judge/bin/activate
uv pip install -r requirements-local/judge.txt
deactivate
```

Why `requirements-local/inference.txt` is needed (and installed **last**): `lagent`
2.2 + the OpenCompass import chain pull transitive versions that break against
current PyPI. The file pins `phx-class-registry==4.1.0` (5.x moved `AutoRegister`),
`griffe<1.0` (removed `griffe.enumerations`), `sentence-transformers>=2.7`
(2.2.x imports the removed `cached_download`), `huggingface-hub<1.0` (transformers
requires it; the dataset download bumps hub past it), plus `importlib_metadata`
and the wrapper deps `boto3`/`pyyaml`/`python-dotenv`.

## 1c. SambaNova cluster (`sc-vnc10`) — conda on shared NFS

Verified on `sc-vnc10.sambanovasystems.com`. Same two conda envs as §1, but conda
lives in the user's scratch space (no system conda on the VNC hosts) and the envs
sit on the shared NFS mount so a GPU node can reuse them:

```bash
CONDA=/import/snvm-sc-scratch2/$USER/miniforge3        # miniforge, not system conda
source "$CONDA/etc/profile.d/conda.sh"

# Keep caches off $HOME — model/pip caches are large and $HOME is quota'd.
# Defer to anything already exported (e.g. a shell rc that already redirects these):
# an unconditional assignment here would override it, and Qwen-VL-Chat's ~20GB would
# be downloaded a second time into a redundant tree.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/import/snvm-sc-scratch2/$USER/.cache}"
export HF_HOME="${HF_HOME:-/import/snvm-sc-scratch2/$USER/hf_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/import/snvm-sc-scratch2/$USER/pip_cache}"
```

EasyOCR (the `OCR` tool) ignores `XDG_CACHE_HOME` and defaults to `~/.EasyOCR`, i.e.
**on** the quota — `agentlego/tools/ocr/ocr.py` builds `easyocr.Reader` without a
`model_storage_directory`, so the env var is the only knob:

```bash
export EASYOCR_MODULE_PATH="${EASYOCR_MODULE_PATH:-/import/snvm-sc-scratch2/$USER/.cache/EasyOCR}"
```

Envs, once built, are `agentlego` (py3.11: torch 2.1.2, mmcv 2.1.0, mmengine,
mmpretrain, easyocr) and `opencompass` (py3.10, `opencompass` installed
**editable** from this repo), plus the judge venv from §1b. Verify with:

```bash
conda activate opencompass
python -c "from opencompass.models.openai_api import OpenAI; import inspect; \
  print(inspect.getsourcefile(OpenAI))"     # MUST print this repo's path, not site-packages
```

If that prints a `site-packages` path the install is non-editable and the vendored
fixes in `opencompass/opencompass/models/openai_api.py` (image handling, timeout,
provider error handling) are silently inactive — reinstall with `pip install -e .`.

**Three things differ from §1/§1b on this host:**

1. **`conda activate` is mandatory — do not call the interpreter by absolute
   path.** `run_agentx.py` spawns a bare `python run.py` subprocess, so `python`
   must be on `PATH`. Running `.../envs/opencompass/bin/python run_agentx.py`
   fails with `[Errno 2] No such file or directory: 'python'` and then, worse,
   falls through to consolidate a *previous* run's output directory.

2. **The judge venv name must match `judge_python` in the config.** §1b creates
   `.venv_agentx_judge` (leading dot); an older layout used `venv_agentx_judge`.
   Both are gitignored. If yours lacks the dot, symlink rather than edit config:
   ```bash
   ln -s venv_agentx_judge .venv_agentx_judge
   ```
   Otherwise inference runs fine and the judge dies on a missing interpreter.

3. **No GPU on the VNC hosts.** Start the tool server with `--device cpu` (§4).
   Only the OCR tool path is practical there, so `tool_accuracy` /
   `toolset_accuracy` will be ~0. For the full tool stack, build the envs here and
   run them on a GPU node — the envs are on shared NFS and were built on RHEL 8
   (glibc 2.28) then executed on RHEL 10 (glibc 2.39), which is the compatible
   direction:
   ```bash
   srun --reservation=vllm-stuff --partition=gpuonly --nodelist=sc3-c128 \
        --gres=gpu:1 --pty bash
   bash tests/agentx_snova/validate_on_gpu.sh
   ```
   See [validate_on_gpu.sh](validate_on_gpu.sh) — it checks CUDA visibility, the
   `mmcv` CUDA ops, and the Qwen-VL-Chat load that backs `ImageDescription` /
   `CountGivenObject` / `RegionAttributeDescription`.

## 2. Dataset

Download from https://huggingface.co/datasets/Tajamul21/Agent-X into
`opencompass/data/agentx_dataset/`. Three JSON files are needed:

```
tests/agentx_snova/opencompass/data/agentx_dataset/
├── dataset.json      # OpenCompass inference input (GTABenchDataset.load reads this)
├── data.json         # judge ground truth (--gt_data_path)
├── toolmeta.json
└── image/  ->  files/   # symlink: dataset.json references image/AgentX_*.jpg
```

The HF repo stores the images in a `files/` folder, but `dataset.json` references
them as `image/…`. Download, then symlink `image → files`:

```bash
cd tests/agentx_snova
source .venv_opencompass/bin/activate
uv pip install -U "huggingface-hub[cli]>=0.34,<1.0"   # constrained so it doesn't clobber transformers
hf download Tajamul21/Agent-X --repo-type dataset --local-dir opencompass/data/agentx_dataset
ln -s files opencompass/data/agentx_dataset/image     # so image/AgentX_*.jpg resolves
```

### Subsets and `limit`

`GTABenchDataset.load()` takes a **directory** and reads `<dir>/dataset.json`,
resolving each task's file paths against that same directory. So a task set is a
*directory*, and `agentx_options.dataset_dir` picks which one:

```yaml
dataset_dir: "data/agentx_dataset"   # the full 828, keyed "0".."827"
dataset_dir: "data/gpu_runnable"     # 726 tasks needing the full tool stack
dataset_dir: "data/cpu_runnable"     # 102 tasks runnable with the OCR-only CPU server
```

`cpu_runnable` and `gpu_runnable` partition the full set. Both keep the original
task keys, so `cpu_runnable` starts at 11, 20, 28, … — the keys are *not*
renumbered, and must not be.

To add a subset, make a directory with your filtered `dataset.json` and a symlink
to the shared images — no copies of the ~2 GB image set:

```bash
cd opencompass/data
mkdir -p cpu_runnable
cp /path/to/your_filtered_tasks.json cpu_runnable/dataset.json
ln -s ../agentx_dataset/image cpu_runnable/image
```

`agentx_options.limit` caps how many of those tasks run (`0` = all). It is applied
through OpenCompass's native row slice — `run_agentx.py` injects
`datasets[0]['reader_cfg']['test_range'] = '[:N]'` — so **no dataset file is ever
copied or rewritten**, and an aborted run cannot corrupt your task set. `limit`
takes the first N rows in file order, so it is a deterministic prefix, not a
random sample.

> Subset task keys must still match the judge ground truth (`data.json`, keyed
> `"0"`, `"1"`, …) or the judge scores against the wrong tasks. Filtering the
> original `dataset.json` preserves those keys; renumbering them breaks the judge.
>
> How the key survives: `GTABenchDataset.load()` copies each task's key into a
> `task_id` column, `AgentInferencer` records it on every prediction, and
> `run_agentx.py::consolidate_predictions()` re-keys the judge input on it, then
> checks every id against `ground_truth` before the judge starts. This is load-
> bearing for subsets — see the note on prediction keys in §5c.

## 3. API keys — `tests/agentx_snova/.env`

See [.env.example](.env.example). For a **SambaNova-only smoke test** you need:

| Key | Needed for | Notes |
|-----|-----------|-------|
| `SAMBANOVA_API_KEY` | model under test | already in the suite |
| `SERPER_API_KEY` | GoogleSearch tool | https://serper.dev (free tier) — skip if avoiding web-search tasks |
| `MATHPIX_APP_ID` / `MATHPIX_APP_KEY` | MathOCR tool | https://mathpix.com — skip if avoiding math tasks |
| `OPENAI_API_KEY` | GPT-4o judge | only if `judge: gpt`. To judge with a non-OpenAI endpoint set `AGENTX_JUDGE_API_BASE` + `AGENTX_JUDGE_MODEL` too — the key var is read regardless of provider. |
| `AWS_*` | S3 upload | standard suite vars |

Cheapest first run: `run_eval: false` (infer only, no judge key) on a small task
subset that avoids Serper/Mathpix.

## 4. Start the AgentLego tool server (env: agentlego)

```bash
conda activate agentlego                 # or: source .venv_agentlego/bin/activate  (local/uv)
set -a; source ../.env; set +a           # exports SERPER/MATHPIX if any tool needs them
cd tests/agentx_snova/agentlego
agentlego-server start --port 16181 --device cpu --no-setup --extra ./benchmark.py OCR --host 0.0.0.0
```

Leave it running (use tmux). Matches `agentx_options.tool_server` in the config.

> **Local/uv tools caveat:** the uv setup (§1b) installs only the OCR tool path
> (easyocr), not the `mmcv`/`mmpretrain` stack. So the server exposes **OCR** (+ the
> agent's `finish`) but not VQA tools like `ImageDescription`. Inference still runs
> and produces full traces, but tool-dependent scores (`tool_accuracy`,
> `toolset_accuracy`) will be low — fine for a plumbing smoke test. For meaningful
> scores install the full tool stack (`mim install mmengine mmcv==2.1.0` + mmpretrain).

## 5. Run (env: opencompass)

```bash
conda activate opencompass       # or: source .venv_opencompass/bin/activate  (local/uv)
cd tests/agentx_snova
python run_agentx.py --dry-run   # prints injected config + commands (API keys masked), runs nothing
python run_agentx.py             # infer -> judge -> CSV -> S3
```

Outputs: `logs/agentx/<ts>/{provider}/{model}/{preds.json,scores.json}` and
`results_<ts>.csv`, uploaded to `s3://.../fc-so-testing-suite/agentx_snova/<ts>/`.

`run_agentx.py` runs the whole flow. To re-run **only** the judge on an existing
`preds.json` (no re-inference), invoke it directly — but export the env first,
since a manual shell won't have `OPENAI_API_KEY` (the runner's `load_env()` does
this automatically in the normal flow):

```bash
set -a; source .env; set +a
.venv_agentx_judge/bin/python evaluation/run_eval_gpt_as_judge.py \
  --save_path <model_dir>/scores.json \
  --gt_data_path opencompass/data/agentx_dataset/data.json \
  --pred_path <model_dir>/preds.json
python agentx_report.py logs/agentx/<ts>          # aggregate scores.json -> results_<ts>.csv
```

> Without AWS creds in `.env`, S3 uploads `[WARN]`-skip and the CSV stays local —
> expected for a local run.

## 5b. Provider caveats

Hosted OpenAI-compatible endpoints are not interchangeable. Each item below was a
real, silent failure — inference "succeeded" while producing empty or unusable
traces.

| Symptom | Cause | Fix |
|---|---|---|
| **Almost no task calls a tool, on any provider, for the same model.** Traces are a single assistant step holding a fluent final answer | ReAct is a plain-text protocol with no tool-call API. With no stop sequence the model writes the WHOLE dialogue in one completion — inventing its own `Response:` lines — and `ReActProtocol.parse()` tests for `Final Answer:` before `Action:`, so every Action it emitted is discarded. Measured on SambaNova/gemma-4-31B tasks 0–7: **0/8 tasks called a tool**; with the stop boundary, **5/8 (10 calls)**. | `agentx_options.stop` must be the ReAct turn boundary (`["Response:", "\nResponse"]`), not a family EOS token. `ReActProtocolFixed` (injected automatically) is the second layer for providers that ignore `stop`. |
| Turns after the first silently do nothing; `final_answer` empty and the trace stops early | `openai_api.py` capped the client-side budget at a hardcoded 4096 tokens, ignoring `max_seq_len`. Once the ReAct history crossed it, `_generate` returned `''` **without issuing a request** — indistinguishable from the model declining to act. | Fixed: `max_seq_len` is now the authority and the skip is logged as an ERROR. Keep `agentx_options.max_seq_len` above `prompt + max_turn * max_out_len` (32768 is ample; gemma-4-31B-it serves 131072). |
| `ARGS_ERROR: invalid json format: {"image": …` with the JSON followed by pages of prose, or a tool invoked with another tool's arguments | Upstream `ReActProtocol.parse()` takes the **last** `Action:` but, because its args regex uses `re.DOTALL`, the **first** `Action Input:` block plus everything after it. | Fixed by `ReActProtocolFixed`, which pairs each action with its own args and cuts at the next `Thought:`/`Action:`/`Response:` marker. |
| Every task ends `"Please follow the format"` + `NoAction`, `final_answer` empty | Reasoning-capable models (gemma-4 on Together/Novita) emit a separate `reasoning` field and leave `content` empty until it finishes. OpenCompass defaults to `max_tokens=512`, entirely consumed by reasoning. | `agentx_options.max_out_len` (default 2048). Raise further if long tasks still come back empty. |
| `400 invalid_image_data` / "Image data could not be decoded" | 110 of 758 dataset images are PNG/WebP named `.jpg`. MIME derived from the extension declares `image/jpeg` over PNG bytes. Lenient providers re-sniff; strict ones (Cerebras) reject. | Already fixed in `openai_api.py` (sniffs magic bytes). Don't revert to `mimetypes.guess_type`. |
| `"extra_body: property 'extra_body' is unsupported"` | LMDeploy KV-session controls were injected into every non-`openai.com` request. Most providers ignore it; Cerebras rejects the request. | Now opt-in: pass `lmdeploy_session_controls=True` in the model dict only when targeting a real LMDeploy server. |
| Task dies with a bare `RuntimeError: Calling OpenAI failed after retrying` and **no** error line | Rate limiting. Cerebras allows as few as **5 requests/minute**, and returns errors as `{"message","type","code"}` at the top level rather than nested under `error`, so the retry logic matched no branch and logged nothing. | `provider_query_per_second` (e.g. `Cerebras: 0.07`). Error handling now logs unrecognised shapes and backs off on `request_quota_exceeded`. |
| `stop` token leaves `` ``` `` glued to the answer | A tokenizer EOS token was put in `agentx_options.stop` — `<|im_end|>` is ChatML (Qwen/InternVL) and wrong for Gemma/Llama. | Keep `stop` to the ReAct turn boundary from the first row; never add a family EOS token to it. `_clean_answer()` strips residual fences. |

Provider rate limits are per-key and vary by tier — check
`x-ratelimit-limit-requests-minute` on a response header before assuming the
default 1 qps is safe.

Before blaming the harness, probe the endpoint directly with the same payload
shape it sends (base64 data URI in an `image_url` content part). Note that a
provider's `/v1/models` listing is the authoritative check for whether a model
exists: `POST /chat/completions` on Fireworks returns an identical `404
NOT_FOUND` for an invalid key *and* an unavailable model, so it cannot
distinguish the two. Models the provider doesn't serve belong in
`model_mappings` as `"not_available"`.

## 5c. Reading a raw prediction file

`run_agentx.py::consolidate_predictions()` converts OpenCompass's on-disk
predictions into the judge's `--pred_path` format. With `infer_mode='every'`
(`configs/datasets/gta_bench.py`), `AgentInferencer.save_multiround_results`
writes each task as `{"gold", "prediction": [[step, …]], "origin_prompt",
"steps": [], "task_id"}` — the reasoning trace is in `prediction` (a list of
turns, each a list of `{role, content|tool_calls}` step dicts) and `steps` is
always `[]`. Accordingly we take `reasoning_steps` from `prediction` and
`final_answer` from the last assistant-content step.

> **The top-level keys in `predictions/<abbr>/Agent-X.json` are row positions,
> not task ids.** A `cpu_runnable` run shows `"0", "1", "2", …` even though the
> task set starts at id 11 — key `"0"` *is* task 11. This looks like the wrong
> dataset loaded; it is not. Read the `task_id` field to get the real id, or
> check which image the trace references.
>
> The positional keys are deliberate: OpenCompass's own eval path does
> `preds[str(i)] for i in range(len(preds))` (`tasks/openicl_eval.py`), so
> re-keying the file would break `run.py --mode eval`. The translation to real
> ids happens in `consolidate_predictions()`, at the judge boundary.

Worth a spot-check on the first run of a new task set: confirm `task_id` is
present, that it matches the ids in your `dataset.json`, and that a
finish-action answer is there. `consolidate_predictions()` raises loudly on an
empty/unparseable result, on duplicate ids (which is how a sharded run would
otherwise silently drop tasks — `SizePartitioner` restarts row numbering in each
`Agent-X_<n>.json`), and on any id missing from `ground_truth`. It warns and
falls back to positional keys only for prediction files written before `task_id`
existed — correct for `agentx_dataset`, wrong for any subset, so re-run
inference rather than trusting that fallback.

## Output & metrics

Each of the 12 metrics is a separate GPT/Qwen-judge call returning
`{'Score': <0–1>, 'Justification': ...}` per task. `agentx_report.py` parses the
`Score`, **averages each metric across all tasks**, and writes one CSV row per
model: `date, provider, model, num_tasks` + the 12 averaged scores (each 0–1).
The CSV feeds the `agentx_results` Athena table; the unified Grafana view surfaces
**`goal_accuracy`** as the cross-benchmark headline (the other 11 stay queryable
in `agentx_results`).

| Metric | Meaning (all 0–1) |
|--------|-------------------|
| **goal_accuracy** ⭐ | Final-answer correctness — cosine similarity to GT (or 0/1 for exact-answer types). **Headline metric.** |
| grounding_accuracy | Per step: reasoning grounded in the actual visual/tool evidence (avg per-step) |
| precision_score | Binary 0/1 — precision of the reasoning vs GT |
| tool_accuracy | Correct tools called vs GT `tool_metadata` |
| toolset_accuracy | F1 over the *set* of tools used vs GT |
| faithfulness_accuracy | Reasoning trace logically faithful to the GT plan |
| step_score | Quality of each reasoning step (avg per-step) |
| context_score | Each step uses available context appropriately (avg per-step) |
| factual_precision | Factual correctness of claims in the reasoning vs GT |
| semantic_accuracy | Semantic match of reasoning + final answer to GT |
| reward_score | Self-correction ability — recognizing and fixing its own mistakes |
| clarity_penalty | Penalty for unclear/verbose reasoning (**higher = worse**, unlike the rest) |

Metric definitions live in `evaluation/multiagent_evaluation.py` (`get_*` fns).
