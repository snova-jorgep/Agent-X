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

## 3. API keys — `tests/agentx_snova/.env`

See [.env.example](.env.example). For a **SambaNova-only smoke test** you need:

| Key | Needed for | Notes |
|-----|-----------|-------|
| `SAMBANOVA_API_KEY` | model under test | already in the suite |
| `SERPER_API_KEY` | GoogleSearch tool | https://serper.dev (free tier) — skip if avoiding web-search tasks |
| `MATHPIX_APP_ID` / `MATHPIX_APP_KEY` | MathOCR tool | https://mathpix.com — skip if avoiding math tasks |
| `OPENAI_API_KEY` | GPT-4o judge | only if `judge: gpt`. Reuse SambaNova as judge to skip. |
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
venv_agentx_judge/bin/python evaluation/run_eval_gpt_as_judge.py \
  --save_path <model_dir>/scores.json \
  --gt_data_path opencompass/data/agentx_dataset/data.json \
  --pred_path <model_dir>/preds.json
python agentx_report.py logs/agentx/<ts>          # aggregate scores.json -> results_<ts>.csv
```

> Without AWS creds in `.env`, S3 uploads `[WARN]`-skip and the CSV stays local —
> expected for a local run.

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

## ⚠️ Verify on first real run

`run_agentx.py::consolidate_predictions()` converts OpenCompass's on-disk
predictions into the judge's `--pred_path` format. The field layout is now
verified against the vendored inferencer: with `infer_mode='every'`
(`configs/datasets/gta_bench.py`), `AgentInferencer.save_multiround_results`
writes each task as `{"gold", "prediction": [[step, …]], "origin_prompt",
"steps": []}` — the reasoning trace is in `prediction` (a list of turns, each a
list of `{role, content|tool_calls}` step dicts) and `steps` is always `[]`.
Accordingly we take `reasoning_steps` from `prediction` and `final_answer` from
the last assistant-content step.

Still worth a spot-check on the first run: inspect a file under
`opencompass/outputs/default/<ts>/predictions/<abbr>/Agent-X.json` and confirm the
task keys match `data.json` (both keyed "0","1",…) and that a finish-action
answer is present. The function raises loudly on an empty/unparseable result
rather than emitting silently-wrong scores.
