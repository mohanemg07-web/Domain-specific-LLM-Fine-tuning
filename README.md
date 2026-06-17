# Domain-specific LLM fine-tuning — Mistral-7B QLoRA (Finance)

Instruction fine-tuning of **Mistral-7B-Instruct-v0.3** with **QLoRA** on a
single **A100**, then served **free on CPU** as a 4-bit `Q4_K_M` GGUF via
llama.cpp. Built to be defended in a technical interview: every metric in this
README comes from a real run, and anything not yet measured is left as an
explicit `<!-- TODO: fill from real A100 run -->` marker rather than invented.

> **Honesty note.** This repo was scaffolded and proven end-to-end with a
> **CPU smoke test on a tiny stand-in model** (`sshleifer/tiny-gpt2`). The
> real 7B has **not** been trained or evaluated here — see
> [Status](#status-proven-vs-pending) and [RUN_ON_COLAB.md](RUN_ON_COLAB.md).

## Domain
**Finance.** Dataset: [`gbharti/finance-alpaca`](https://huggingface.co/datasets/gbharti/finance-alpaca)
(~68k instruction/response pairs; deduped then subsampled to 50k, with a
**fixed 1,000-sample held-out eval set** carved out *before* any training).
Finance was chosen because it is clean for a public demo (no
sensitive-advice gating), and its instruction style maps directly onto
Mistral-Instruct chat formatting. The demo still shows an
"educational, not financial advice" line.

## Why these engineering choices (interview crib sheet)
- **QLoRA is deliberate, not a memory necessity.** Mistral-7B fits in higher
  precision on an A100; we use 4-bit NF4 anyway because (a) it **matches the
  4-bit `Q4_K_M` GGUF deployment target**, keeping train-time and serve-time
  precision aligned; (b) it demonstrates the technique the project is about;
  (c) it makes the 24-cell ablation grid cheap and fast.
- **Flash Attention 2 because the A100 is Ampere (SM 8.0).** Enabled via
  `attn_implementation="flash_attention_2"`. A runtime guard
  (`src/config.get_attn_implementation`) falls back to `sdpa` on older CUDA
  and `eager` on CPU — which is exactly what lets the CPU smoke test run
  without FA2.
- **bf16, never fp16.** bf16 is native on Ampere (`bnb_4bit_compute_dtype`
  and training `bf16=True`). The same guard uses fp32 on CPU.
- **4-bit quant is also guarded.** bitsandbytes needs CUDA, so on CPU the
  quant config is `None` and the model loads in full precision — the
  quantization analogue of the FA2 guard.
- **Q4 GGUF + CPU serving keeps hosting free.** The merged model is quantized
  to `Q4_K_M` and served on a free HF Spaces CPU via `llama-cpp-python`.
- **Disconnect-safe.** Colab checkpoints to mounted Google Drive every
  `save_steps`; `resume_from_checkpoint` continues from the latest checkpoint,
  so a disconnect costs at most `save_steps` steps.

## Cost split (honest)
- **Training: paid.** Runs on a paid A100 (Colab compute units).
  Approximate cost: <!-- TODO: fill from real A100 run --> compute units.
- **Serving: free.** HF Spaces **CPU** running the Q4_K_M GGUF.

Training is **not** free; only serving is.

## Repo layout
```
configs/        train.yaml (rank16/alpha32/NF4/bf16/FA2), ablations.yaml (24-cell grid)
src/config.py   model switch (MODEL_ID vs TINY_MODEL) + attn/quant/dtype GPU guards
src/data/       prepare.py — download→chat-template→dedupe→50k subsample→1k holdout→cache (idempotent)
src/train/      sft.py (QLoRA SFT, FA2, bf16, grad-ckpt, resume), ablations.py (24 short diagnostic runs)
src/eval/       evaluate.py — base vs fine-tuned on held-out 1k: BLEU (sacrebleu) + ROUGE-L → metrics.json
src/export/     merge.py (adapter→bf16 merge), to_gguf.py (llama.cpp → Q4_K_M, real sizes)
colab/          train_notebook.ipynb — select A100 → confirm secrets → Run all
spaces/         app.py — Gradio + llama-cpp-python on CPU, measures live tok/s
notebooks/      one_click_inference.ipynb — published CPU inference demo
scripts/        push_checkpoint.py / pull_checkpoint.py — Drive sync helpers
tests/          test_smoke.py — full pipeline, tiny model, CPU, FA2/4-bit disabled via guard
```

## Quickstart (CPU smoke test — what's actually proven here)
```bash
pip install -r requirements-cpu.txt      # CPU subset (no bitsandbytes/flash-attn)
python -m pytest tests/test_smoke.py     # full pipeline on tiny-gpt2, ~CPU
```
On Windows, the test auto-enables UTF-8 mode (`conftest.py`) to dodge a trl
1.5.x template-encoding bug; on Linux/Colab this is a no-op.

## Real A100 run
See **[RUN_ON_COLAB.md](RUN_ON_COLAB.md)** for the exact ordered steps.
The pinned A100 stack (incl. the `flash-attn --no-build-isolation` note) is in
[`requirements.txt`](requirements.txt).

## Results
All numbers below are filled from the real A100 run — none are invented.

### Training (Mistral-7B, A100)
| metric | value |
|---|---|
| peak VRAM (GB) | <!-- TODO: fill from real A100 run --> |
| train runtime (s) | <!-- TODO: fill from real A100 run --> |
| train steps | <!-- TODO: fill from real A100 run --> |
| final train loss | <!-- TODO: fill from real A100 run --> |
| attn implementation (read back from loaded model) | flash_attention_2 (training aborts on Ampere if not active) |

### Evaluation on held-out 1k (base vs fine-tuned)
| metric | base | fine-tuned | delta |
|---|---|---|---|
| BLEU (sacrebleu) | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |
| ROUGE-L | <!-- TODO --> | <!-- TODO --> | <!-- TODO --> |

Source of truth: generated `metrics.json` (gitignored until real).

### Export + serving
| metric | value |
|---|---|
| merged model size | <!-- TODO: fill from real A100 run --> |
| f16 GGUF size | <!-- TODO --> |
| Q4_K_M GGUF size | <!-- TODO --> |
| live CPU tok/s (Space) | <!-- TODO: measured live, do not assume --> |

### Ablations
24 **short diagnostic** runs (rank {8,16,32} × LR {1e-4,2e-4} × scheduler
{cosine, linear, constant, constant_with_warmup}) — these are few-hundred-step
diagnostics to compare loss trajectories, **not** 24 full trainings. The best
config is selected programmatically (lowest eval_loss, tie-break smaller rank).
Comparison table: `ablation_results/comparison_table.md` (generated).

## Status: proven vs pending
**Proven by the CPU smoke test (`pytest tests/test_smoke.py`, 8/8 passing):**
- config + GPU guards: eager/sdpa/FA2 selection, 4-bit disabled on CPU, bf16
  gated to Ampere;
- data pipeline: format → dedupe → subsample → 1k holdout, idempotent cache
  hit, train/eval disjointness (intersection = 0);
- QLoRA SFT on the tiny model: trains, saves adapter, checkpoints, and
  **resumes from checkpoint**;
- ablation grid: runs, writes comparison table, selects best config;
- evaluation: generates base + fine-tuned, computes real BLEU + ROUGE-L,
  writes `metrics.json`;
- merge: produces a merged model with a real on-disk size;
- GGUF export: runs, and **honestly skips** (no invented sizes) when llama.cpp
  isn't present.

**Also proven by the pre-A100 hardening pass:**
- **torch-pin landmine closed**: torch is no longer in `requirements.txt`; the
  Colab notebook pins torch to Colab's own CUDA build (constraints file) and
  hard-aborts if torch became a CPU wheel before training.
- **flash-attn install fixed + honesty**: notebook installs
  `flash-attn==2.7.4.post1 --no-build-isolation`; training reads back the
  **active** attn impl from the loaded model (`attn_implementation_active`) and
  **fails loudly** on an Ampere GPU if FA2 isn't actually active (no silent
  sdpa fallback).
- **real finance-alpaca schema** (streaming dry run, ≤200 rows, no full
  download): confirmed columns `instruction` / `input` / `output` / `text`,
  empty-`input` rows handled, formatting non-empty, split overlap = 0.
- **GGUF f16 path proven on a tiny Llama-arch model** (SmolLM2-135M, mirrors
  Mistral): merge → `convert_hf_to_gguf.py` → real **f16 GGUF = 270,885,376
  bytes (258.34 MB)**. This is the *tiny stand-in*, not the 7B. Loadback
  (llama-cpp-python) and `Q4_K_M` quantize honest-skip locally (no compiler)
  and are exercised on Colab/Linux.

**Pending real A100 execution (you run this — see RUN_ON_COLAB.md):**
- training/evaluating the real **Mistral-7B** and recording its metrics;
- **4-bit NF4 loading** actually running (CUDA-only; guarded out on CPU);
- **FA2 actually running** on the A100 (locally only the guard selection +
  loud-fail enforcement are proven, not FA2 execution);
- the **full 50k** finance-alpaca pipeline (only a ≤200-row slice is proven);
- building the real `Q4_K_M` GGUF and recording file sizes;
- deploying the Space and recording **live** CPU tok/s;
- recording the approximate Colab compute-unit cost.

No large-model metric is reported as achieved. Every `<!-- TODO -->` above is
a real number waiting to be filled from your run.
```bash
python -m src.config        # prints the runtime guard decisions on your box
```
