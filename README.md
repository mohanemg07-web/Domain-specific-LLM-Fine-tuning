# Domain-specific LLM fine-tuning — Mistral-7B QLoRA (Finance)

Instruction fine-tuning of **Mistral-7B-Instruct-v0.3** with **QLoRA** on a
single **A100**, then served **free on CPU** as a 4-bit `Q4_K_M` GGUF via
llama.cpp. Built to be defended in a technical interview: every metric in this
README comes from the real end-to-end run below — nothing is invented, and where
a number has limited scope (e.g. the 50-sample eval) the scope is stated.

> **Honesty note.** The full run is **done**: Mistral-7B was trained, evaluated,
> exported to GGUF, and deployed to a free CPU Space — see [Results](#results)
> and [Links](#links). The repo is *also* reproducible locally, end-to-end,
> via a **CPU smoke test on a tiny stand-in model** (`sshleifer/tiny-gpt2`,
> `pytest tests/test_smoke.py`) that proves the mechanics without a GPU. Two
> infrastructure facts from the real run are documented, not hidden: training
> ran on **SDPA, not Flash-Attention-2** (no prebuilt FA2 wheel for Colab's
> torch), and eval loads the two 7B models **sequentially in 4-bit** to fit a
> 15 GB T4.

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
- **Attention: SDPA on this run (FA2 attempted, no wheel).** The code prefers
  Flash-Attention-2 on Ampere, but Colab's current torch had **no prebuilt
  flash-attn wheel** and the source build stalled, so the real run used **SDPA**
  and recorded it honestly as `attn_implementation_active='sdpa'`
  (`SKIP_FLASH_ATTN=1` + `ALLOW_SDPA_FALLBACK=1`). The same runtime guard
  (`src/config.get_attn_implementation`) selects FA2 → `sdpa` → `eager` by
  capability — which is also what lets the CPU smoke test run without FA2.
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
- **Training: paid.** One paid A100 for **~1.8 h** (6,539 s; 1 epoch, 1,563
  steps) — the only paid step. The exact Colab compute-unit total wasn't
  recorded, so it is not stated here.
- **Serving: free.** HF Spaces **CPU** running the Q4_K_M GGUF (~1.3 tok/s).

Training is **not** free; only serving is.

## Repo layout
```
configs/        train.yaml (rank16/alpha32/NF4/bf16/FA2), ablations.yaml (24-cell grid)
src/config.py   model switch (MODEL_ID vs TINY_MODEL) + attn/quant/dtype GPU guards
src/data/       prepare.py — download→chat-template→dedupe→50k subsample→1k holdout→cache (idempotent)
src/train/      sft.py (QLoRA SFT, FA2, bf16, grad-ckpt, resume), ablations.py (24 short diagnostic runs)
src/eval/       evaluate.py — base vs fine-tuned on held-out set (4-bit sequential load; ADAPTER_DIR/--max-samples): BLEU (sacrebleu) + ROUGE-L → metrics.json
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
All numbers below are from the real end-to-end run. Nothing is invented; scope
is stated where it is limited.

### Training (Mistral-7B, single A100)
| metric | value |
|---|---|
| base model | `mistralai/Mistral-7B-Instruct-v0.3` |
| method | QLoRA, 4-bit NF4, double-quant, rank 16, alpha 32 |
| data | `gbharti/finance-alpaca`, 50,000 train instructions (1,000 held out) |
| precision | bf16 |
| peak VRAM | **10.2 GB** |
| runtime | **6,539 s (~1.8 h)** |
| steps / epochs | 1,563 steps, 1 epoch |
| final train loss | **1.82** |
| attn implementation (read back) | **sdpa** — no FA2 wheel for Colab's torch; `ALLOW_SDPA_FALLBACK=1`, recorded honestly |
| W&B run | `astral-river-1` |

### Evaluation — base vs fine-tuned
**Scope: a 50-sample subset of the 1,000-example held-out set** (4-bit,
sequential load of base then fine-tuned). This is *not* the full 1k.

| metric | base | fine-tuned | delta |
|---|---|---|---|
| BLEU (sacrebleu) | 5.35 | 8.13 | **+2.77** |
| ROUGE-L | 20.71 | 30.22 | **+9.50** |

Fine-tuned wins on both metrics. Source of truth: generated `metrics.json`.

### Export + serving
GGUF export: merge the adapter on a T4 **GPU** (a CPU merge OOMs), then
llama.cpp `convert_hf_to_gguf.py` → `llama-quantize` Q4_K_M.

| metric | value |
|---|---|
| f16 GGUF size | **13.5 GB** |
| Q4_K_M GGUF size | **4.07 GB** |
| compression (f16 → Q4_K_M) | **3.32×** |
| live CPU tok/s (Space, Q4_K_M) | **~1.3 tok/s** (measured live on free CPU) |

### Ablations
24 **short diagnostic** runs (rank {8,16,32} × LR {1e-4,2e-4} × scheduler
{cosine, linear, constant, constant_with_warmup}) — these are few-hundred-step
diagnostics to compare loss trajectories, **not** 24 full trainings. The best
config is selected programmatically (lowest eval_loss, tie-break smaller rank).
Comparison table: `ablation_results/comparison_table.md` (generated).

## Links
- **GitHub repo:** https://github.com/mohanemg07-web/Domain-specific-LLM-Fine-tuning
- **Adapter (QLoRA):** [`MohanGen/mistral7b-finance-qlora`](https://huggingface.co/MohanGen/mistral7b-finance-qlora)
- **GGUF (Q4_K_M):** [`MohanGen/mistral7b-finance-gguf`](https://huggingface.co/MohanGen/mistral7b-finance-gguf) — file `mistral7b-finance-q4_k_m.gguf`
- **Demo Space (free CPU):** [`MohanGen/mistral7b-finance-demo`](https://huggingface.co/spaces/MohanGen/mistral7b-finance-demo)
- **W&B run:** `astral-river-1`

## Status: proven end-to-end
**Real run (Mistral-7B) — done.** Training, evaluation, GGUF export, and a
free-CPU Space deploy all completed; the numbers are in [Results](#results) and
the artifacts in [Links](#links). Honest caveats from that run: **SDPA not FA2**
(no wheel for Colab's torch), eval on a **50-sample** subset of the 1k holdout,
and the GGUF merge done **on GPU** to avoid a CPU OOM.

**Reproducible locally via the CPU smoke test (`pytest tests/test_smoke.py`, 11/11 passing):**
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
- **flash-attn handled honestly**: the installer (`scripts/install_flash_attn.py`)
  picks a prebuilt wheel matching Colab's torch when one exists; for torch 2.11
  (no wheel, source build stalls) the notebook sets `SKIP_FLASH_ATTN=1` +
  `ALLOW_SDPA_FALLBACK=1` so the run proceeds on **SDPA**. Training reads back
  the **active** attn impl (`attn_implementation_active`) instead of assuming
  FA2 — which is why this run reports `sdpa`.
- **real finance-alpaca schema** (streaming dry run, ≤200 rows, no full
  download): confirmed columns `instruction` / `input` / `output` / `text`,
  empty-`input` rows handled, formatting non-empty, split overlap = 0.
- **GGUF f16 path proven on a tiny Llama-arch model** (SmolLM2-135M, mirrors
  Mistral): merge → `convert_hf_to_gguf.py` → real **f16 GGUF = 270,885,376
  bytes (258.34 MB)**. This is the *tiny stand-in*, not the 7B. Loadback
  (llama-cpp-python) and `Q4_K_M` quantize honest-skip locally (no compiler)
  and are exercised on Colab/Linux.

**Executed on the real A100 run (see RUN_ON_COLAB.md to reproduce):**
- trained + evaluated the real **Mistral-7B**; metrics in [Results](#results);
- **4-bit NF4 loading** ran for both training and eval (CUDA-only path);
- attention ran on **SDPA** (no FA2 wheel for Colab's torch — recorded as such,
  not assumed);
- the **full 50k** finance-alpaca pipeline (1k held out, 50 used for eval);
- built the real `Q4_K_M` GGUF (4.07 GB, 3.32× vs f16) and published it;
- deployed the free-CPU Space and recorded **live** ~1.3 tok/s.

Every large-model number in [Results](#results) is real and measured; the only
cost figure deliberately omitted is the exact Colab compute-unit total, which
wasn't recorded.
```bash
python -m src.config        # prints the runtime guard decisions on your box
```
