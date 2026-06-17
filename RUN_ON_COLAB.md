# RUN_ON_COLAB.md — exact steps to get the REAL numbers on an A100

Claude Code built and CPU-smoke-tested this whole pipeline, but it has no GPU
and did **not** train or evaluate the real Mistral-7B. The numbers below come
from *your* A100 run. Do these in order.

## Prerequisites (once)
1. Accept the Mistral-7B-Instruct-v0.3 license on its HF model page.
2. Create an HF token with read access (and write, to push the adapter).
3. (Optional) Create a W&B account + API key for run logging.
4. Push this repo to GitHub (or upload to Drive) so the notebook can clone it.

## A. Train + ablations (paid A100, Colab)
1. Open `colab/train_notebook.ipynb` in Colab.
2. Runtime → Change runtime type → **A100 GPU**.
3. Add `HF_TOKEN` and `WANDB_API_KEY` in Colab **Secrets** (key icon).
4. **Runtime → Run all.** This will:
   - confirm the guards picked `flash_attention_2` + bf16 + 4-bit NF4,
   - prepare + cache data to Drive,
   - train QLoRA, checkpointing to Drive every `save_steps` (disconnect-safe),
   - push the final adapter to the Hub.
5. **Record the REAL metrics** the notebook prints: `peak_vram_gb`,
   `train_runtime_seconds`, `train_loss`, `train_steps`.
6. Ablations (optional, cheap): `python -m src.train.ablations`
   → copy `ablation_results/comparison_table.md` + `best_config.json`.
7. **Record approximate Colab compute-unit cost** for the run (Colab shows it).

## B. Evaluate on the held-out 1k (A100 or any GPU)
```bash
python -m src.eval.evaluate            # writes metrics.json (base vs fine-tuned)
```
Record `base`, `fine_tuned`, and `delta` for BLEU + ROUGE-L from `metrics.json`.

## C. Merge + quantize to GGUF
```bash
python -m src.export.merge             # prints REAL merged size
git clone https://github.com/ggerganov/llama.cpp
cmake -B build llama.cpp && cmake --build build --config Release
export LLAMA_CPP_DIR=$PWD/llama.cpp
python -m src.export.to_gguf           # prints REAL f16 + Q4_K_M sizes
```
Record `f16_size_bytes`, `quantized_size_bytes`, `compression_ratio` from
`gguf_out/gguf_info.json`. Upload the `.gguf` to an HF repo.

## D. Deploy the free CPU demo
1. Create an HF Space (Gradio, **CPU**).
2. Upload `spaces/app.py` + `spaces/requirements.txt`.
3. Set Space secrets `GGUF_REPO_ID` + `GGUF_FILENAME`.
4. Open the Space, send a prompt, and **record the live tok/s** it shows.

## E. Fill the README
Replace every `<!-- TODO: fill from real A100 run -->` marker in `README.md`
with the recorded numbers. Do not invent any.
```bash
python -m pytest tests/test_smoke.py   # must still pass after edits
```
