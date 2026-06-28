"""Phase 5 -- evaluation on the held-out 1k set.

Generates completions from BOTH the base model and the fine-tuned (adapter-
merged) model on the held-out set, then computes:
  * BLEU  via sacrebleu
  * ROUGE-L via evaluate/rouge_score

Reports absolute scores for base and fine-tuned plus the delta, and writes
metrics.json. NOTHING is hardcoded -- every number is computed from real
generations. On CPU/tiny model the numbers are real but tiny/meaningless;
that is the point of the smoke test (prove the harness, not the quality).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from src.config import (
    REPO_ROOT,
    Settings,
    cuda_available,
    load_settings,
)


def _split_prompt_reference(text: str) -> tuple[str, str]:
    """Recover (prompt, reference) from a formatted training example.

    Works for both the chat-template format and the fallback
    '### Instruction ... ### Response: ...' format.
    """
    markers = ["### Response:\n", "[/INST]", "<|assistant|>", "assistant\n"]
    for m in markers:
        if m in text:
            idx = text.index(m) + len(m)
            return text[:idx], text[idx:].strip()
    # Fallback: split halfway (keeps the harness running on odd formats).
    mid = len(text) // 2
    return text[:mid], text[mid:].strip()


def _generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def _load_model(settings: Settings, adapter_dir: Optional[str] = None):
    """Load the eval model with the SAME guards + quantization as training.

    Reuses src.train.sft.load_model_and_tokenizer so eval and training share one
    code path: 4-bit NF4 on CUDA (via config.get_quantization_config -> the 7B
    fits in ~10 GB on a 15 GB T4 instead of OOM-ing in full precision), the attn
    impl chosen by get_attn_implementation() (SDPA on a T4, never FA2; honors
    SKIP_FLASH_ATTN / ALLOW_SDPA_FALLBACK), and bf16 compute. On CPU the same
    guards load the tiny model in full precision (the smoke test).

    Pass adapter_dir to wrap the base with the trained LoRA adapter.
    """
    from src.train.sft import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(settings)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    # Training disables the KV cache for gradient checkpointing; eval only
    # generates, so re-enable it (faster, and there's no grad-checkpoint here).
    model.config.use_cache = True
    model.eval()
    return model, tokenizer


def _free_cuda() -> None:
    """Reclaim VRAM from a just-deleted model so only one 7B is ever resident."""
    import gc

    gc.collect()
    if cuda_available():
        import torch

        torch.cuda.empty_cache()


def _score(predictions: list[str], references: list[str]) -> dict:
    import sacrebleu
    from rouge_score import rouge_scorer

    bleu = sacrebleu.corpus_bleu(predictions, [references]).score

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    rouge_l = 100.0 * (sum(rl) / len(rl)) if rl else 0.0
    return {"bleu": round(bleu, 4), "rouge_l": round(rouge_l, 4)}


def evaluate(
    settings: Optional[Settings] = None,
    adapter_dir: Optional[str] = None,
    max_eval_samples: Optional[int] = None,
    max_new_tokens: int = 64,
    out_path: Optional[str] = None,
) -> dict:
    settings = settings or load_settings()

    from src.data.prepare import load_splits

    _, eval_ds = load_splits(settings)
    if max_eval_samples:
        eval_ds = eval_ds.select(range(min(len(eval_ds), max_eval_samples)))

    prompts, refs = [], []
    for ex in eval_ds:
        p, r = _split_prompt_reference(ex["text"])
        prompts.append(p)
        refs.append(r)

    base_id = settings.active_model_id
    if adapter_dir is None:
        # Precedence: explicit arg (above) > ADAPTER_DIR env > default path. The
        # env lets eval point at a Drive checkpoint OR a Hub repo (e.g.
        # 'MohanGen/mistral7b-finance-qlora', which PeftModel pulls directly) with
        # no symlink -- the default path doesn't exist on a fresh runtime.
        adapter_dir = os.environ.get("ADAPTER_DIR") or str(
            REPO_ROOT / settings.output_dir / "final_adapter"
        )

    # Run the two 7B models SEQUENTIALLY so only ONE is ever in VRAM. On a 15 GB
    # T4 holding both OOMs before generation even starts; 4-bit + this one-at-a-
    # time pattern keeps peak VRAM at a single ~10 GB model.

    # ---- base: load -> generate over the eval set -> free VRAM ---------
    base_model, base_tok = _load_model(settings)
    t0 = time.time()
    base_preds = [_generate(base_model, base_tok, p, max_new_tokens) for p in prompts]
    base_secs = round(time.time() - t0, 2)
    del base_model, base_tok
    _free_cuda()

    # ---- fine-tuned (base + adapter): load -> generate -> free ---------
    ft_model, ft_tok = _load_model(settings, adapter_dir=adapter_dir)
    t1 = time.time()
    ft_preds = [_generate(ft_model, ft_tok, p, max_new_tokens) for p in prompts]
    ft_secs = round(time.time() - t1, 2)
    del ft_model, ft_tok
    _free_cuda()

    base_scores = _score(base_preds, refs)
    ft_scores = _score(ft_preds, refs)

    metrics = {
        "model_id": base_id,
        "adapter_dir": adapter_dir,
        "num_eval_samples": len(prompts),
        "smoke_test": settings.smoke_test,
        "base": base_scores,
        "fine_tuned": ft_scores,
        "delta": {
            "bleu": round(ft_scores["bleu"] - base_scores["bleu"], 4),
            "rouge_l": round(ft_scores["rouge_l"] - base_scores["rouge_l"], 4),
        },
        "generation_seconds": {"base": base_secs, "fine_tuned": ft_secs},
        # REAL_METRIC: these are filled from the A100 run; do not invent.
        "note": "Numbers are computed from real generations. On CPU/tiny model they are not meaningful.",
    }

    out_path = out_path or str(REPO_ROOT / "metrics.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument(
        "--adapter-dir", default=None,
        help="LoRA adapter location (local dir or Hub repo id). "
             "Overrides the ADAPTER_DIR env var and the default path.",
    )
    args = ap.parse_args()
    s = load_settings(smoke_test=args.smoke or None)
    n = args.max_samples or (5 if args.smoke else None)
    print(json.dumps(
        evaluate(s, adapter_dir=args.adapter_dir, max_eval_samples=n,
                 max_new_tokens=16 if args.smoke else 128),
        indent=2,
    ))
