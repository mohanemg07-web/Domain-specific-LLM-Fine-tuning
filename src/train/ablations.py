"""Ablation grid: 24 SHORT diagnostic runs (NOT 24 full trainings).

ranks {8,16,32} x learning_rates {1e-4,2e-4} x schedulers {cosine, linear,
constant, constant_with_warmup} = 24 cells. Each cell trains for a few
hundred steps on a small subset -- just enough to compare loss trajectories.
On the A100 the whole grid is cheap precisely because QLoRA keeps each run
fast and low-VRAM.

Outputs:
  <results_dir>/ablation_results.json   (every cell + its measured eval_loss)
  <results_dir>/comparison_table.md     (human-readable ranking)
  <results_dir>/best_config.json        (programmatically selected winner)

All numbers come from real runs. Nothing here is hardcoded.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

from src.config import (
    REPO_ROOT,
    Settings,
    cuda_available,
    load_ablation_config,
    load_settings,
    use_bf16_training,
)


def _grid(cfg: dict):
    ranks = cfg["ranks"]
    lrs = cfg["learning_rates"]
    scheds = cfg["schedulers"]
    for r, lr, sched in itertools.product(ranks, lrs, scheds):
        yield {"rank": r, "learning_rate": lr, "scheduler": sched}


def _run_cell(base_settings: Settings, cfg: dict, cell: dict, train_ds, eval_ds) -> dict:
    """Train one short cell, return its measured eval_loss."""
    import torch
    from trl import SFTConfig, SFTTrainer

    from src.train.sft import build_lora_config, load_model_and_tokenizer

    alpha = cell["rank"] * int(cfg.get("lora_alpha_multiplier", 2))
    s = replace(
        base_settings,
        lora_rank=cell["rank"],
        lora_alpha=alpha,
        learning_rate=cell["learning_rate"],
        lr_scheduler_type=cell["scheduler"],
        max_steps=int(cfg["max_steps"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
    )

    model, tokenizer = load_model_and_tokenizer(s)
    peft_config = build_lora_config(s)

    tag = f"r{cell['rank']}_lr{cell['learning_rate']}_{cell['scheduler']}"
    out_dir = str(Path(cfg["results_dir"]).resolve() / "runs" / tag)

    args = SFTConfig(
        output_dir=out_dir,
        dataset_text_field="text",
        max_length=s.max_seq_length,
        packing=False,
        max_steps=s.max_steps,
        per_device_train_batch_size=s.per_device_train_batch_size,
        gradient_accumulation_steps=s.gradient_accumulation_steps,
        learning_rate=s.learning_rate,
        lr_scheduler_type=s.lr_scheduler_type,
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        bf16=use_bf16_training(),
        fp16=False,
        gradient_checkpointing=s.gradient_checkpointing,
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=max(1, s.max_steps // 2),
        report_to=([s.report_to] if s.report_to != "none" else []),
        run_name=tag,
        seed=s.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    t0 = time.time()
    trainer.train()
    eval_metrics = trainer.evaluate()
    runtime = round(time.time() - t0, 2)

    return {
        **cell,
        "lora_alpha": alpha,
        "eval_loss": float(eval_metrics.get("eval_loss")),
        "runtime_seconds": runtime,
    }


def run_ablations(
    settings: Optional[Settings] = None,
    ablation_cfg: Optional[dict] = None,
) -> dict:
    settings = settings or load_settings()
    cfg = ablation_cfg or load_ablation_config()

    results_dir = Path(cfg["results_dir"])
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    cfg["results_dir"] = str(results_dir)

    # Data: small subsets for fast signal. Reuse the cached splits.
    from src.data.prepare import load_splits

    train_ds, eval_ds = load_splits(settings)
    train_ds = train_ds.select(range(min(len(train_ds), int(cfg["subset_size"]))))
    eval_ds = eval_ds.select(range(min(len(eval_ds), int(cfg["eval_subset_size"]))))

    cells = list(_grid(cfg))
    results = []
    for i, cell in enumerate(cells, 1):
        print(f"[ablation {i}/{len(cells)}] {cell}")
        results.append(_run_cell(settings, cfg, cell, train_ds, eval_ds))

    # Select winner programmatically.
    mode = cfg.get("selection_mode", "min")
    metric = cfg.get("selection_metric", "eval_loss")
    best = sorted(
        results,
        key=lambda r: (r[metric], r["rank"]),  # tie-break: smaller rank cheaper to serve
        reverse=(mode == "max"),
    )[0]

    # Write artifacts.
    with open(results_dir / "ablation_results.json", "w", encoding="utf-8") as fh:
        json.dump({"results": results, "best": best, "grid_size": len(cells)}, fh, indent=2)
    with open(results_dir / "best_config.json", "w", encoding="utf-8") as fh:
        json.dump(best, fh, indent=2)
    _write_table(results, best, results_dir / "comparison_table.md")

    print(f"Selected best config: {best}")
    return {"results": results, "best": best, "results_dir": str(results_dir)}


def _write_table(results, best, path: Path):
    ranked = sorted(results, key=lambda r: r["eval_loss"])
    lines = [
        "# Ablation comparison (short diagnostic runs)",
        "",
        "These are fast, few-hundred-step diagnostic runs on a small subset,",
        "NOT 24 full trainings. Ranking is by measured eval_loss.",
        "",
        "| rank | lr | scheduler | eval_loss | runtime_s |",
        "|------|------|-----------|-----------|-----------|",
    ]
    for r in ranked:
        star = "  **<-- selected**" if r is best else ""
        lines.append(
            f"| {r['rank']} | {r['learning_rate']} | {r['scheduler']} "
            f"| {r['eval_loss']:.4f} | {r['runtime_seconds']}{star} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    s = load_settings(smoke_test=args.smoke or None)
    cfg = load_ablation_config()
    if args.smoke:
        # Tiny grid + tiny steps so the smoke test stays fast: 2x1x2 = 4 cells.
        cfg.update(
            ranks=[8, 16],
            learning_rates=[2e-4],
            schedulers=["cosine", "constant"],
            max_steps=3,
            subset_size=40,
            eval_subset_size=20,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            logging_steps=1,
        )
    print(json.dumps(run_ablations(s, cfg)["best"], indent=2))
