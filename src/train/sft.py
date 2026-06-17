"""QLoRA supervised fine-tuning (SFT).

Real run (A100): 4-bit NF4 + double quant, bf16 compute, FA2 attention,
gradient checkpointing, paged 8-bit optimizer, checkpoint-to-Drive every N
steps, auto-resume.

Smoke run (CPU): the SAME code, but config.py guards swap in the tiny model,
disable the bitsandbytes quant config, fall back to eager attention, and use
fp32 + adamw_torch. Nothing about the call sites changes.

All metrics (peak VRAM, train runtime, steps/sec) are MEASURED here and
returned/saved. Nothing is hardcoded.
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
    get_attn_implementation,
    get_compute_dtype,
    get_quantization_config,
    gpu_compute_capability,
    load_settings,
    use_bf16_training,
)


def _is_ampere_or_newer() -> bool:
    cc = gpu_compute_capability()
    return cc is not None and cc >= 8.0


# ---------------------------------------------------------------------------
# Model + tokenizer loading (with the runtime guards)
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(settings: Settings):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = settings.active_model_id
    attn_impl = get_attn_implementation()
    quant_config = get_quantization_config(settings)  # None on CPU
    compute_dtype = get_compute_dtype()

    # Task 2: on an Ampere+ GPU we REQUIRE Flash Attention 2. If the guard fell
    # back to sdpa it means flash-attn isn't importable -- fail loudly instead
    # of silently training with the wrong (slower, non-advertised) kernel.
    if _is_ampere_or_newer() and attn_impl != "flash_attention_2":
        raise RuntimeError(
            "Ampere+ GPU detected but Flash Attention 2 is not available "
            f"(attn guard selected '{attn_impl}'). Install it with:\n"
            "    pip install flash-attn==2.7.4.post1 --no-build-isolation\n"
            "Refusing to silently fall back to sdpa on an A100."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # transformers 5.x renamed torch_dtype -> dtype; support both.
    load_kwargs = dict(attn_implementation=attn_impl)
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=compute_dtype, **load_kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=compute_dtype, **load_kwargs
        )

    # Read back what the LOADED model is ACTUALLY using (not what we requested).
    active_attn = getattr(model.config, "_attn_implementation", attn_impl)
    if _is_ampere_or_newer() and active_attn != "flash_attention_2":
        raise RuntimeError(
            f"Model loaded with attn_implementation='{active_attn}' on an "
            "Ampere+ GPU; expected 'flash_attention_2'. Aborting rather than "
            "recording a metric that doesn't match the advertised setup."
        )

    model.config.use_cache = False  # required with gradient checkpointing
    return model, tokenizer


def active_attn_implementation(model) -> str:
    """The attention impl the loaded model is actually using (read back)."""
    return getattr(model.config, "_attn_implementation", "unknown")


def build_lora_config(settings: Settings):
    from peft import LoraConfig

    return LoraConfig(
        r=settings.lora_rank,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=settings.active_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


# ---------------------------------------------------------------------------
# Checkpoint discovery (for resume)
# ---------------------------------------------------------------------------
def find_latest_checkpoint(output_dir: str | Path) -> Optional[str]:
    p = Path(output_dir)
    if not p.exists():
        return None
    cks = [d for d in p.glob("checkpoint-*") if d.is_dir()]
    if not cks:
        return None
    cks.sort(key=lambda d: int(d.name.split("-")[-1]))
    return str(cks[-1])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    settings: Optional[Settings] = None,
    train_dataset=None,
    eval_dataset=None,
    resume: Optional[bool] = None,
) -> dict:
    """Run QLoRA SFT. Returns a dict of MEASURED metrics."""
    import torch
    from trl import SFTConfig, SFTTrainer

    settings = settings or load_settings()
    resume = settings.resume_from_checkpoint if resume is None else resume

    # Data (build/cache if not supplied).
    if train_dataset is None:
        from src.data.prepare import load_splits

        train_dataset, eval_holdout = load_splits(settings)
        # NOTE: we deliberately do NOT use the held-out eval set during
        # training. It is reserved for Phase 5. eval_dataset stays None.

    output_dir = settings.output_dir
    if not Path(output_dir).is_absolute():
        output_dir = str(REPO_ROOT / output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(settings)
    # Task 2: record the attn impl the model ACTUALLY loaded with (read back
    # from model.config), not the value we requested. This is what the README
    # should report.
    active_attn = active_attn_implementation(model)
    peft_config = build_lora_config(settings)

    # Reset peak-memory counter so the reported VRAM is for THIS run only.
    if cuda_available():
        torch.cuda.reset_peak_memory_stats()

    sft_config = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_length=settings.max_seq_length,
        packing=False,
        num_train_epochs=settings.num_train_epochs,
        max_steps=settings.max_steps,
        per_device_train_batch_size=settings.per_device_train_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        learning_rate=settings.learning_rate,
        lr_scheduler_type=settings.lr_scheduler_type,
        warmup_ratio=settings.warmup_ratio,
        weight_decay=settings.weight_decay,
        optim=settings.effective_optim,           # paged_adamw_8bit on CUDA, adamw_torch on CPU
        bf16=use_bf16_training(),                  # True only on Ampere
        fp16=False,                                # never fp16 (see README)
        gradient_checkpointing=settings.gradient_checkpointing,
        logging_steps=settings.logging_steps,
        save_steps=settings.save_steps,
        save_total_limit=settings.save_total_limit,
        save_strategy="steps",
        eval_strategy="no",                        # eval is Phase 5, on held-out set
        report_to=([settings.report_to] if settings.report_to != "none" else []),
        seed=settings.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Resume from the latest Drive checkpoint if present (disconnect-safe).
    resume_ckpt = find_latest_checkpoint(output_dir) if resume else None

    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    wall_seconds = time.time() - t0

    # Save final adapter (PEFT adapter only, not the full base model).
    adapter_dir = str(Path(output_dir) / "final_adapter")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # ---- MEASURED metrics (never hardcoded) ----------------------------
    peak_vram_gb = None
    if cuda_available():
        peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)

    metrics = {
        "attn_implementation_requested": get_attn_implementation(),
        "attn_implementation_active": active_attn,  # read back from loaded model
        "used_4bit_quant": cuda_available(),
        "bf16_training": use_bf16_training(),
        "resumed_from": resume_ckpt,
        "train_runtime_seconds": round(wall_seconds, 2),
        "train_steps": int(train_result.global_step),
        "train_loss": float(train_result.training_loss),
        "peak_vram_gb": peak_vram_gb,  # None on CPU; REAL number on A100
        "adapter_dir": adapter_dir,
        "model_id": settings.active_model_id,
        "smoke_test": settings.smoke_test,
    }
    with open(Path(output_dir) / "train_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    s = load_settings(smoke_test=args.smoke or None)
    print(json.dumps(train(s), indent=2))
