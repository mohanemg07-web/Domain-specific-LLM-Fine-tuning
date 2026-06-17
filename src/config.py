"""Central configuration + hardware guards.

This module is the single source of truth for:
  * which model to load (real Mistral-7B vs the tiny CPU smoke-test model),
  * the attention implementation (FA2 on Ampere+, sdpa on older CUDA, eager on CPU),
  * whether to use 4-bit NF4 quantization (CUDA only -- bitsandbytes needs a GPU),
  * the compute dtype (bf16 on Ampere, fp32 on CPU).

The guards exist so the SAME code path runs both the real A100 job and the
CPU smoke test. On CPU we never import bitsandbytes or flash_attn (both fail
without CUDA), so those imports are intentionally lazy/local, not top-level.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs" / "train.yaml"
DEFAULT_ABLATIONS_CONFIG = REPO_ROOT / "configs" / "ablations.yaml"


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------
def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def gpu_compute_capability() -> Optional[float]:
    """Return CUDA compute capability as a float (e.g. 8.0 for A100), or None."""
    if not cuda_available():
        return None
    try:
        import torch

        major, minor = torch.cuda.get_device_capability()
        return float(f"{major}.{minor}")
    except Exception:
        return None


def _flash_attn_importable() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


def get_attn_implementation() -> str:
    """Pick the best available attention implementation.

    * Ampere+ (compute capability >= 8.0) AND flash-attn installed -> flash_attention_2
    * any other CUDA GPU -> sdpa
    * CPU (no CUDA) -> eager   (lets the smoke test run without FA2)
    """
    cc = gpu_compute_capability()
    if cc is None:
        return "eager"  # CPU smoke test
    if cc >= 8.0 and _flash_attn_importable():
        return "flash_attention_2"
    return "sdpa"


def get_compute_dtype():
    """bf16 on Ampere (native), fp32 on CPU. We never use fp16 here."""
    import torch

    cc = gpu_compute_capability()
    if cc is not None and cc >= 8.0:
        return torch.bfloat16
    return torch.float32


def use_bf16_training() -> bool:
    """Whether to pass bf16=True to TrainingArguments (Ampere only)."""
    cc = gpu_compute_capability()
    return cc is not None and cc >= 8.0


def get_quantization_config(settings: "Settings"):
    """Return a BitsAndBytesConfig for 4-bit NF4, or None on CPU.

    bitsandbytes requires CUDA, so on CPU we return None and the model loads
    in full precision. This is the quantization analogue of the FA2 guard:
    the smoke test exercises the same code without a GPU.
    """
    if not cuda_available():
        return None  # CPU: load fp32, no bitsandbytes import at all
    import torch
    from transformers import BitsAndBytesConfig

    compute_dtype = (
        torch.bfloat16
        if settings.bnb_4bit_compute_dtype == "bfloat16"
        else torch.float16
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=settings.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=settings.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    # models
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    tiny_model: str = "sshleifer/tiny-gpt2"
    smoke_test: bool = False

    # data
    dataset_id: str = "gbharti/finance-alpaca"
    domain: str = "finance"
    subsample_size: int = 50000
    eval_holdout_size: int = 1000
    seed: int = 42

    # lora
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # quant
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    # training
    max_seq_length: int = 1024
    num_train_epochs: int = 1
    max_steps: int = -1  # -1 => use epochs
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    optim: str = "paged_adamw_8bit"
    bf16: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    eval_steps: int = 200

    # output
    output_dir: str = "outputs/mistral7b-finance-qlora"
    resume_from_checkpoint: bool = True

    # hub
    push_to_hub: bool = False
    hub_model_id: str = ""

    # tracking
    report_to: str = "wandb"
    wandb_project: str = "mistral7b-finance-qlora"

    # cache
    data_cache_dir: str = "data_cache"

    # ---- derived helpers -------------------------------------------------
    @property
    def active_model_id(self) -> str:
        """The model actually loaded: tiny stand-in under smoke test."""
        return self.tiny_model if self.smoke_test else self.model_id

    @property
    def active_target_modules(self):
        """GPT-2 (tiny) exposes Conv1D 'c_attn', not q_proj/k_proj/...

        Under the smoke test we target the tiny model's attention projection
        so PEFT actually finds something to adapt. This is honest: the smoke
        test proves the *mechanics*, not the exact module set of the 7B.
        """
        if self.smoke_test:
            return ["c_attn"]
        return self.lora_target_modules

    @property
    def effective_optim(self) -> str:
        """paged_adamw_8bit needs bitsandbytes/CUDA; fall back on CPU."""
        if not cuda_available():
            return "adamw_torch"
        return self.optim


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _coerce_compute_dtype(raw: Any) -> str:
    return str(raw)


def load_settings(
    config_path: Optional[str | Path] = None,
    smoke_test: Optional[bool] = None,
    overrides: Optional[dict] = None,
) -> Settings:
    """Build Settings from YAML + environment + explicit overrides.

    Precedence (low -> high): dataclass defaults < YAML < env < overrides.

    Environment switches:
      SMOKE_TEST=1            -> use tiny model + tiny sizes
      MODEL_ID / TINY_MODEL   -> override either model id
      HF_TOKEN / WANDB_API_KEY are read by the trainer, not stored here.
    """
    data: dict = {}
    path = Path(config_path) if config_path else DEFAULT_TRAIN_CONFIG
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

    # YAML uses learning_rate etc. directly; map dataset/model names through.
    settings = Settings()
    valid_fields = set(settings.__dataclass_fields__.keys())
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    settings = replace(settings, **filtered)

    # Environment overrides
    env_smoke = os.environ.get("SMOKE_TEST")
    if env_smoke is not None and smoke_test is None:
        smoke_test = env_smoke not in ("", "0", "false", "False")
    if os.environ.get("MODEL_ID"):
        settings = replace(settings, model_id=os.environ["MODEL_ID"])
    if os.environ.get("TINY_MODEL"):
        settings = replace(settings, tiny_model=os.environ["TINY_MODEL"])

    if smoke_test is not None:
        settings = replace(settings, smoke_test=smoke_test)

    if overrides:
        valid = {k: v for k, v in overrides.items() if k in valid_fields}
        settings = replace(settings, **valid)

    # Smoke-test shrink: tiny model, tiny data, tiny steps, CPU-safe knobs.
    if settings.smoke_test:
        settings = replace(
            settings,
            subsample_size=200,
            eval_holdout_size=40,
            max_seq_length=64,
            num_train_epochs=1,
            max_steps=10,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            save_steps=5,
            save_total_limit=2,
            eval_steps=5,
            logging_steps=1,
            gradient_checkpointing=False,  # avoids PEFT input-grad pitfall on tiny CPU model
            report_to="none",
            push_to_hub=False,
            output_dir="outputs/smoke",
            data_cache_dir="data_cache/smoke",
        )

    return settings


def load_ablation_config(config_path: Optional[str | Path] = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_ABLATIONS_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def describe_environment() -> dict:
    """Snapshot of the runtime guard decisions -- handy for logs/README."""
    return {
        "cuda_available": cuda_available(),
        "compute_capability": gpu_compute_capability(),
        "attn_implementation": get_attn_implementation(),
        "uses_4bit_quant": cuda_available(),
        "training_bf16": use_bf16_training(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(describe_environment(), indent=2))
