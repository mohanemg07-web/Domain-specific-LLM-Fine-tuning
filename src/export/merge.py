"""Phase 6a -- merge the LoRA adapter into the base model.

Loads the base model in bf16 (fp32 on CPU), attaches the trained adapter,
merges the weights, and saves a standalone merged model. Prints REAL on-disk
sizes (no invented numbers).

Note: merging is done in bf16/fp32, NOT 4-bit. You cannot merge LoRA deltas
into 4-bit packed weights; the standard recipe is to reload the base in
half/full precision, merge, then quantize to GGUF Q4_K_M afterwards
(export/to_gguf.py). This keeps the merge lossless before the deliberate
4-bit deployment step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.config import REPO_ROOT, Settings, get_compute_dtype, load_settings


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def merge_adapter(
    settings: Optional[Settings] = None,
    adapter_dir: Optional[str] = None,
    merged_dir: Optional[str] = None,
) -> dict:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = settings or load_settings()
    base_id = settings.active_model_id
    adapter_dir = adapter_dir or str(REPO_ROOT / settings.output_dir / "final_adapter")
    merged_dir = merged_dir or str(REPO_ROOT / "merged_model")
    Path(merged_dir).mkdir(parents=True, exist_ok=True)

    try:
        base = AutoModelForCausalLM.from_pretrained(base_id, dtype=get_compute_dtype())
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=get_compute_dtype())

    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()  # fold LoRA deltas into base weights
    model.save_pretrained(merged_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    tokenizer.save_pretrained(merged_dir)

    size_bytes = _dir_size_bytes(Path(merged_dir))
    info = {
        "base_model": base_id,
        "adapter_dir": adapter_dir,
        "merged_dir": merged_dir,
        "merged_size_bytes": size_bytes,                       # REAL
        "merged_size_mb": round(size_bytes / (1024 ** 2), 2),  # REAL
    }
    with open(Path(merged_dir) / "merge_info.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    print(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    merge_adapter(load_settings(smoke_test=args.smoke or None))
