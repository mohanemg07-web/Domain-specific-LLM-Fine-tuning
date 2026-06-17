"""End-to-end CPU smoke test on the TINY stand-in model.

Proves the FULL pipeline runs without a GPU and without FA2/bitsandbytes:
  config guards -> data (idempotent + disjoint) -> QLoRA SFT -> resume
  -> ablation grid -> eval (BLEU/ROUGE-L) -> merge -> GGUF (skip if no tools).

This does NOT train or evaluate the real 7B. Every metric produced here is
real but tiny/meaningless; the point is to prove the mechanics.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.config import (
    Settings,
    cuda_available,
    describe_environment,
    get_attn_implementation,
    get_quantization_config,
    load_settings,
)


@pytest.fixture(scope="session")
def smoke_settings(tmp_path_factory) -> Settings:
    base = tmp_path_factory.mktemp("smoke")
    s = load_settings(smoke_test=True)
    # isolate artifacts in the tmp dir so the test is hermetic
    return replace(
        s,
        output_dir=str(base / "out"),
        data_cache_dir=str(base / "cache"),
    )


# ---------------------------------------------------------------------------
# Phase 0 -- guards
# ---------------------------------------------------------------------------
def test_config_imports_and_guards_on_cpu():
    env = describe_environment()
    assert env["cuda_available"] is False, "this smoke test must run on CPU"
    # FA2 must be disabled on CPU; eager fallback expected.
    assert get_attn_implementation() == "eager"
    # bitsandbytes 4-bit must be disabled on CPU (no GPU import).
    assert get_quantization_config(load_settings(smoke_test=True)) is None
    assert env["training_bf16"] is False


def test_model_switch():
    s = load_settings(smoke_test=True)
    assert s.active_model_id == s.tiny_model
    assert s.active_target_modules == ["c_attn"]   # tiny GPT-2 attention proj
    assert s.effective_optim == "adamw_torch"      # paged optim only on CUDA


# ---------------------------------------------------------------------------
# Phase 1 -- data pipeline
# ---------------------------------------------------------------------------
def test_data_prepare_idempotent_and_disjoint(smoke_settings):
    from src.data.prepare import cache_exists, prepare_dataset

    m1 = prepare_dataset(smoke_settings, force=True)
    assert m1["cache_hit"] is False
    assert m1["disjointness"]["intersection"] == 0      # provably disjoint
    assert m1["counts"]["after_dedupe"] < m1["counts"]["raw"]  # dedupe did something

    assert cache_exists(smoke_settings)
    m2 = prepare_dataset(smoke_settings)                # second run = no-op
    assert m2["cache_hit"] is True


# ---------------------------------------------------------------------------
# Phase 2 -- QLoRA SFT + resume
# ---------------------------------------------------------------------------
def test_train_and_resume(smoke_settings):
    from src.train.sft import find_latest_checkpoint, train

    m = train(smoke_settings, resume=False)
    assert m["train_steps"] == smoke_settings.max_steps
    assert m["attn_implementation_active"] == "eager"   # read back from loaded model
    assert m["used_4bit_quant"] is False
    assert m["peak_vram_gb"] is None                    # CPU: no VRAM number invented
    assert Path(m["adapter_dir"]).exists()

    # a checkpoint should have been written (save_steps=5)
    ckpt = find_latest_checkpoint(smoke_settings.output_dir)
    assert ckpt is not None

    # resume: bump max_steps and confirm we continue from the checkpoint
    resumed = replace(smoke_settings, max_steps=smoke_settings.max_steps + 5)
    m2 = train(resumed, resume=True)
    assert m2["resumed_from"] is not None
    assert m2["train_steps"] == resumed.max_steps        # continued, not restarted


# ---------------------------------------------------------------------------
# Phase 3 -- ablations
# ---------------------------------------------------------------------------
def test_ablation_grid(smoke_settings):
    from src.config import load_ablation_config
    from src.train.ablations import run_ablations

    cfg = load_ablation_config()
    cfg.update(
        ranks=[8, 16],
        learning_rates=[2e-4],
        schedulers=["cosine", "constant"],   # 2x1x2 = 4 cells
        max_steps=3,
        subset_size=40,
        eval_subset_size=20,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=1,
        results_dir=str(Path(smoke_settings.output_dir).parent / "ablations"),
    )
    out = run_ablations(smoke_settings, cfg)
    assert len(out["results"]) == 4
    assert "eval_loss" in out["best"]
    assert Path(out["results_dir"], "comparison_table.md").exists()
    assert Path(out["results_dir"], "best_config.json").exists()


# ---------------------------------------------------------------------------
# Phase 5 -- evaluation
# ---------------------------------------------------------------------------
def test_evaluate_writes_real_metrics(smoke_settings, tmp_path):
    from src.eval.evaluate import evaluate

    out_path = tmp_path / "metrics.json"
    m = evaluate(
        smoke_settings,
        max_eval_samples=4,
        max_new_tokens=8,
        out_path=str(out_path),
    )
    assert out_path.exists()
    for key in ("bleu", "rouge_l"):
        assert key in m["base"] and key in m["fine_tuned"] and key in m["delta"]
        assert isinstance(m["base"][key], (int, float))
    assert m["num_eval_samples"] == 4


# ---------------------------------------------------------------------------
# Phase 6 -- merge + GGUF
# ---------------------------------------------------------------------------
def test_merge_adapter(smoke_settings, tmp_path):
    from src.export.merge import merge_adapter

    merged = tmp_path / "merged"
    info = merge_adapter(smoke_settings, merged_dir=str(merged))
    assert merged.exists()
    assert info["merged_size_bytes"] > 0                 # REAL on-disk size


def test_gguf_skips_cleanly_without_llama_cpp(smoke_settings, tmp_path, monkeypatch):
    from src.export import to_gguf as gguf_mod

    # Point at a non-existent llama.cpp so we exercise the honest-skip path.
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path / "no_llama_cpp"))
    info = gguf_mod.to_gguf(
        merged_dir=str(tmp_path / "merged"),
        out_dir=str(tmp_path / "gguf"),
        llama_cpp_dir=str(tmp_path / "no_llama_cpp"),
    )
    # Honest skip, not a crash and not an invented size.
    assert info["skipped"] is True


# ---------------------------------------------------------------------------
# FA2 Ampere enforcement + ALLOW_SDPA_FALLBACK (simulated Ampere on CPU)
# ---------------------------------------------------------------------------
def test_ampere_fa2_enforcement_and_sdpa_fallback(smoke_settings, monkeypatch):
    """On a simulated Ampere GPU where FA2 is unavailable:
      * default -> load aborts loudly,
      * ALLOW_SDPA_FALLBACK=1 -> proceeds on sdpa and records it.
    Proven on CPU by faking the capability + attn-guard return values.
    """
    import torch

    from src.train import sft

    # Pretend we're on an A100 but flash-attn is missing (guard -> sdpa).
    monkeypatch.setattr(sft, "gpu_compute_capability", lambda: 8.0)
    monkeypatch.setattr(sft, "get_attn_implementation", lambda: "sdpa")
    monkeypatch.setattr(sft, "get_compute_dtype", lambda: torch.float32)

    # Case 1: no escape hatch -> loud abort, no wasted session.
    monkeypatch.delenv("ALLOW_SDPA_FALLBACK", raising=False)
    with pytest.raises(RuntimeError, match="Flash Attention 2 is not available"):
        sft.load_model_and_tokenizer(smoke_settings)

    # Case 2: opt-in fallback -> proceeds on sdpa, records sdpa.
    monkeypatch.setenv("ALLOW_SDPA_FALLBACK", "1")
    model, _ = sft.load_model_and_tokenizer(smoke_settings)
    assert sft.active_attn_implementation(model) == "sdpa"


# ---------------------------------------------------------------------------
# Network helper for the real-data / real-model proofs
# ---------------------------------------------------------------------------
def _hf_reachable() -> bool:
    import urllib.request

    try:
        urllib.request.urlopen("https://huggingface.co", timeout=10)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Task 3 -- real finance-alpaca schema dry run (small slice, no full download)
# ---------------------------------------------------------------------------
def test_real_finance_alpaca_schema():
    if not _hf_reachable():
        pytest.skip("HF unreachable; real-data schema check needs network")
    from src.data.prepare import real_schema_dry_run

    report = real_schema_dry_run(n=60)
    # Real columns map correctly to the chat template.
    assert report["instruction_present"] and report["output_present"]
    assert report["all_formatted_non_empty"] is True
    assert report["empty_input_handled"] is True
    assert report["split_overlap"] == 0
    assert report["rows_streamed"] > 0


# ---------------------------------------------------------------------------
# Task 4 -- real f16 GGUF from a tiny Llama-arch (SmolLM2) merged model
# ---------------------------------------------------------------------------
def test_gguf_f16_proof(tmp_path):
    if not _hf_reachable():
        pytest.skip("HF unreachable; GGUF f16 proof needs to download SmolLM2")
    try:
        import gguf  # noqa: F401
    except Exception:
        pytest.skip("gguf package not installed; f16 conversion unavailable")

    from src.config import REPO_ROOT
    from src.export.gguf_proof import prove_gguf_f16

    llama_cpp_dir = REPO_ROOT / "llama.cpp"
    if not (llama_cpp_dir / "convert_hf_to_gguf.py").exists():
        pytest.skip("llama.cpp checkout missing; clone it to run the f16 proof")

    report = prove_gguf_f16(workdir=str(tmp_path / "gguf_proof"),
                            llama_cpp_dir=str(llama_cpp_dir))
    assert report.get("skipped") is not True
    assert report["f16_size_bytes"] > 0          # REAL on-disk f16 GGUF size
    assert Path(report["f16_gguf"]).exists()
    # Q4_K_M honest-skips locally without the compiled binary (no invented size).
    assert "quantize" in report
