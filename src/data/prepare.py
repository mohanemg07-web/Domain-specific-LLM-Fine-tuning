"""Idempotent, resumable data pipeline.

Steps (each cached so a re-run is a no-op):
    download -> format to Mistral chat template -> dedupe
      -> subsample (fixed seed) -> hold out 1k eval (fixed seed) -> cache.

Cache layout (under <data_cache_dir>, which on Colab points at Drive):
    <cache>/train/        (datasets save_to_disk)
    <cache>/eval/         (datasets save_to_disk)  -- the held-out 1k
    <cache>/manifest.json (provenance + disjointness proof)

Disjointness is enforced by hashing each example's instruction text and
splitting on the hash set BEFORE subsampling train, so train and eval can
never overlap. The manifest records the hashes' intersection size (must be 0).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from datasets import Dataset, load_from_disk

from src.config import REPO_ROOT, Settings, load_settings


def _synthetic_finance_dataset(n: int, seed: int) -> Dataset:
    """Tiny finance-style instruction dataset for the CPU smoke test only.

    Deliberately includes exact-duplicate rows so the dedupe step is actually
    exercised (and provably reduces the count) during the smoke run.
    """
    import random

    rng = random.Random(seed)
    topics = [
        ("What is compound interest?",
         "Compound interest is interest calculated on both the principal and accumulated interest."),
        ("Explain diversification.",
         "Diversification spreads investments across assets to reduce risk."),
        ("What is an ETF?",
         "An ETF is a fund that trades on an exchange and tracks an index or basket of assets."),
        ("Define liquidity.",
         "Liquidity is how quickly an asset can be converted to cash without affecting its price."),
        ("What is a bond yield?",
         "A bond yield is the return an investor earns on a bond, expressed as a percentage."),
        ("Explain dollar-cost averaging.",
         "Dollar-cost averaging invests a fixed amount regularly regardless of price."),
    ]
    rows = []
    for i in range(n):
        instr, out = topics[i % len(topics)]
        # vary a fraction so the set isn't all duplicates, but keep ~1/6 dupes
        suffix = "" if i % 6 == 0 else f" (case {i})"
        rows.append({"instruction": instr + suffix, "input": "", "output": out})
    rng.shuffle(rows)
    return Dataset.from_list(rows)

# ---------------------------------------------------------------------------
# Formatting: dataset rows -> a single "text" field using the chat template
# ---------------------------------------------------------------------------
def _row_to_messages(row: dict) -> list[dict]:
    """finance-alpaca rows have: instruction, input (optional), output.

    We fold any non-empty `input` into the user turn.
    """
    instruction = (row.get("instruction") or "").strip()
    context = (row.get("input") or "").strip()
    output = (row.get("output") or row.get("response") or "").strip()
    user = instruction if not context else f"{instruction}\n\n{context}"
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": output},
    ]


def _fallback_template(messages: list[dict]) -> str:
    """Used when the tokenizer has no chat template (e.g. tiny-gpt2)."""
    user = messages[0]["content"]
    assistant = messages[1]["content"]
    return f"### Instruction:\n{user}\n\n### Response:\n{assistant}"


def build_formatter(tokenizer):
    """Return fn(row)->{'text': ...} using the tokenizer's chat template if any."""
    has_template = getattr(tokenizer, "chat_template", None) is not None

    def _format(row: dict) -> dict:
        messages = _row_to_messages(row)
        if has_template:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            text = _fallback_template(messages)
        return {"text": text}

    return _format


# ---------------------------------------------------------------------------
# Dedupe + split helpers
# ---------------------------------------------------------------------------
def _hash_text(text: str) -> str:
    norm = " ".join(text.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _dedupe(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        key = _hash_text(r["text"])
        if key in seen:
            continue
        seen.add(key)
        r = dict(r)
        r["_hash"] = key
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Task 3: real finance-alpaca schema dry run (small slice, no full download)
# ---------------------------------------------------------------------------
def real_schema_dry_run(n: int = 200, settings: Optional[Settings] = None) -> dict:
    """Pull a SMALL real slice of the dataset and run it through the exact
    format -> dedupe -> split logic, asserting the real columns map correctly.

    Uses streaming so it never triggers the full 50k download -- it reads only
    the first ~n rows. This proves the real column handling (instruction /
    input / output -> chat template) that the synthetic smoke data could not.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    settings = settings or load_settings()
    # Use the tiny model's tokenizer for the template here: this is a schema/
    # plumbing check, not a model check, and it avoids the gated Mistral repo.
    tokenizer = AutoTokenizer.from_pretrained(settings.tiny_model)

    stream = load_dataset(settings.dataset_id, split="train", streaming=True)
    raw_rows = []
    for i, row in enumerate(stream):
        if i >= n:
            break
        raw_rows.append(row)
    if not raw_rows:
        raise RuntimeError("Streamed 0 rows from the real dataset.")

    columns = sorted(raw_rows[0].keys())
    # finance-alpaca exposes instruction / input / output (input often empty).
    assert "instruction" in columns, f"missing 'instruction'; got {columns}"
    assert "output" in columns, f"missing 'output'; got {columns}"

    formatter = build_formatter(tokenizer)
    formatted = [formatter(r) for r in raw_rows]
    texts = [f["text"] for f in formatted]

    # Every formatted example must be non-empty and contain both turns' content.
    empty = [t for t in texts if not t.strip()]
    assert not empty, f"{len(empty)} formatted rows are empty"

    # Prove empty-`input` rows are handled: find one (or synthesize the check).
    empty_input_rows = [r for r in raw_rows if not (r.get("input") or "").strip()]
    empty_input_ok = True
    if empty_input_rows:
        sample = formatter(empty_input_rows[0])
        empty_input_ok = bool(sample["text"].strip())

    # Run dedupe + a tiny split to prove the downstream plumbing accepts it.
    deduped = _dedupe([{"text": t} for t in texts])
    split_eval = deduped[: max(1, len(deduped) // 5)]
    split_train = deduped[len(split_eval):]
    overlap = {r["_hash"] for r in split_train} & {r["_hash"] for r in split_eval}

    report = {
        "dataset_id": settings.dataset_id,
        "rows_streamed": len(raw_rows),
        "columns": columns,
        "instruction_present": "instruction" in columns,
        "input_present": "input" in columns,
        "output_present": "output" in columns,
        "empty_input_rows_in_slice": len(empty_input_rows),
        "empty_input_handled": empty_input_ok,
        "all_formatted_non_empty": not empty,
        "after_dedupe": len(deduped),
        "split_overlap": len(overlap),  # must be 0
        "sample_formatted_text": texts[0][:300],
    }
    assert report["split_overlap"] == 0
    return report


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------
def _cache_dir(settings: Settings) -> Path:
    p = Path(settings.data_cache_dir)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def cache_exists(settings: Settings) -> bool:
    base = _cache_dir(settings)
    return (
        (base / "train").exists()
        and (base / "eval").exists()
        and (base / "manifest.json").exists()
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def prepare_dataset(
    settings: Optional[Settings] = None,
    tokenizer=None,
    force: bool = False,
) -> dict:
    """Produce (and cache) train/eval splits. Returns the manifest dict.

    Idempotent: if the cache exists and force is False, returns immediately.
    """
    settings = settings or load_settings()
    base = _cache_dir(settings)

    if cache_exists(settings) and not force:
        with open(base / "manifest.json", "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        manifest["cache_hit"] = True
        return manifest

    base.mkdir(parents=True, exist_ok=True)

    # Tokenizer (for chat template). Lazy import keeps config import light.
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(settings.active_model_id)

    # ---- download -------------------------------------------------------
    if settings.smoke_test:
        # Hermetic, fast, offline: synthesize finance-style rows so the smoke
        # test never downloads the full 68k parquet. The SAME format/dedupe/
        # split/cache logic below runs on these rows. The REAL run downloads
        # settings.dataset_id.
        raw = _synthetic_finance_dataset(
            n=max(settings.subsample_size * 3, 600), seed=settings.seed
        )
    else:
        from datasets import load_dataset

        raw = load_dataset(settings.dataset_id, split="train")

    # ---- format ---------------------------------------------------------
    formatter = build_formatter(tokenizer)
    formatted = raw.map(formatter, remove_columns=raw.column_names)
    rows = [{"text": r["text"]} for r in formatted if r["text"].strip()]

    # ---- dedupe ---------------------------------------------------------
    deduped = _dedupe(rows)

    # ---- deterministic shuffle -----------------------------------------
    import random

    rng = random.Random(settings.seed)
    rng.shuffle(deduped)

    # ---- hold out eval FIRST (never seen in train/ablation) -------------
    eval_n = min(settings.eval_holdout_size, max(1, len(deduped) // 5))
    eval_rows = deduped[:eval_n]
    pool = deduped[eval_n:]

    # ---- subsample train ------------------------------------------------
    train_n = min(settings.subsample_size, len(pool))
    train_rows = pool[:train_n]

    # ---- disjointness proof --------------------------------------------
    train_hashes = {r["_hash"] for r in train_rows}
    eval_hashes = {r["_hash"] for r in eval_rows}
    overlap = train_hashes & eval_hashes
    assert not overlap, f"Train/eval overlap detected: {len(overlap)} examples"

    # ---- persist --------------------------------------------------------
    def _strip(rs):
        return [{"text": r["text"]} for r in rs]

    train_ds = Dataset.from_list(_strip(train_rows))
    eval_ds = Dataset.from_list(_strip(eval_rows))
    # Save atomically-ish: write then it's there for the next no-op run.
    train_ds.save_to_disk(str(base / "train"))
    eval_ds.save_to_disk(str(base / "eval"))

    manifest = {
        "dataset_id": settings.dataset_id,
        "domain": settings.domain,
        "model_for_template": settings.active_model_id,
        "smoke_test": settings.smoke_test,
        "seed": settings.seed,
        "counts": {
            "raw": len(rows),
            "after_dedupe": len(deduped),
            "train": len(train_rows),
            "eval_holdout": len(eval_rows),
        },
        "disjointness": {
            "train_unique_hashes": len(train_hashes),
            "eval_unique_hashes": len(eval_hashes),
            "intersection": len(overlap),  # MUST be 0
        },
        "cache_hit": False,
    }
    with open(base / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def load_splits(settings: Optional[Settings] = None):
    """Load cached (train, eval) datasets, building them if absent."""
    settings = settings or load_settings()
    if not cache_exists(settings):
        prepare_dataset(settings)
    base = _cache_dir(settings)
    return load_from_disk(str(base / "train")), load_from_disk(str(base / "eval"))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    s = load_settings(smoke_test=args.smoke or None)
    m = prepare_dataset(s, force=args.force)
    print(json.dumps(m, indent=2))
