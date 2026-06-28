"""Phase 6b -- convert the merged model to GGUF and quantize to Q4_K_M.

Uses llama.cpp's convert_hf_to_gguf.py (HF -> GGUF f16) then llama-quantize
(f16 -> Q4_K_M). Prints REAL source/output file sizes.

This step needs a local llama.cpp checkout. We do NOT vendor it. Point at it
with LLAMA_CPP_DIR or pass --llama-cpp-dir. If it is missing, this function
returns {'skipped': True, ...} with an actionable message instead of inventing
sizes -- the smoke test treats a missing llama.cpp as a skip, not a failure.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from src.config import REPO_ROOT


def _find_convert_script(llama_cpp_dir: Path) -> Optional[Path]:
    for name in ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py"):
        p = llama_cpp_dir / name
        if p.exists():
            return p
    return None


def _find_quantize_bin(llama_cpp_dir: Path) -> Optional[str]:
    # llama.cpp build outputs vary by version/platform. An OUT-OF-SOURCE build run
    # from the repo root (`cmake -B build llama.cpp`) drops the binary under
    # <repo>/build/bin -- NOT under the llama.cpp checkout -- so search both. The
    # repo-root path is the one verified on Colab:
    #   /content/Domain-specific-LLM-Fine-tuning/build/bin/llama-quantize
    candidates = [
        llama_cpp_dir / "build" / "bin" / "llama-quantize",
        llama_cpp_dir / "build" / "bin" / "llama-quantize.exe",
        REPO_ROOT / "build" / "bin" / "llama-quantize",
        REPO_ROOT / "build" / "bin" / "llama-quantize.exe",
        llama_cpp_dir / "llama-quantize",
        llama_cpp_dir / "quantize",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    found = shutil.which("llama-quantize")
    return found


def to_gguf(
    merged_dir: Optional[str] = None,
    out_dir: Optional[str] = None,
    llama_cpp_dir: Optional[str] = None,
    quant_type: str = "Q4_K_M",
) -> dict:
    merged_dir = Path(merged_dir or (REPO_ROOT / "merged_model"))
    out_dir = Path(out_dir or (REPO_ROOT / "gguf_out"))
    out_dir.mkdir(parents=True, exist_ok=True)

    llama_cpp = Path(llama_cpp_dir or os.environ.get("LLAMA_CPP_DIR", str(REPO_ROOT / "llama.cpp")))
    convert = _find_convert_script(llama_cpp) if llama_cpp.exists() else None
    if convert is None:
        msg = (
            "llama.cpp not found. Set LLAMA_CPP_DIR (or --llama-cpp-dir) to a "
            "checkout with convert_hf_to_gguf.py. Install:\n"
            "  git clone https://github.com/ggerganov/llama.cpp\n"
            "  cmake -B build llama.cpp && cmake --build build --config Release"
        )
        print(msg)
        return {"skipped": True, "reason": "llama.cpp not found", "hint": msg}

    f16_path = out_dir / "model-f16.gguf"
    q_path = out_dir / f"model-{quant_type}.gguf"

    # 1) HF -> GGUF f16
    subprocess.run(
        [sys.executable, str(convert), str(merged_dir),
         "--outfile", str(f16_path), "--outtype", "f16"],
        check=True,
    )

    # 2) f16 -> Q4_K_M
    quant_bin = _find_quantize_bin(llama_cpp)
    if quant_bin is None:
        return {
            "skipped": True,
            "reason": "llama-quantize binary not built",
            "f16_gguf": str(f16_path),
            "f16_size_bytes": f16_path.stat().st_size,  # REAL
        }
    subprocess.run([quant_bin, str(f16_path), str(q_path), quant_type], check=True)

    info = {
        "skipped": False,
        "quant_type": quant_type,
        "f16_gguf": str(f16_path),
        "f16_size_bytes": f16_path.stat().st_size,          # REAL
        "quantized_gguf": str(q_path),
        "quantized_size_bytes": q_path.stat().st_size,      # REAL
        "compression_ratio": round(f16_path.stat().st_size / q_path.stat().st_size, 2),
    }
    with open(out_dir / "gguf_info.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    print(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--llama-cpp-dir", default=None)
    ap.add_argument("--quant-type", default="Q4_K_M")
    args = ap.parse_args()
    to_gguf(args.merged_dir, args.out_dir, args.llama_cpp_dir, args.quant_type)
