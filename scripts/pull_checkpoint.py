"""Sync a checkpoint/output dir FROM Google Drive (or any source) to local.

Run this at the START of a Colab session so training resumes from the latest
checkpoint instead of restarting. No-op if the source doesn't exist yet
(first run), so it's safe to always call.

Usage:
    python scripts/pull_checkpoint.py --src /content/drive/MyDrive/ckpts/run --dst outputs/run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def pull(src: str, dst: str) -> str | None:
    src_p, dst_p = Path(src), Path(dst)
    if not src_p.exists():
        print(f"No checkpoint at {src_p} yet -- starting fresh.")
        return None
    dst_p.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_p, dst_p, dirs_exist_ok=True)
    print(f"Pulled {src_p} -> {dst_p}")
    return str(dst_p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    pull(a.src, a.dst)
