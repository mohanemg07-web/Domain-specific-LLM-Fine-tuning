"""Sync a local checkpoint/output dir TO Google Drive (or any target dir).

On Colab, Drive is mounted at /content/drive/MyDrive. Copying checkpoints
there means a runtime disconnect costs at most `save_steps` steps, never the
whole run. Idempotent: re-copying an unchanged dir is effectively a no-op
(same files overwritten).

Usage:
    python scripts/push_checkpoint.py --src outputs/run --dst /content/drive/MyDrive/ckpts/run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def push(src: str, dst: str) -> str:
    src_p, dst_p = Path(src), Path(dst)
    if not src_p.exists():
        raise FileNotFoundError(src_p)
    dst_p.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_p, dst_p, dirs_exist_ok=True)
    print(f"Pushed {src_p} -> {dst_p}")
    return str(dst_p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    push(a.src, a.dst)
