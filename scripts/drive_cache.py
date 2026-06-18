"""Durable data cache on Google Drive via a single tar.gz.

Why a tarball and not a raw Drive (FUSE) copy: the processed dataset is many
small Arrow shard files, and writing/reading thousands of small files over the
Drive FUSE mount is painfully slow and flaky. One compressed tarball is fast
and atomic-ish.

Subcommands (both idempotent, both safe when the source is missing):

  restore  --drive <tar.gz> --local <dir>
      If the Drive tarball exists AND the local dir has no cache yet, extract
      it into <dir>. No-op if the local cache already exists or the tarball is
      absent (cold start).

  snapshot --local <dir> --drive <tar.gz>
      Tar the local cache dir's contents to the Drive path. No-op if the local
      cache is missing. Writes to a .tmp then atomically renames, so a partial
      tarball never masquerades as a complete one.

"Cache present" is detected by the presence of <dir>/manifest.json (written by
src/data/prepare.py), so this never half-restores over a real cache.
"""

from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path


def _has_cache(local: Path) -> bool:
    return (local / "manifest.json").exists()


def restore(drive_tar: str, local_dir: str) -> int:
    drive = Path(drive_tar)
    local = Path(local_dir)
    if _has_cache(local):
        print(f"[drive_cache] local cache already present at {local} -- skip restore.")
        return 0
    if not drive.exists():
        print(f"[drive_cache] no Drive tarball at {drive} yet -- nothing to restore (cold start).")
        return 0
    local.mkdir(parents=True, exist_ok=True)
    print(f"[drive_cache] restoring {drive} -> {local}/ ...")
    with tarfile.open(drive, "r:gz") as tf:
        tf.extractall(local)
    print(f"[drive_cache] restored. manifest present: {_has_cache(local)}")
    return 0


def snapshot(local_dir: str, drive_tar: str) -> int:
    local = Path(local_dir)
    drive = Path(drive_tar)
    if not _has_cache(local):
        print(f"[drive_cache] no local cache at {local} (no manifest.json) -- nothing to snapshot.")
        return 0
    drive.parent.mkdir(parents=True, exist_ok=True)
    tmp = drive.with_name(drive.name + ".tmp")
    print(f"[drive_cache] snapshotting {local}/ -> {drive} ...")
    with tarfile.open(tmp, "w:gz") as tf:
        for item in sorted(local.iterdir()):
            tf.add(item, arcname=item.name)
    os.replace(tmp, drive)  # atomic-ish swap on the same filesystem
    print(f"[drive_cache] snapshot written ({drive.stat().st_size} bytes).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("restore")
    r.add_argument("--drive", required=True)
    r.add_argument("--local", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--local", required=True)
    s.add_argument("--drive", required=True)
    a = ap.parse_args()
    if a.cmd == "restore":
        return restore(a.drive, a.local)
    return snapshot(a.local, a.drive)


if __name__ == "__main__":
    raise SystemExit(main())
