"""Pytest bootstrap.

Two jobs:
  1. Make `src` importable (repo root on sys.path).
  2. WINDOWS FIX: trl 1.5.x reads its bundled .jinja chat templates with the
     OS default codec, which crashes on Windows (cp1252) at import time.
     If we're on Windows and NOT already in UTF-8 mode, re-exec pytest with
     `-X utf8`. Linux/Colab default to UTF-8, so this is a no-op there.
"""

import os
import sys
from pathlib import Path

# 1) repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 2) Windows UTF-8 re-exec guard (run before trl is ever imported)
if (
    sys.platform == "win32"
    and not sys.flags.utf8_mode
    and os.environ.get("_UTF8_REEXEC") != "1"
):
    os.environ["_UTF8_REEXEC"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    # Re-run pytest in UTF-8 mode with the same arguments.
    args = [sys.executable, "-X", "utf8", "-m", "pytest", *sys.argv[1:]]
    os.execv(sys.executable, args)
