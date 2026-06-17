"""Install a flash-attn build that MATCHES the runtime torch -- no fixed pin.

Why no pin: a hardcoded flash-attn version (e.g. 2.7.4.post1) goes stale fast
and won't have a prebuilt wheel for whatever torch/CUDA Colab currently ships,
forcing a slow (often failing) source build. Instead we:

  1. Detect the runtime: torch major.minor, CUDA major, cpython tag, C++ ABI.
  2. Query flash-attn's GitHub releases and pick the newest prebuilt wheel whose
     name matches that runtime exactly.
  3. pip install that wheel URL.
  4. Fallback: `pip install -U flash-attn --no-build-isolation` (build/latest).

This is meant to run on the Colab A100 (Linux). It is a no-op-ish helper on a
non-CUDA box (it will just attempt the fallback, which the caller can ignore).
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request

RELEASES_API = "https://api.github.com/repos/Dao-AILab/flash-attention/releases"


def _runtime_tags() -> dict:
    import torch

    tv = torch.__version__.split("+")[0].split(".")
    torch_mm = f"{tv[0]}.{tv[1]}"                       # e.g. "2.5"
    cuda = torch.version.cuda                            # e.g. "12.4" or None
    cuda_major = f"cu{cuda.split('.')[0]}" if cuda else None
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    try:
        abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
    except Exception:
        abi = "FALSE"
    return {
        "torch": torch.__version__,
        "torch_mm": torch_mm,
        "cuda_major": cuda_major,
        "py_tag": py_tag,
        "cxx11abi": abi,
        "cuda_available": torch.cuda.is_available(),
    }


def _iter_release_assets(per_page: int = 10):
    """Yield asset (name, url) pairs newest-release-first."""
    req = urllib.request.Request(
        f"{RELEASES_API}?per_page={per_page}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "fa-installer"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        releases = json.load(resp)
    for rel in releases:
        for asset in rel.get("assets", []):
            yield asset.get("name", ""), asset.get("browser_download_url", "")


def _match_wheel(tags: dict) -> str | None:
    if not tags["cuda_major"]:
        return None  # no CUDA -> no prebuilt GPU wheel applies
    needles = [
        tags["cuda_major"],              # cu12
        f"torch{tags['torch_mm']}",      # torch2.5
        f"cxx11abi{tags['cxx11abi']}",   # cxx11abiFALSE / TRUE
        f"{tags['py_tag']}-{tags['py_tag']}",  # cp310-cp310
        "linux_x86_64",
    ]
    for name, url in _iter_release_assets():
        if name.endswith(".whl") and all(n in name for n in needles):
            return url
    return None


def _pip(*args: str) -> int:
    cmd = [sys.executable, "-m", "pip", "install", *args]
    print("+", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    tags = _runtime_tags()
    print("Runtime:", json.dumps(tags))
    if not tags["cuda_available"]:
        print("No CUDA runtime detected; flash-attn is GPU-only. Skipping "
              "(the attn guard will use sdpa/eager).")
        return 0

    wheel = None
    try:
        wheel = _match_wheel(tags)
    except Exception as e:
        print(f"Could not query flash-attn releases ({type(e).__name__}: {e}); "
              "will use the build fallback.")

    if wheel:
        print(f"Matched prebuilt wheel:\n  {wheel}")
        if _pip(wheel) == 0:
            _verify()
            return 0
        print("Prebuilt wheel install failed; falling back to source build.")
    else:
        print("No prebuilt wheel matched this torch/CUDA/cpython/abi.")

    # Fallback: latest flash-attn, built against the installed torch.
    rc = _pip("-U", "flash-attn", "--no-build-isolation")
    if rc == 0:
        _verify()
    else:
        print("flash-attn install FAILED. On an A100 the trainer will abort "
              "unless you set ALLOW_SDPA_FALLBACK=1 to proceed on SDPA.")
    return rc


def _verify() -> None:
    try:
        import flash_attn  # noqa: F401

        print("flash-attn installed:", flash_attn.__version__)
    except Exception as e:
        print(f"flash-attn import check failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
