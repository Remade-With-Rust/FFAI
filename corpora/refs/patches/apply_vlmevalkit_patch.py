#!/usr/bin/env python3
"""Apply FFai's patches to the vendored VLMEvalKit checkout.

Two patches, both documented in sibling .patch files:

  1. vlmevalkit-windows-timeout.patch — upstream is UNIMPORTABLE on Windows
     (`timeout()` has no else branch, so `@timeout(...)` decorates with None).
  2. vlmevalkit-cpu-device.patch — the SmolVLM wrapper hardcodes CUDA, so
     Arm 1 cannot run on the CPU-first hardware Argus targets.

Both are recorded rather than merely applied, because a patched reference is
not the published reference and any ledger line produced through one has to be
readable as such later. `.tools-bench/` is gitignored, so a re-clone means a
re-apply — hence this script rather than a manual edit.

Idempotent, and it verifies each anchor matched exactly rather than assuming.
"""

import os
import sys

ROOT = os.path.join(".tools-bench", "VLMEvalKit", "vlmeval")
HIPHO = os.path.join(ROOT, "dataset", "utils", "hipho_verifier.py")
SMOLVLM = os.path.join(ROOT, "vlm", "smolvlm.py")

MARKER = "PATCHED FOR FFai"

# ---------------------------------------------------------------- patch 1 --
TIMEOUT_ANCHOR = "            return wrapper\n        return decorator\n"
TIMEOUT_FIXED = '''            return wrapper
        return decorator

    # PATCHED FOR FFai (corpora/refs/patches/vlmevalkit-windows-timeout.patch):
    # upstream has no else branch here, so on Windows this returns None and
    # `@timeout(...)` further down raises TypeError at IMPORT time, making the
    # whole package unimportable. A no-op decorator is what the posix path
    # already degrades to when called off the main thread.
    def _noop_decorator(func):
        return func

    return _noop_decorator
'''

# ---------------------------------------------------------------- patch 2 --
DEVICE_SHIM = '''

# PATCHED FOR FFai (corpora/refs/patches/vlmevalkit-cpu-device.patch):
# upstream hardcodes CUDA; FFai is CPU-first, so the hardcoded "cuda" values
# below are routed through these helpers. DEVICE SELECTION ONLY - weights,
# dtype, preprocessing, chat template and generation config are untouched, so
# the SCORE this wrapper produces is unchanged and only its SPEED differs.
def _ffai_device():
    import os as _os
    import torch as _torch
    d = _os.environ.get("FFAI_VLMEVAL_DEVICE")
    if d:
        return d
    return "cuda" if _torch.cuda.is_available() else "cpu"


def _ffai_empty_cache():
    import torch as _torch
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()

'''

CUDA_SUBS = (
    ('device_map="cuda"', "device_map=_ffai_device()"),
    ("device_map='cuda'", "device_map=_ffai_device()"),
    ('.to("cuda")', ".to(_ffai_device())"),
    (".to('cuda')", ".to(_ffai_device())"),
    ("torch.cuda.empty_cache()", "_ffai_empty_cache()"),
)


def patch_timeout() -> int:
    if not os.path.exists(HIPHO):
        print(f"not found: {HIPHO}\n"
              f"  git clone --depth 1 https://github.com/open-compass/VLMEvalKit "
              f".tools-bench/VLMEvalKit", file=sys.stderr)
        return 1
    src = open(HIPHO, encoding="utf-8").read()
    if MARKER in src:
        print("hipho_verifier.py: already patched")
        return 0
    if src.count(TIMEOUT_ANCHOR) != 1:
        print(f"hipho_verifier.py: anchor matched {src.count(TIMEOUT_ANCHOR)}x, expected 1 "
              f"— upstream moved; re-read before forcing", file=sys.stderr)
        return 1
    open(HIPHO, "w", encoding="utf-8").write(src.replace(TIMEOUT_ANCHOR, TIMEOUT_FIXED))
    print(f"hipho_verifier.py: patched (Windows import fix)")
    return 0


def patch_device() -> int:
    if not os.path.exists(SMOLVLM):
        print(f"not found: {SMOLVLM}", file=sys.stderr)
        return 1
    src = open(SMOLVLM, encoding="utf-8").read()
    if MARKER in src:
        print("smolvlm.py: already patched")
        return 0
    out, n = src, 0
    for old, new in CUDA_SUBS:
        n += out.count(old)
        out = out.replace(old, new)
    if n == 0:
        print("smolvlm.py: no cuda hardcodes found — upstream changed; re-read before forcing",
              file=sys.stderr)
        return 1
    # Insert the shim after the last top-level import, so the helpers exist
    # before any class body references them.
    lines = out.splitlines(keepends=True)
    last_import = 0
    for i, ln in enumerate(lines[:100]):
        if ln.startswith(("import ", "from ")):
            last_import = i
    lines.insert(last_import + 1, DEVICE_SHIM)
    open(SMOLVLM, "w", encoding="utf-8").write("".join(lines))
    print(f"smolvlm.py: patched ({n} cuda hardcodes routed through _ffai_device)")
    return 0


if __name__ == "__main__":
    raise SystemExit(patch_timeout() | patch_device())
