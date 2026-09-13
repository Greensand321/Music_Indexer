"""Record / verify hashes of files that parallel page agents must NOT modify.

Parallel redesign agents each own exactly one gui/workspaces/<page>.py. The theme
engine, the workspace base class and the main window are shared: if two agents edit
them concurrently the result is silent corruption. This records their hashes before
a fan-out and verifies them after.

  python3 scripts/guard_shared.py record   > .guard.json
  python3 scripts/guard_shared.py verify   < .guard.json     # exit 1 on drift
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROTECTED = sorted(
    set(
        glob.glob(os.path.join(REPO, "gui/themes/*.py"))
        + glob.glob(os.path.join(REPO, "gui/fonts/*.py"))
        + [
            os.path.join(REPO, "gui/workspaces/base.py"),
            os.path.join(REPO, "gui/main_window.py"),
            os.path.join(REPO, "gui/compat.py"),
            os.path.join(REPO, "gui/widgets/sidebar.py"),
            os.path.join(REPO, "gui/widgets/top_bar.py"),
            os.path.join(REPO, "gui/widgets/log_drawer.py"),
            os.path.join(REPO, "scripts/page_smoke.py"),
            os.path.join(REPO, "scripts/ui_smoke.py"),
        ]
    )
)


def digest(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "record"
    current = {os.path.relpath(p, REPO): digest(p) for p in PROTECTED if os.path.exists(p)}

    if mode == "record":
        json.dump(current, sys.stdout, indent=1, sort_keys=True)
        print()
        return 0

    before = json.load(sys.stdin)
    drift = [k for k in sorted(set(before) | set(current)) if before.get(k) != current.get(k)]
    if drift:
        print(f"!! {len(drift)} PROTECTED FILE(S) MODIFIED — a page agent edited shared code:")
        for k in drift:
            print(f"   {k}")
        print("\nReview with: git diff -- " + " ".join(drift))
        return 1
    print(f"OK — all {len(current)} protected files unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
