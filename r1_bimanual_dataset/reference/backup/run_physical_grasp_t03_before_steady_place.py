"""Saved T03 entry point before the steady-start/place adjustment on 2026-09-07."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
T03_TARGET = "/World/TaskSetup/MovablePayloads/T03"
T03_SCENARIO = PACKAGE_ROOT / "reference" / "scenario_t03.json"


def main() -> int:
    from r1_bimanual_dataset.run_physical_grasp import main as base_main

    args = list(sys.argv[1:])
    if "--scenario" not in args:
        args.extend(["--scenario", str(T03_SCENARIO)])
    if "--target-object" not in args:
        args.extend(["--target-object", T03_TARGET])
    original = sys.argv
    try:
        sys.argv = [str(Path(__file__))] + args
        return base_main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    raise SystemExit(main())
