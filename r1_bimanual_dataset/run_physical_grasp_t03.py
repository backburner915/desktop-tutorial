"""Run the verified physical R1 grasp runner against target object T03.

The original T01 runner remains unchanged in its default behavior.  This
entry point only supplies a different target Prim and scenario file; all
robot, fixed-base, IK, collision, and trajectory logic stays shared.
"""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
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
    if "--steady-place" not in args:
        args.append("--steady-place")
    original = sys.argv
    try:
        sys.argv = [str(Path(__file__))] + args
        return base_main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    raise SystemExit(main())
