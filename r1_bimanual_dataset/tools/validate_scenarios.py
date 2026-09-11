from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r1_bimanual_dataset.config import ACTION_DIM, ScenarioConfig


def validate(directory: Path) -> dict:
    errors = []
    files = sorted(
        path for path in directory.glob("scenario_*.json") if path.stem.removeprefix("scenario_").isdigit()
    )
    intended = {"success": 0, "failure": 0}
    for path in files:
        try:
            scenario = ScenarioConfig.from_json(path)
            intended[scenario.intended_outcome] += 1
        except Exception as exc:
            errors.append({"file": path.name, "error": str(exc)})
    ratio_applicable = len(files) >= 10 and len(files) % 10 == 0
    ratio_ok = not ratio_applicable or intended["success"] * 2 == intended["failure"] * 3
    if ratio_applicable and not ratio_ok:
        errors.append({"error": f"intended ratio is {intended}, expected 6:4"})
    result = {
        "directory": str(directory.resolve()),
        "total": len(files),
        "intended_success": intended["success"],
        "intended_failure": intended["failure"],
        "action_dim": ACTION_DIM,
        "ratio_6_to_4": ratio_ok,
        "ratio_check_applicable": ratio_applicable,
        "valid": not errors,
        "errors": errors,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = validate(args.directory)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
