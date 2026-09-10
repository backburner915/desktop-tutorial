"""Build nominal/pregrasp frames from a candidate geometry audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r1_bimanual_dataset.core.contact_frame import write_contact_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("sugar_box", "cracker_box"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stem = "004_sugar_box_physics" if args.candidate == "sugar_box" else "003_cracker_box_physics"
    geometry = Path("reports") / f"candidate_{stem}.json"
    static = Path("reports") / f"candidate_{stem}_static_audit.json"
    output = args.output or Path("reports") / f"candidate_{stem}_contact_frames.json"
    report = write_contact_frame(geometry, static, output)
    print(json.dumps({"output": str(output), "validated": report["validated"], "station_distance_m": report["station_distance_m"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
