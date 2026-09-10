"""Open the corrected USD and keep the Isaac Sim GUI alive for inspection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r1_bimanual_dataset.core.launch import (
    available_system_memory_mb,
    prepare_isaac_environment,
    simulation_app_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Open an R1 USD scene for visual inspection")
    parser.add_argument("--scene", type=Path, default=Path("D:/Galaxea_Lab-galaxea-main/spacerobot.usd"))
    parser.add_argument("--play", action="store_true", help="start the timeline immediately")
    args = parser.parse_args()
    scene = args.scene.resolve()
    if not scene.is_file():
        raise FileNotFoundError(scene)

    prepare_isaac_environment()
    available_mb = available_system_memory_mb()
    if 0 <= available_mb < 1536:
        print(
            f"GUI scene viewer not started; only {available_mb} MB physical RAM is free. "
            "At least 1536 MB is required; "
            "Close other heavy applications and retry.",
            flush=True,
        )
        return 3
    from isaacsim import SimulationApp

    app = SimulationApp(simulation_app_config(headless=False))
    try:
        import omni.timeline
        import omni.usd

        context = omni.usd.get_context()
        if not context.open_stage(str(scene)):
            raise RuntimeError(f"unable to open stage: {scene}")
        for _ in range(2400):
            app.update()
            if context.get_stage() is not None:
                break
        if context.get_stage() is None:
            raise RuntimeError("stage did not load")
        print(f"OPENED_SCENE: {scene}", flush=True)
        print("Use Isaac Sim File/Stage and viewport tools to inspect the scene. Close the window to exit.", flush=True)
        if args.play:
            omni.timeline.get_timeline_interface().play()
        while app.is_running():
            app.update()
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
