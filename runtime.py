from __future__ import annotations

import importlib.metadata
from typing import Any


class IsaacRuntime:
    """Own stage loading and simulation stepping for one Isaac Sim process."""

    def __init__(self, simulation_app: Any, scene_cfg: dict[str, Any]):
        self.app = simulation_app
        self.scene_cfg = scene_cfg
        self.stage = None
        self.timeline = None
        self.physics_dt = float(scene_cfg.get("physics_dt_fallback", 1.0 / 60.0))
        self.sim_time = 0.0

    def open(self) -> None:
        import omni.timeline
        import omni.usd

        # Do not let an authored timeline/physics scene run while the USD is
        # loading. The imported R1 starts above the support surface and must
        # be placed by the episode reset before gravity is allowed to advance.
        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.stop()
        scene_path = self.scene_cfg["stage_path"]
        print(f"R1 runner: loading stage {scene_path}", flush=True)
        context = omni.usd.get_context()
        if not context.open_stage(scene_path):
            raise RuntimeError(f"unable to open stage: {scene_path}")
        for _ in range(2400):
            self.app.update()
            self.stage = context.get_stage()
            if self.stage is not None:
                break
            if _ and _ % 120 == 0:
                print(f"R1 runner: waiting for stage ({_}/2400)", flush=True)
        if self.stage is None:
            raise RuntimeError("stage did not load")
        self.timeline.stop()
        # The USD context becoming non-null is earlier than PhysX tensor
        # readiness on this scene. Give Kit a short initialization window
        # before Articulation.initialize() asks for the physics view.
        for _ in range(120):
            self.app.update()
        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.stop()
        self.physics_dt = self._read_physics_dt()
        print(f"R1 runner: stage ready, physics_dt={self.physics_dt:.6f}s", flush=True)

    def _read_physics_dt(self) -> float:
        try:
            from pxr import UsdPhysics

            for prim in self.stage.Traverse():
                if prim.IsA(UsdPhysics.Scene):
                    value = prim.GetTimeStepsPerSecondAttr().Get()
                    if value and float(value) > 0:
                        return 1.0 / float(value)
        except Exception:
            pass
        return float(self.scene_cfg.get("physics_dt_fallback", 1.0 / 60.0))

    def play(self) -> None:
        if self.timeline is None:
            raise RuntimeError("runtime is not open")
        self.timeline.play()

    def prepare_physics(self, warmup_steps: int = 2) -> None:
        """Start PhysX and create the tensor simulation view before wrappers initialize."""

        if self.timeline is None:
            raise RuntimeError("runtime is not open")
        self.timeline.play()
        try:
            from isaacsim.core.simulation_manager import SimulationManager

            SimulationManager.initialize_physics()
        except Exception as exc:
            raise RuntimeError(f"unable to initialize Isaac Sim physics: {exc}") from exc
        for _ in range(max(1, int(warmup_steps))):
            self.app.update()
        # Physics tensor initialization is complete. Pause immediately so
        # RobotInterface can be initialized and reset without a falling-body
        # window between stage load and the scenario reset.
        self.timeline.stop()
        try:
            from isaacsim.core.simulation_manager import SimulationManager

            if SimulationManager.get_physics_sim_view() is None:
                raise RuntimeError("Isaac Sim physics tensor view is not available after warmup")
        except ImportError:
            # Older compatible runtimes do not expose SimulationManager; the
            # articulation wrapper will provide the final API error if needed.
            pass
        print("R1 runner: PhysX tensor view ready", flush=True)

    def step(self) -> None:
        self.app.update()
        self.sim_time += self.physics_dt

    def stop(self) -> None:
        if self.timeline is not None:
            self.timeline.stop()

    def metadata(self, robot: Any) -> dict[str, Any]:
        try:
            isaac_version = importlib.metadata.version("isaacsim")
        except importlib.metadata.PackageNotFoundError:
            isaac_version = "unknown"
        return {
            "isaac_sim_package": isaac_version,
            "physics_dt": self.physics_dt,
            "all_joint_names": robot.all_joint_names,
            "all_dof_count": robot.dof_count,
        }
