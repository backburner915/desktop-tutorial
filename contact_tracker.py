"""Real PhysX contact evidence for the R1 two-finger grippers.

This module only applies ``PhysxContactReportAPI`` in the current Session
Layer and reads the simulator's collision reports.  It never modifies a
collision filter, kinematically constrains the payload, or infers contact
from end-effector distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContactSample:
    left: bool
    right: bool
    pairs: dict[str, list[tuple[str, str]]]

    def as_dict(self) -> dict[str, object]:
        return {"left": self.left, "right": self.right, "pairs": self.pairs}


class GripperContactTracker:
    """Report contacts between each R1 gripper's two finger links and target."""

    def __init__(self, stage: Any, robot_path: str, target_path: str):
        from pxr import UsdPhysics

        self.stage = stage
        self.robot_path = robot_path.rstrip("/")
        self.target_path = target_path.rstrip("/")
        self._finger_links = {
            side: [
                f"{self.robot_path}/{side}_gripper_link1",
                f"{self.robot_path}/{side}_gripper_link2",
            ]
            for side in ("left", "right")
        }
        # A Session-Layer geometry repair can place the finite collision proxy
        # below each finger link.  PhysX raw contact data may be indexed by
        # that collision prim rather than the inherited link path, so retain
        # both query forms while still reporting contacts per finger side.
        self._finger_paths: dict[str, list[str]] = {}
        for side, links in self._finger_links.items():
            paths: list[str] = []
            for link in links:
                paths.append(link)
                for prim in stage.Traverse():
                    path = str(prim.GetPath())
                    if path.startswith(link + "/") and (
                        prim.HasAPI(UsdPhysics.CollisionAPI)
                        or "PhysicsCollisionProxy" in path
                        or "/collisions" in path
                    ):
                        paths.append(path)
            self._finger_paths[side] = sorted(set(paths))
        self._interface = None

    def prepare(self) -> None:
        """Apply contact reporting to the actual finger and target rigid bodies.

        Must run before physics initialization.  The stage's edit target is
        assumed to be its Session Layer by the physical runner.
        """

        from pxr import PhysxSchema, UsdPhysics

        paths = [path for paths in self._finger_paths.values() for path in paths]
        target_rigids: list[str] = []
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith(self.target_path) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                target_rigids.append(path)
        if not target_rigids:
            raise RuntimeError(f"target has no rigid body under {self.target_path}")
        for path in paths + target_rigids:
            prim = self.stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                raise RuntimeError(f"contact-report body is missing: {path}")
            report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            report.CreateThresholdAttr().Set(0.0)

    def initialize(self) -> None:
        from isaacsim.sensors.physics import _sensor

        self._interface = _sensor.acquire_contact_sensor_interface()

    def sample(self) -> ContactSample:
        if self._interface is None:
            raise RuntimeError("GripperContactTracker.initialize() has not completed")
        pairs: dict[str, list[tuple[str, str]]] = {"left": [], "right": []}
        for side, finger_paths in self._finger_paths.items():
            found: set[tuple[str, str]] = set()
            for finger_path in finger_paths:
                for contact in self._interface.get_rigid_body_raw_data(finger_path):
                    body0 = self._interface.decode_body_name(contact["body0"])
                    body1 = self._interface.decode_body_name(contact["body1"])
                    if body0.startswith(self.target_path) or body1.startswith(self.target_path):
                        found.add((body0, body1))
            pairs[side] = sorted(found)
        return ContactSample(bool(pairs["left"]), bool(pairs["right"]), pairs)
