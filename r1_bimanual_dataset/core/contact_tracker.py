"""Real PhysX contact evidence for the R1 grippers and arms.

All reporting is authored only in the active Session Layer. The tracker never
changes collision filtering, constrains the payload, or infers contact from
end-effector distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContactWatchSpec:
    """One directional raw-contact query.

    ``query_roots`` are queried from PhysX. A raw contact is retained only
    when either reported body is below a ``counterpart_roots`` path. Keeping
    this relation explicit prevents a target-only filter from silently
    discarding support or arm contacts.
    """

    name: str
    query_roots: tuple[str, ...]
    counterpart_roots: tuple[str, ...]


@dataclass(frozen=True)
class ContactSample:
    left: bool
    right: bool
    pairs: dict[str, list[tuple[str, str]]]

    def as_dict(self) -> dict[str, object]:
        return {"left": self.left, "right": self.right, "pairs": self.pairs}


def t03_contact_watch_specs(
    robot_path: str, target_path: str, support_path: str
) -> tuple[ContactWatchSpec, ...]:
    """Return the frozen T03 contact contract without touching a stage.

    ``left`` and ``right`` preserve the established two-finger-to-target
    evidence. ``left_table``/``right_table`` are the R11 link6 watches.
    ``arm_arm`` watches every left arm link against every right arm link
    because R14 is defined over any arm-link collision.
    """

    robot = robot_path.rstrip("/")
    target = target_path.rstrip("/")
    support = support_path.rstrip("/")
    fingers = {
        side: tuple(f"{robot}/{side}_gripper_link{index}" for index in (1, 2))
        for side in ("left", "right")
    }
    arms = {
        side: tuple(f"{robot}/{side}_arm_link{index}" for index in range(1, 7))
        for side in ("left", "right")
    }
    return (
        ContactWatchSpec("left", fingers["left"], (target,)),
        ContactWatchSpec("right", fingers["right"], (target,)),
        ContactWatchSpec("left_table", (arms["left"][-1],), (support,)),
        ContactWatchSpec("right_table", (arms["right"][-1],), (support,)),
        ContactWatchSpec("arm_arm", arms["left"], arms["right"]),
    )


class GripperContactTracker:
    """Read explicit PhysX raw-contact watches from the live scene.

    The legacy three-argument constructor remains target/finger-only. A caller
    must pass explicit ``watch_specs`` to enable R11/R14 monitoring; this keeps
    old collection behaviour unchanged until the new bootstrap injects the
    frozen T03 watch set.
    """

    def __init__(
        self,
        stage: Any,
        robot_path: str,
        target_path: str,
        *,
        watch_specs: tuple[ContactWatchSpec, ...] | None = None,
    ):
        from pxr import UsdPhysics

        self.stage = stage
        self.robot_path = robot_path.rstrip("/")
        self.target_path = target_path.rstrip("/")
        self._usd_physics = UsdPhysics
        if watch_specs is None:
            watch_specs = t03_contact_watch_specs(
                self.robot_path,
                self.target_path,
                "/__legacy_unused_support__",
            )[:2]
        if not watch_specs:
            raise ValueError("GripperContactTracker needs at least one watch spec")
        names = [spec.name for spec in watch_specs]
        if len(names) != len(set(names)):
            raise ValueError(f"contact watch names must be unique: {names}")
        self.watch_specs = watch_specs
        self._query_paths = {
            spec.name: self._expand_query_paths(spec.query_roots, spec.name)
            for spec in self.watch_specs
        }
        self._interface = None

    @staticmethod
    def _under(path: str, root: str) -> bool:
        root = root.rstrip("/")
        return path == root or path.startswith(root + "/")

    def _expand_query_paths(self, roots: tuple[str, ...], watch_name: str) -> list[str]:
        paths: set[str] = set()
        for root in roots:
            root = root.rstrip("/")
            prim = self.stage.GetPrimAtPath(root)
            if not prim or not prim.IsValid():
                raise RuntimeError(f"contact-watch root is missing ({watch_name}): {root}")
            paths.add(root)
            for child in self.stage.Traverse():
                path = str(child.GetPath())
                if self._under(path, root) and (
                    child.HasAPI(self._usd_physics.CollisionAPI)
                    or "PhysicsCollisionProxy" in path
                    or "/collisions" in path
                ):
                    paths.add(path)
        return sorted(paths)

    def _report_paths(self) -> list[str]:
        """Bodies/proxies that need PhysX contact reporting for every watch."""

        paths = {path for watched in self._query_paths.values() for path in watched}
        for spec in self.watch_specs:
            for root in spec.counterpart_roots:
                root = root.rstrip("/")
                prim = self.stage.GetPrimAtPath(root)
                if not prim or not prim.IsValid():
                    raise RuntimeError(
                        f"contact-watch counterpart is missing ({spec.name}): {root}"
                    )
                for child in self.stage.Traverse():
                    path = str(child.GetPath())
                    if self._under(path, root) and (
                        child.HasAPI(self._usd_physics.RigidBodyAPI)
                        or child.HasAPI(self._usd_physics.CollisionAPI)
                        or "PhysicsCollisionProxy" in path
                        or "/collisions" in path
                    ):
                        paths.add(path)
        return sorted(paths)

    def prepare(self) -> None:
        """Enable reporting for configured watches before physics starts."""

        from pxr import PhysxSchema

        report_paths = self._report_paths()
        if not report_paths:
            raise RuntimeError("contact watches resolved to no reportable bodies")
        for path in report_paths:
            prim = self.stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                raise RuntimeError(f"contact-report body is missing: {path}")
            report = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            report.CreateThresholdAttr().Set(0.0)

    def initialize(self) -> None:
        from isaacsim.sensors.physics import _sensor

        self._interface = _sensor.acquire_contact_sensor_interface()

    def _watch_pairs(self, spec: ContactWatchSpec) -> list[tuple[str, str]]:
        assert self._interface is not None
        found: set[tuple[str, str]] = set()
        for query_path in self._query_paths[spec.name]:
            for contact in self._interface.get_rigid_body_raw_data(query_path):
                body0 = self._interface.decode_body_name(contact["body0"])
                body1 = self._interface.decode_body_name(contact["body1"])
                if any(
                    self._under(body0, root) or self._under(body1, root)
                    for root in spec.counterpart_roots
                ):
                    found.add((body0, body1))
        return sorted(found)

    def sample(self) -> ContactSample:
        if self._interface is None:
            raise RuntimeError("GripperContactTracker.initialize() has not completed")
        pairs = {spec.name: self._watch_pairs(spec) for spec in self.watch_specs}
        return ContactSample(bool(pairs.get("left")), bool(pairs.get("right")), pairs)
