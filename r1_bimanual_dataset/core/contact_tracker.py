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


class ContactWatchUnresolvedError(RuntimeError):
    """A required T03 raw-contact watch cannot observe live collision geometry.

    This is deliberately fatal during bootstrap: a watch that resolves to no
    collision geometry must never degrade into an empty contact stream.
    """


@dataclass(frozen=True)
class ContactPair:
    """One raw PhysX contact with its authoritative configured roots.

    PhysX may identify a collision proxy below a link rather than the link
    prim itself. ``query_root`` and ``counterpart_root`` are therefore the
    frozen watch-spec roots, never inferred later from raw body path names.
    """

    query_body: str
    counterpart_body: str
    query_root: str
    counterpart_root: str

    def as_dict(self) -> dict[str, str]:
        return {
            "query_body": self.query_body,
            "counterpart_body": self.counterpart_body,
            "query_root": self.query_root,
            "counterpart_root": self.counterpart_root,
        }


@dataclass(frozen=True)
class ContactSample:
    left: bool
    right: bool
    pairs: dict[str, list[ContactPair]]

    def as_dict(self) -> dict[str, object]:
        return {
            "left": self.left,
            "right": self.right,
            "pairs": {
                name: [pair.as_dict() for pair in pairs]
                for name, pairs in self.pairs.items()
            },
        }


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
        self._watch_resolution: dict[str, dict[str, object]] = {}
        self._query_paths: dict[str, list[str]] = {}
        self._counterpart_collision_paths: dict[str, list[str]] = {}
        for spec in self.watch_specs:
            query_collision_paths = self._resolve_collision_paths(
                spec.query_roots, spec.name, "query"
            )
            counterpart_collision_paths = self._resolve_collision_paths(
                spec.counterpart_roots, spec.name, "counterpart"
            )
            # Query both the link root and every collision child. This retains
            # the legacy raw-data behavior while proving that each root owns
            # actual collision geometry before the watch is considered live.
            self._query_paths[spec.name] = sorted(
                set(spec.query_roots).union(query_collision_paths)
            )
            self._counterpart_collision_paths[spec.name] = counterpart_collision_paths
            self._watch_resolution[spec.name] = {
                "status": "RESOLVED",
                "query_roots": list(spec.query_roots),
                "query_collision_paths": query_collision_paths,
                "counterpart_roots": list(spec.counterpart_roots),
                "counterpart_collision_paths": counterpart_collision_paths,
            }
        self._interface = None

    @staticmethod
    def _under(path: str, root: str) -> bool:
        root = root.rstrip("/")
        return path == root or path.startswith(root + "/")

    def _resolve_collision_paths(
        self, roots: tuple[str, ...], watch_name: str, role: str
    ) -> list[str]:
        """Resolve collision geometry or report this watch explicitly invalid."""

        paths: set[str] = set()
        for root in roots:
            root = root.rstrip("/")
            prim = self.stage.GetPrimAtPath(root)
            if not prim or not prim.IsValid():
                self._watch_resolution[watch_name] = {
                    "status": "UNRESOLVED",
                    "role": role,
                    "root": root,
                    "reason": "prim_missing",
                }
                raise ContactWatchUnresolvedError(
                    f"UNRESOLVED contact watch '{watch_name}': {role} prim is missing: {root}"
                )
            root_paths: set[str] = set()
            if (
                prim.HasAPI(self._usd_physics.CollisionAPI)
                or "PhysicsCollisionProxy" in root
                or "/collisions" in root
            ):
                root_paths.add(root)
            for child in self.stage.Traverse():
                path = str(child.GetPath())
                if self._under(path, root) and (
                    child.HasAPI(self._usd_physics.CollisionAPI)
                    or "PhysicsCollisionProxy" in path
                    or "/collisions" in path
                ):
                    root_paths.add(path)
            if not root_paths:
                self._watch_resolution[watch_name] = {
                    "status": "UNRESOLVED",
                    "role": role,
                    "root": root,
                    "reason": "collision_missing",
                }
                raise ContactWatchUnresolvedError(
                    f"UNRESOLVED contact watch '{watch_name}': {role} prim has no collision geometry: {root}"
                )
            paths.update(root_paths)
        return sorted(paths)

    @property
    def watch_resolution(self) -> dict[str, dict[str, object]]:
        """Resolved paths for bootstrap assertions and smoke-record evidence."""

        return {
            name: {
                key: list(value) if isinstance(value, list) else value
                for key, value in report.items()
            }
            for name, report in self._watch_resolution.items()
        }

    def _report_paths(self) -> list[str]:
        """Bodies/proxies that need PhysX contact reporting for every watch."""

        paths = {path for watched in self._query_paths.values() for path in watched}
        for watched in self._counterpart_collision_paths.values():
            paths.update(watched)
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

    def _watch_pairs(self, spec: ContactWatchSpec) -> list[ContactPair]:
        assert self._interface is not None
        found: set[ContactPair] = set()
        for query_path in self._query_paths[spec.name]:
            for contact in self._interface.get_rigid_body_raw_data(query_path):
                body0 = self._interface.decode_body_name(contact["body0"])
                body1 = self._interface.decode_body_name(contact["body1"])
                for query_root in spec.query_roots:
                    for counterpart_root in spec.counterpart_roots:
                        if self._under(body0, query_root) and self._under(
                            body1, counterpart_root
                        ):
                            found.add(
                                ContactPair(
                                    query_body=body0,
                                    counterpart_body=body1,
                                    query_root=query_root,
                                    counterpart_root=counterpart_root,
                                )
                            )
                        elif self._under(body1, query_root) and self._under(
                            body0, counterpart_root
                        ):
                            found.add(
                                ContactPair(
                                    query_body=body1,
                                    counterpart_body=body0,
                                    query_root=query_root,
                                    counterpart_root=counterpart_root,
                                )
                            )
        return sorted(
            found,
            key=lambda pair: (
                pair.query_root,
                pair.counterpart_root,
                pair.query_body,
                pair.counterpart_body,
            ),
        )

    def sample(self) -> ContactSample:
        if self._interface is None:
            raise RuntimeError("GripperContactTracker.initialize() has not completed")
        pairs = {spec.name: self._watch_pairs(spec) for spec in self.watch_specs}
        return ContactSample(bool(pairs.get("left")), bool(pairs.get("right")), pairs)
