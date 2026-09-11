"""Pure-Python contract checks for the frozen T03 contact instrumentation."""

from __future__ import annotations

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from episode_gen.sim_adapter import (
    IsaacLabR1Adapter,
    R1_SUPPORT_PRIM_PATH,
    R1_TARGET_OBJECT_PRIM_PATH,
)
from r1_bimanual_dataset.core.contact_tracker import (
    ContactWatchSpec,
    ContactWatchUnresolvedError,
    GripperContactTracker,
    t03_contact_watch_specs,
)


class _FakePrim:
    def __init__(self, path: str, *, valid: bool = True, collision: bool = False) -> None:
        self.path = path
        self.valid = valid
        self.collision = collision

    def IsValid(self) -> bool:
        return self.valid

    def HasAPI(self, api: object) -> bool:
        return self.collision

    def GetPath(self) -> str:
        return self.path


class _FakeStage:
    def __init__(self, existing_paths: set[str], paths_with_collision: set[str]) -> None:
        self._prims = {
            path: _FakePrim(path, collision=path in paths_with_collision) for path in existing_paths
        }
        for path in tuple(paths_with_collision):
            if path.endswith("/collision"):
                root = path.removesuffix("/collision")
                self._prims.setdefault(root, _FakePrim(root))

    def GetPrimAtPath(self, path: str) -> _FakePrim:
        return self._prims.get(path, _FakePrim(path, valid=False))

    def Traverse(self) -> list[_FakePrim]:
        return list(self._prims.values())


def _fake_pxr() -> dict[str, ModuleType]:
    pxr = ModuleType("pxr")
    pxr.UsdPhysics = SimpleNamespace(CollisionAPI=object())
    return {"pxr": pxr}


class TestT03ContactContract(unittest.TestCase):
    def test_support_and_object_are_mapped_only_by_explicit_prim_root(self) -> None:
        self.assertEqual(IsaacLabR1Adapter._normalise_contact_name(R1_SUPPORT_PRIM_PATH), "table")
        self.assertEqual(
            IsaacLabR1Adapter._normalise_contact_name(R1_SUPPORT_PRIM_PATH + "/collision"),
            "table",
        )
        self.assertEqual(
            IsaacLabR1Adapter._normalise_contact_name(R1_TARGET_OBJECT_PRIM_PATH), "object"
        )
        self.assertEqual(
            IsaacLabR1Adapter._normalise_contact_name(
                R1_TARGET_OBJECT_PRIM_PATH + "/collision"
            ),
            "object",
        )
        self.assertEqual(IsaacLabR1Adapter._normalise_contact_name("/World/Elsewhere/Top"), "Top")
        self.assertEqual(
            IsaacLabR1Adapter._normalise_contact_name("/World/Legacy/crew_lock_bag_collision"),
            "crew_lock_bag_collision",
        )
        self.assertEqual(
            IsaacLabR1Adapter._normalise_contact_name("/World/Legacy/workbench_top"), "workbench_top")

    def test_t03_watch_specs_cover_target_table_and_all_arm_links(self) -> None:
        robot = "/World/R1"
        specs = {spec.name: spec for spec in t03_contact_watch_specs(robot, "/World/T03", "/World/Top")}
        self.assertEqual(set(specs), {"left", "right", "left_table", "right_table", "arm_arm"})
        self.assertEqual(specs["left_table"].query_roots, (robot + "/left_arm_link6",))
        self.assertEqual(specs["right_table"].query_roots, (robot + "/right_arm_link6",))
        self.assertEqual(specs["left_table"].counterpart_roots, ("/World/Top",))
        self.assertEqual(specs["arm_arm"].query_roots, tuple(f"{robot}/left_arm_link{i}" for i in range(1, 7)))
        self.assertEqual(specs["arm_arm"].counterpart_roots, tuple(f"{robot}/right_arm_link{i}" for i in range(1, 7)))

    def test_watch_without_collision_geometry_is_explicitly_unresolved(self) -> None:
        stage = _FakeStage(
            {"/World/query", "/World/counterpart", "/World/counterpart/collision"},
            {"/World/counterpart/collision"},
        )
        spec = ContactWatchSpec("test", ("/World/query",), ("/World/counterpart",))
        with patch.dict("sys.modules", _fake_pxr()):
            with self.assertRaisesRegex(ContactWatchUnresolvedError, "UNRESOLVED.*collision geometry"):
                GripperContactTracker(stage, "/World/R1", "/World/T03", watch_specs=(spec,))

    def test_resolved_watch_reports_its_collision_paths(self) -> None:
        stage = _FakeStage(
            {
                "/World/query",
                "/World/query/collision",
                "/World/counterpart",
                "/World/counterpart/collision",
            },
            {"/World/query/collision", "/World/counterpart/collision"},
        )
        spec = ContactWatchSpec("test", ("/World/query",), ("/World/counterpart",))
        with patch.dict("sys.modules", _fake_pxr()):
            tracker = GripperContactTracker(stage, "/World/R1", "/World/T03", watch_specs=(spec,))
        self.assertEqual(tracker.watch_resolution["test"]["status"], "RESOLVED")
        self.assertEqual(
            tracker.watch_resolution["test"]["query_collision_paths"], ["/World/query/collision"]
        )
        self.assertEqual(
            tracker.watch_resolution["test"]["counterpart_collision_paths"], ["/World/counterpart/collision"]
        )


if __name__ == "__main__":
    unittest.main()
