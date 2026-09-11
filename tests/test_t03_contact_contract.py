"""Pure-Python contract checks for the frozen T03 contact instrumentation."""

from __future__ import annotations

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from episode_gen.sim_adapter import (
    ContactRootTokenError,
    IsaacLabR1Adapter,
    R1_ROBOT_PRIM_PATH,
    R1_SUPPORT_PRIM_PATH,
    R1_TARGET_OBJECT_PRIM_PATH,
)
from r1_bimanual_dataset.core.contact_tracker import (
    ContactPair,
    ContactSample,
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


class _FakeContactInterface:
    def __init__(self, contacts_by_query_path: dict[str, list[dict[str, str]]]) -> None:
        self.contacts_by_query_path = contacts_by_query_path

    def get_rigid_body_raw_data(self, query_path: str) -> list[dict[str, str]]:
        return self.contacts_by_query_path.get(query_path, [])

    @staticmethod
    def decode_body_name(value: str) -> str:
        return value


class TestT03ContactContract(unittest.TestCase):
    def _sample_for_collision_proxy(
        self, watch_name: str, query_root: str, counterpart_root: str
    ):
        query_proxy = query_root + "/collisions/mesh_0"
        counterpart_proxy = counterpart_root + "/collisions/mesh_0"
        stage = _FakeStage(
            {query_root, query_proxy, counterpart_root, counterpart_proxy},
            {query_proxy, counterpart_proxy},
        )
        spec = ContactWatchSpec(watch_name, (query_root,), (counterpart_root,))
        with patch.dict("sys.modules", _fake_pxr()):
            tracker = GripperContactTracker(
                stage, R1_ROBOT_PRIM_PATH, R1_TARGET_OBJECT_PRIM_PATH, watch_specs=(spec,)
            )
        tracker._interface = _FakeContactInterface(
            {query_proxy: [{"body0": query_proxy, "body1": counterpart_proxy}]}
        )
        return tracker.sample().pairs[watch_name][0]

    def test_arm_collision_proxy_is_mapped_by_watch_root(self) -> None:
        query_root = R1_ROBOT_PRIM_PATH + "/left_arm_link6"
        pair = self._sample_for_collision_proxy("left_table", query_root, R1_SUPPORT_PRIM_PATH)
        self.assertEqual(pair.query_root, query_root)
        self.assertEqual(pair.counterpart_root, R1_SUPPORT_PRIM_PATH)
        self.assertTrue(pair.query_body.endswith("/collisions/mesh_0"))
        contacts = IsaacLabR1Adapter._contacts_from_tracker_sample(
            ContactSample(left=False, right=False, pairs={"left_table": [pair]})
        )
        self.assertEqual(contacts, {("left_arm_link6", "table")})

    def test_finger_collision_proxy_is_mapped_by_watch_root(self) -> None:
        query_root = R1_ROBOT_PRIM_PATH + "/right_gripper_link2"
        pair = self._sample_for_collision_proxy("right", query_root, R1_TARGET_OBJECT_PRIM_PATH)
        self.assertEqual(pair.query_root, query_root)
        self.assertEqual(pair.counterpart_root, R1_TARGET_OBJECT_PRIM_PATH)
        self.assertTrue(pair.query_body.endswith("/collisions/mesh_0"))
        contacts = IsaacLabR1Adapter._contacts_from_tracker_sample(
            ContactSample(left=False, right=True, pairs={"right": [pair]})
        )
        self.assertEqual(contacts, {("right_gripper_link2", "object")})

    def test_unmapped_watch_root_is_not_silently_tokenized(self) -> None:
        pair = ContactPair(
            query_body="/World/Unknown/collisions/mesh_0",
            counterpart_body=R1_TARGET_OBJECT_PRIM_PATH + "/collisions/mesh_0",
            query_root="/World/Unknown",
            counterpart_root=R1_TARGET_OBJECT_PRIM_PATH,
        )
        with self.assertRaisesRegex(ContactRootTokenError, "unmapped watch root"):
            IsaacLabR1Adapter._contacts_from_tracker_sample(
                ContactSample(left=False, right=False, pairs={"unknown": [pair]})
            )

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
