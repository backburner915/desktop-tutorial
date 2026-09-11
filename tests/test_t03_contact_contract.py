"""Pure-Python contract checks for the frozen T03 contact instrumentation."""

from __future__ import annotations

import unittest

from episode_gen.sim_adapter import IsaacLabR1Adapter, R1_SUPPORT_PRIM_PATH
from r1_bimanual_dataset.core.contact_tracker import t03_contact_watch_specs


class TestT03ContactContract(unittest.TestCase):
    def test_support_is_mapped_only_by_explicit_prim_root(self) -> None:
        self.assertEqual(IsaacLabR1Adapter._normalise_contact_name(R1_SUPPORT_PRIM_PATH), "table")
        self.assertEqual(
            IsaacLabR1Adapter._normalise_contact_name(R1_SUPPORT_PRIM_PATH + "/collision"),
            "table",
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


if __name__ == "__main__":
    unittest.main()
