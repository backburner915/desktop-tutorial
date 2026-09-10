"""Closed-loop validation for the R15-R19 closures and the F02
compound-failure path, mirroring test_r11_recovery_loop.py's approach:
MockSimAdapter injects each anomaly's fault signature exactly once, so a
correct episode_runner should recover (R15-R19) or fail without a second
retreat (F02).
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from episode_gen.episode_runner import EpisodeRunConfig, EpisodeRunner
from episode_gen.scenario import ScenarioConfig
from episode_gen.sim_adapter import MockSimAdapter

R_CONFIGS = {
    "R15": "configs/scenarios/r15_grip_slip.yaml",
    "R16": "configs/scenarios/r16_object_displaced.yaml",
    "R17": "configs/scenarios/r17_stalled_approach.yaml",
    "R18": "configs/scenarios/r18_joint_limit_margin.yaml",
    "R19": "configs/scenarios/r19_lift_failed.yaml",
}
F02_CONFIG = "configs/scenarios/f02_compound_failure.yaml"


class TestR15ThroughR19(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, scenario: ScenarioConfig):
        adapter = MockSimAdapter()
        runner = EpisodeRunner(adapter)
        run_cfg = EpisodeRunConfig(out_dir=str(self.tmp / scenario.scenario_id), episode_index=0, task_index=0)
        return runner.run(scenario, run_cfg)

    def test_each_closure_recovers_exactly_once(self) -> None:
        for class_id, path in R_CONFIGS.items():
            with self.subTest(class_id=class_id):
                scenario = ScenarioConfig.from_yaml(path)
                trace = self._run(scenario)
                self.assertTrue(trace.result.success, f"{class_id}: phase_log={trace.phase_log}")
                self.assertEqual(trace.result.recovery_count, 1, f"{class_id}: phase_log={trace.phase_log}")
                self.assertEqual(trace.phase_log.count("RETREAT"), 1, f"{class_id}: phase_log={trace.phase_log}")

    def test_recovery_exhausted_still_works_for_new_classes(self) -> None:
        scenario = ScenarioConfig.from_yaml(R_CONFIGS["R19"])
        assert scenario.anomaly is not None
        scenario.anomaly.max_recovery_attempts = 0
        trace = self._run(scenario)
        self.assertFalse(trace.result.success)
        assert trace.result.failure_reason is not None
        self.assertTrue(trace.result.failure_reason.startswith("F01_recovery_exhausted:R19"))


class TestF02CompoundFailure(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, scenario: ScenarioConfig):
        adapter = MockSimAdapter()
        runner = EpisodeRunner(adapter)
        run_cfg = EpisodeRunConfig(out_dir=str(self.tmp / scenario.scenario_id), episode_index=0, task_index=0)
        return runner.run(scenario, run_cfg)

    def test_secondary_anomaly_during_recovery_fails_without_second_retreat(self) -> None:
        scenario = ScenarioConfig.from_yaml(F02_CONFIG)
        self.assertEqual(scenario.family, "F")
        assert scenario.anomaly is not None and scenario.secondary_anomaly is not None

        trace = self._run(scenario)

        self.assertFalse(trace.result.success, f"phase_log={trace.phase_log}")
        assert trace.result.failure_reason is not None
        self.assertTrue(trace.result.failure_reason.startswith("F02_secondary_anomaly:R14"))
        self.assertIn("during_recovery_of:R11", trace.result.failure_reason)
        # exactly one retreat: for the primary R11 fault. The secondary R14
        # fault must go straight to FAILURE, not a second retreat/retry.
        self.assertEqual(trace.phase_log.count("RETREAT"), 1, f"phase_log={trace.phase_log}")
        self.assertEqual(trace.phase_log[-1], "FAILURE")
        # recovery_count reflects only the primary's one successful retry
        self.assertEqual(trace.result.recovery_count, 1)

    def test_primary_alone_without_secondary_still_recovers(self) -> None:
        """Sanity check: the same primary (R11) config without a
        secondary_anomaly still succeeds — F02 requires both to fire."""
        scenario = ScenarioConfig.from_yaml(F02_CONFIG)
        scenario.secondary_anomaly = None
        scenario.family = "R"
        scenario.scenario_id = "F02_primary_only_control"
        trace = self._run(scenario)
        self.assertTrue(trace.result.success, f"phase_log={trace.phase_log}")


class TestPerturbationScenarios(unittest.TestCase):
    """P21-P23: boundary parameter values that should still complete
    normally — no anomaly, no recovery."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_boundary_configs_complete_without_recovery(self) -> None:
        for path in (
            "configs/scenarios/p21_min_pregrasp_distance.yaml",
            "configs/scenarios/p22_max_approach_angle.yaml",
            "configs/scenarios/p23_max_object_rotation.yaml",
            "configs/scenarios/p24_max_phase_offset.yaml",
        ):
            with self.subTest(path=path):
                scenario = ScenarioConfig.from_yaml(path)
                self.assertEqual(scenario.family, "P")
                adapter = MockSimAdapter()
                runner = EpisodeRunner(adapter)
                run_cfg = EpisodeRunConfig(out_dir=str(self.tmp / scenario.scenario_id), episode_index=0)
                trace = runner.run(scenario, run_cfg)
                self.assertTrue(trace.result.success, f"{path}: phase_log={trace.phase_log}")
                self.assertEqual(trace.result.recovery_count, 0)


if __name__ == "__main__":
    unittest.main()
