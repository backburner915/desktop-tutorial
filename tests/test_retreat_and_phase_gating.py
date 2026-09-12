"""Phase advancement must come from the adapter, never from a default.

Two properties are load-bearing under real physics but invisible against a
mock that always reports a phase as finished:

  - RETREAT runs until the adapter says it finished. Ending it after one
    control step leaves the arm at the pose that triggered the anomaly, so
    every retry re-triggers it and each R-family episode burns through
    recovery.
  - A missing ``phase_complete`` is an error, not "advance now".
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from episode_gen.detectors import MissingObservationFieldError
from episode_gen.episode_runner import EpisodeRunConfig, EpisodeRunner, _side_contact
from episode_gen.fsm import Phase
from episode_gen.scenario import JOINT_ORDER, ScenarioConfig
from episode_gen.sim_adapter import CAMERA_NAMES, MockSimAdapter, SimAdapter
from episode_gen.types import Observation

R11_YAML = "configs/scenarios/r11_premature_table_contact.yaml"

RETREAT_STEPS = 5


class SlowRetreatAdapter(SimAdapter):
    """Mock whose RETREAT takes several control steps, like real physics."""

    control_hz = 30.0

    def __init__(self, *, report_phase_complete: bool = True) -> None:
        self.report_phase_complete = report_phase_complete
        self.phase_steps: list[Phase] = []
        self._t = 0.0
        self._retreat_progress = 0
        self._approach_attempt = 0

    def reset(self, scenario: ScenarioConfig) -> Observation:
        self._t = 0.0
        return self._observe(Phase.RESET, complete=True, contacts=set())

    def step(self, phase: Phase, scenario: ScenarioConfig) -> tuple[Observation, list[float]]:
        self.phase_steps.append(phase)
        self._t += 1.0 / self.control_hz

        if phase == Phase.RETREAT:
            self._retreat_progress += 1
            return self._observe(
                phase, complete=self._retreat_progress >= RETREAT_STEPS, contacts=set()
            ), [0.0] * len(JOINT_ORDER)

        # Fire R11 on the first APPROACH only, so a correct runner recovers once.
        contacts: set[tuple[str, str]] = set()
        if phase == Phase.APPROACH:
            self._approach_attempt += 1
            if self._approach_attempt == 1:
                contacts = {("left_arm_link6", "table")}
        return self._observe(phase, complete=True, contacts=contacts), [0.0] * len(JOINT_ORDER)

    def _observe(
        self, phase: Phase, *, complete: bool, contacts: set[tuple[str, str]]
    ) -> Observation:
        extra: dict[str, Any] = {
            "left_grasp_alignment_error": 0.0,
            "right_grasp_alignment_error": 0.0,
            "left_grasp_slip_m": 0.0,
            "right_grasp_slip_m": 0.0,
            "left_approach_stall_s": 0.0,
            "right_approach_stall_s": 0.0,
            "lift_stalled": False,
            "gripper_closed": {"left": True, "right": True},
        }
        if self.report_phase_complete:
            extra["phase_complete"] = complete
        return Observation(
            t=self._t,
            qpos=[0.0] * len(JOINT_ORDER),
            qvel=[0.0] * len(JOINT_ORDER),
            left_ee_pos=(0.0, 0.2, 1.0),
            right_ee_pos=(0.0, -0.2, 1.0),
            object_pos=(0.4, 0.0, 1.0),
            contacts=contacts,
            dist_to_grasp_frame={"left": 1.0, "right": 1.0},
            grip_force={"left": 10.0, "right": 10.0},
            joint_limit_margin={joint: 1.0 for joint in JOINT_ORDER},
            extra=extra,
        )

    def render_cameras(self) -> dict[str, Any]:
        return {name: {"shape": (480, 640, 3), "dtype": "uint8"} for name in CAMERA_NAMES}

    def check_place_success(self, obs: Observation, scenario: ScenarioConfig) -> bool:
        return True

    def check_timeout(self, obs: Observation, scenario: ScenarioConfig) -> bool:
        return obs.t > 30.0


class TestRetreatRunsToCompletion(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, adapter: SimAdapter):
        runner = EpisodeRunner(adapter)
        return runner.run(
            ScenarioConfig.from_yaml(R11_YAML),
            EpisodeRunConfig(out_dir=str(self.tmp / "ep"), episode_index=0, task_index=0),
        )

    def test_retreat_gets_every_step_the_adapter_asks_for(self) -> None:
        adapter = SlowRetreatAdapter()
        trace = self._run(adapter)

        retreat_steps = [p for p in adapter.phase_steps if p == Phase.RETREAT]
        self.assertEqual(
            len(retreat_steps),
            RETREAT_STEPS,
            f"RETREAT must run until the adapter reports it complete, phase_log={trace.phase_log}",
        )
        self.assertIn("RETREAT", trace.phase_log)
        self.assertTrue(trace.result.success, f"phase_log={trace.phase_log}")

    def test_missing_phase_complete_is_an_error_not_an_advance(self) -> None:
        adapter = SlowRetreatAdapter(report_phase_complete=False)
        with self.assertRaises(MissingObservationFieldError):
            self._run(adapter)

    def test_mock_adapter_still_reports_phase_complete(self) -> None:
        # The shipped mock must satisfy the same contract as a real adapter.
        obs = MockSimAdapter().reset(ScenarioConfig.from_yaml(R11_YAML))
        self.assertIn("phase_complete", obs.extra)


class TestSideContact(unittest.TestCase):
    def test_gripper_link_counts_as_side_contact(self) -> None:
        self.assertTrue(_side_contact({("left_gripper_link1", "object")}, "left"))

    def test_pair_order_does_not_matter(self) -> None:
        self.assertTrue(_side_contact({("object", "right_gripper_link2")}, "right"))

    def test_other_side_is_not_reported(self) -> None:
        self.assertFalse(_side_contact({("left_arm_link6", "table")}, "right"))
