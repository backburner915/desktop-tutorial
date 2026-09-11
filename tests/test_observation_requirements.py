"""Fail-closed checks for detector inputs and real-adapter object pose."""

from __future__ import annotations

import unittest

from episode_gen.detectors import (
    MissingObservationFieldError,
    detect_r12,
    detect_r15,
    detect_r17,
    detect_r19,
)
from episode_gen.fsm import Phase
from episode_gen.scenario import AnomalyConfig, ScenarioConfig
from episode_gen.sim_adapter import IsaacLabR1Adapter, ObservationUnavailableError
from episode_gen.types import Observation


def _scenario(class_id: str, *, arm: str = "left") -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id=class_id,
        family="R",
        anomaly=AnomalyConfig(scenario_class=class_id, contact_arm=arm, surface="object"),
    )


def _observation(*, contacts: set[tuple[str, str]] | None = None, extra: dict | None = None) -> Observation:
    return Observation(
        t=0.0,
        qpos=[0.0] * 16,
        qvel=[0.0] * 16,
        left_ee_pos=(0.0, 0.0, 0.0),
        right_ee_pos=(0.0, 0.0, 0.0),
        object_pos=(0.0, 0.0, 0.0),
        contacts=contacts or set(),
        extra=extra or {},
    )


class TestObservationRequirements(unittest.TestCase):
    def test_r12_uses_finger_contact_and_rejects_missing_alignment(self) -> None:
        scenario = _scenario("R12")
        finger_contact = {("left_gripper_link1", "object")}
        with self.assertRaisesRegex(MissingObservationFieldError, "left_grasp_alignment_error"):
            detect_r12(scenario, Phase.APPROACH, _observation(contacts=finger_contact))
        event = detect_r12(
            scenario,
            Phase.APPROACH,
            _observation(
                contacts=finger_contact,
                extra={"left_grasp_alignment_error": 0.10},
            ),
        )
        self.assertIsNotNone(event)
        self.assertIsNone(
            detect_r12(
                scenario,
                Phase.APPROACH,
                _observation(
                    contacts={("left_arm_link6", "object")},
                    extra={"left_grasp_alignment_error": 0.10},
                ),
            )
        )

    def test_r15_rejects_missing_slip_distance(self) -> None:
        with self.assertRaisesRegex(MissingObservationFieldError, "left_grasp_slip_m"):
            detect_r15(_scenario("R15"), Phase.LIFT, _observation())

    def test_r17_rejects_missing_stall_duration(self) -> None:
        with self.assertRaisesRegex(MissingObservationFieldError, "left_approach_stall_s"):
            detect_r17(_scenario("R17"), Phase.APPROACH, _observation())

    def test_r19_rejects_missing_lift_result(self) -> None:
        with self.assertRaisesRegex(MissingObservationFieldError, "lift_stalled"):
            detect_r19(_scenario("R19"), Phase.LIFT, _observation())

    def test_real_adapter_rejects_yaml_object_pose_fallback(self) -> None:
        adapter = IsaacLabR1Adapter({})
        scenario = ScenarioConfig(scenario_id="N01", family="N")
        with self.assertRaisesRegex(ObservationUnavailableError, "refusing ScenarioConfig/YAML pose fallback"):
            adapter._read_object_pose(scenario)


if __name__ == "__main__":
    unittest.main()
