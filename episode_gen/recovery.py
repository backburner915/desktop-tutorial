"""Recovery policy: what to do once a detector fires.

One RecoveryManager instance lives for the duration of one episode. It
turns an AnomalyEvent into a RecoveryAction (retreat, then resume at
some phase with adjusted scenario parameters), and enforces the R20/F01
"recovery_exhausted" cutoff from the taxonomy doc — this is the only
place `max_recovery_attempts` is checked, detectors never see it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from episode_gen.fsm import Phase
from episode_gen.scenario import ScenarioConfig
from episode_gen.types import AnomalyEvent

_RESUME_PHASE_FOR_CLASS: dict[str, Phase] = {
    "R11": Phase.APPROACH,
    "R12": Phase.APPROACH,
    "R13": Phase.APPROACH,
    "R14": Phase.APPROACH,
}


@dataclass
class RecoveryAction:
    next_phase: Phase  # Phase.RETREAT normally, Phase.FAILURE once exhausted
    resume_phase: Phase  # phase to resume into once RETREAT completes
    note: str


@dataclass
class RecoveryManager:
    recovery_count: int = 0
    history: list[AnomalyEvent] = field(default_factory=list)

    def handle(self, event: AnomalyEvent, scenario: ScenarioConfig) -> RecoveryAction:
        self.history.append(event)
        anomaly = scenario.anomaly
        if anomaly is None:
            raise ValueError("recovery invoked but scenario has no anomaly config")

        if self.recovery_count >= anomaly.max_recovery_attempts:
            return RecoveryAction(
                next_phase=Phase.FAILURE,
                resume_phase=Phase.FAILURE,
                note=(
                    f"recovery_exhausted: {event.scenario_class} recurred after "
                    f"{self.recovery_count} attempts (taxonomy R20 -> F01)"
                ),
            )

        self.recovery_count += 1
        self._apply_retry_adjustment(event, scenario)
        resume = _RESUME_PHASE_FOR_CLASS.get(event.scenario_class, Phase.APPROACH)
        return RecoveryAction(
            next_phase=Phase.RETREAT,
            resume_phase=resume,
            note=(
                f"{event.scenario_class} on arm={event.arm} at t={event.t:.2f}s, "
                f"recovery attempt {self.recovery_count}/{anomaly.max_recovery_attempts}"
            ),
        )

    @staticmethod
    def _apply_retry_adjustment(event: AnomalyEvent, scenario: ScenarioConfig) -> None:
        """Mutate the scenario in place so the next APPROACH attempt is
        actually different, not a verbatim replay of the failing one."""
        anomaly = scenario.anomaly
        assert anomaly is not None

        def _offset(base: tuple[float, float, float]) -> tuple[float, float, float]:
            dx, dy, dz = anomaly.retry_grasp_offset_delta
            return (base[0] + dx, base[1] + dy, base[2] + dz)

        if event.scenario_class in ("R11", "R12"):
            if event.arm in ("left", "both"):
                scenario.left_approach_angle += anomaly.retry_approach_angle_delta
                scenario.left_grasp_offset = _offset(scenario.left_grasp_offset)
            if event.arm in ("right", "both"):
                scenario.right_approach_angle += anomaly.retry_approach_angle_delta
                scenario.right_grasp_offset = _offset(scenario.right_grasp_offset)
        elif event.scenario_class == "R13":
            if event.arm == "left":
                scenario.left_grasp_offset = _offset(scenario.left_grasp_offset)
            else:
                scenario.right_grasp_offset = _offset(scenario.right_grasp_offset)
        elif event.scenario_class == "R14":
            # widen the approach corridor between the two arms
            scenario.left_approach_angle += anomaly.retry_approach_angle_delta
            scenario.right_approach_angle -= anomaly.retry_approach_angle_delta
