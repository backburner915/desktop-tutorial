"""Failure/anomaly detectors for the R-family closures (taxonomy §2).

Each function is a pure predicate: (scenario, phase, obs) -> AnomalyEvent
| None. A detector only fires for the scenario_class it implements, so
episode_runner can call `run_detectors` unconditionally every step
without branching on the scenario ID itself — the branching lives in
config (`scenario.anomaly.scenario_class`), not in code.

Adding "R11 but hitting a cabinet instead of a table" is a config change
(`anomaly.surface: "cabinet"`), not a new function — that's the
"closure" property the taxonomy doc asks for.
"""

from __future__ import annotations

from collections.abc import Callable

from episode_gen.fsm import Phase
from episode_gen.scenario import AnomalyConfig, ScenarioConfig
from episode_gen.types import AnomalyEvent, Observation


def _arms(anomaly: AnomalyConfig) -> list[str]:
    return ["left", "right"] if anomaly.contact_arm == "both" else [anomaly.contact_arm]


def _touching(obs: Observation, a: str, b: str) -> bool:
    return (a, b) in obs.contacts or (b, a) in obs.contacts


def detect_r11(scenario: ScenarioConfig, phase: Phase, obs: Observation) -> AnomalyEvent | None:
    """Premature surface contact: arm touches a surface during APPROACH
    while still far from its intended grasp frame."""
    an = scenario.anomaly
    if an is None or an.scenario_class != "R11" or phase != Phase.APPROACH:
        return None
    for arm in _arms(an):
        link = f"{arm}_arm_link6"
        if _touching(obs, link, an.surface):
            dist = obs.dist_to_grasp_frame.get(arm, 0.0)
            if dist > an.object_distance_at_contact:
                return AnomalyEvent(
                    scenario_class="R11", phase=phase, t=obs.t, arm=arm,
                    detail={"surface": an.surface, "dist_to_grasp_frame": dist},
                )
    return None


def detect_r12(scenario: ScenarioConfig, phase: Phase, obs: Observation) -> AnomalyEvent | None:
    """Contact with the object itself, but grasp alignment is wrong
    (touching the wrong face / at the wrong point)."""
    an = scenario.anomaly
    if an is None or an.scenario_class != "R12" or phase not in (Phase.APPROACH, Phase.CONTACT_CHECK):
        return None
    threshold = an.extra.get("alignment_error_threshold", 0.03)
    for arm in _arms(an):
        link = f"{arm}_arm_link6"
        if _touching(obs, link, "object"):
            align_err = obs.extra.get(f"{arm}_grasp_alignment_error", 0.0)
            if align_err > threshold:
                return AnomalyEvent(
                    scenario_class="R12", phase=phase, t=obs.t, arm=arm,
                    detail={"alignment_error": align_err, "threshold": threshold},
                )
    return None


def detect_r13(scenario: ScenarioConfig, phase: Phase, obs: Observation) -> AnomalyEvent | None:
    """Only one gripper actually established a grasp once both finished
    closing."""
    an = scenario.anomaly
    if an is None or an.scenario_class != "R13" or phase != Phase.GRASP:
        return None
    closed = obs.extra.get("gripper_closed")
    if not closed or not all(closed.values()):
        return None  # still closing — nothing to judge yet
    f_min = an.extra.get("grip_force_min", 5.0)
    left_ok = obs.grip_force.get("left", 0.0) >= f_min
    right_ok = obs.grip_force.get("right", 0.0) >= f_min
    if left_ok != right_ok:
        failed_arm = "right" if left_ok else "left"
        return AnomalyEvent(
            scenario_class="R13", phase=phase, t=obs.t, arm=failed_arm,
            detail={"left_grip_force": obs.grip_force.get("left", 0.0),
                    "right_grip_force": obs.grip_force.get("right", 0.0),
                    "f_min": f_min},
        )
    return None


def detect_r14(scenario: ScenarioConfig, phase: Phase, obs: Observation) -> AnomalyEvent | None:
    """The two arms collide with each other, in any phase."""
    an = scenario.anomaly
    if an is None or an.scenario_class != "R14":
        return None
    for a, b in obs.contacts:
        if (a.startswith("left_arm") and b.startswith("right_arm")) or (
            b.startswith("left_arm") and a.startswith("right_arm")
        ):
            return AnomalyEvent(
                scenario_class="R14", phase=phase, t=obs.t, arm="both",
                detail={"link_a": a, "link_b": b},
            )
    return None


DetectorFn = Callable[[ScenarioConfig, Phase, Observation], AnomalyEvent | None]

DETECTORS: dict[str, DetectorFn] = {
    "R11": detect_r11,
    "R12": detect_r12,
    "R13": detect_r13,
    "R14": detect_r14,
}


def run_detectors(scenario: ScenarioConfig, phase: Phase, obs: Observation) -> AnomalyEvent | None:
    if scenario.anomaly is None:
        return None
    detector = DETECTORS.get(scenario.anomaly.scenario_class)
    if detector is None:
        return None
    return detector(scenario, phase, obs)
