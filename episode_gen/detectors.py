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


class MissingObservationFieldError(RuntimeError):
    """A detector input is absent, so the episode must not be labelled normal."""


def _arms(anomaly: AnomalyConfig) -> list[str]:
    return ["left", "right"] if anomaly.contact_arm == "both" else [anomaly.contact_arm]


def _touching(obs: Observation, a: str, b: str) -> bool:
    return (a, b) in obs.contacts or (b, a) in obs.contacts


def _required_extra(
    obs: Observation, scenario_class: str, phase: Phase, key: str
) -> object:
    if key not in obs.extra or obs.extra[key] is None:
        raise MissingObservationFieldError(
            f"{scenario_class} requires Observation.extra[{key!r}] during {phase.value}; "
            "refusing a silent normal result"
        )
    return obs.extra[key]


def _required_mapping_value(
    mapping: dict[str, object], scenario_class: str, phase: Phase, key: str
) -> object:
    if key not in mapping or mapping[key] is None:
        raise MissingObservationFieldError(
            f"{scenario_class} requires Observation field {key!r} during {phase.value}; "
            "refusing a silent normal result"
        )
    return mapping[key]


def detect_r11(scenario: ScenarioConfig, phase: Phase, obs: Observation) -> AnomalyEvent | None:
    """Premature surface contact: arm touches a surface during APPROACH
    while still far from its intended grasp frame."""
    an = scenario.anomaly
    if an is None or an.scenario_class != "R11" or phase != Phase.APPROACH:
        return None
    for arm in _arms(an):
        link = f"{arm}_arm_link6"
        if _touching(obs, link, an.surface):
            dist = _required_mapping_value(
                obs.dist_to_grasp_frame, "R11", phase, arm
            )
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
        finger_links = (f"{arm}_gripper_link1", f"{arm}_gripper_link2")
        if any(_touching(obs, link, "object") for link in finger_links):
            align_err = _required_extra(
                obs, "R12", phase, f"{arm}_grasp_alignment_error"
            )
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
    closed = _required_extra(obs, "R13", phase, "gripper_closed")
    if not isinstance(closed, dict) or any(arm not in closed for arm in ("left", "right")):
        raise MissingObservationFieldError(
            "R13 requires Observation.extra['gripper_closed'] for both arms during GRASP"
        )
    if not all(bool(closed[arm]) for arm in ("left", "right")):
        return None  # still closing — nothing to judge yet
    f_min = an.extra.get("grip_force_min", 5.0)
    left_force = _required_mapping_value(obs.grip_force, "R13", phase, "left")
    right_force = _required_mapping_value(obs.grip_force, "R13", phase, "right")
    left_ok = left_force >= f_min
    right_ok = right_force >= f_min
    if left_ok != right_ok:
        failed_arm = "right" if left_ok else "left"
        return AnomalyEvent(
            scenario_class="R13", phase=phase, t=obs.t, arm=failed_arm,
            detail={"left_grip_force": left_force,
                    "right_grip_force": right_force,
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


def detect_r15(scenario: ScenarioConfig, phase: Phase, obs: Observation) -> AnomalyEvent | None:
    """Grip slip: object keeps sliding relative to a closed gripper during
    LIFT/HOLD. The adapter is responsible for integrating the slip
    distance since GRASP and reporting it via
    `obs.extra["{arm}_grasp_slip_m"]` — this detector only thresholds it,
    it does not (and cannot, from one frame) compute a trend itself."""
    an = scenario.anomaly
    if an is None or an.scenario_class != "R15" or phase not in (Phase.LIFT, Phase.HOLD):
        return None
    threshold = an.extra.get("slip_threshold_m", 0.02)
    for arm in _arms(an):
        slip = _required_extra(obs, "R15", phase, f"{arm}_grasp_slip_m")
        if slip > threshold:
            return AnomalyEvent(
                scenario_class="R15", phase=phase, t=obs.t, arm=arm,
                detail={"slip_m": slip, "threshold": threshold},
            )
    return None


def detect_r16(scenario: ScenarioConfig, phase: Phase, obs: Observation) -> AnomalyEvent | None:
    """Object got pushed/displaced during APPROACH: current object_pos has
    drifted from the scenario's nominal object pose beyond a threshold."""
    an = scenario.anomaly
    if an is None or an.scenario_class != "R16" or phase != Phase.APPROACH:
        return None
    nominal = (scenario.object_x, scenario.object_y, scenario.object_z)
    displacement = sum((c - n) ** 2 for c, n in zip(obs.object_pos, nominal)) ** 0.5
    threshold = an.extra.get("displacement_threshold_m", 0.05)
    if displacement > threshold:
        return AnomalyEvent(
            scenario_class="R16", phase=phase, t=obs.t, arm="both",
            detail={"displacement_m": displacement, "threshold": threshold, "object_pos": obs.object_pos},
        )
    return None


def detect_r17(scenario: ScenarioConfig, phase: Phase, obs: Observation) -> AnomalyEvent | None:
    """Stalled approach: position error hasn't decreased for T_stall
    seconds. Like R15, the adapter integrates "how long has this arm been
    stalled" and reports it via `obs.extra["{arm}_approach_stall_s"]` —
    a single Observation can't tell a stall from normal progress on its
    own."""
    an = scenario.anomaly
    if an is None or an.scenario_class != "R17" or phase != Phase.APPROACH:
        return None
    stall_max = an.extra.get("stall_time_s", 1.0)
    for arm in _arms(an):
        stalled_for = _required_extra(obs, "R17", phase, f"{arm}_approach_stall_s")
        if stalled_for >= stall_max:
            return AnomalyEvent(
                scenario_class="R17", phase=phase, t=obs.t, arm=arm,
                detail={"stalled_for_s": stalled_for, "stall_max_s": stall_max},
            )
    return None


def detect_r18(scenario: ScenarioConfig, phase: Phase, obs: Observation) -> AnomalyEvent | None:
    """Any joint approaching its limit, in any phase."""
    an = scenario.anomaly
    if an is None or an.scenario_class != "R18":
        return None
    if not obs.joint_limit_margin:
        raise MissingObservationFieldError(
            f"R18 requires Observation.joint_limit_margin during {phase.value}; "
            "refusing a silent normal result"
        )
    margin_min = an.extra.get("margin_min", 0.05)
    joint, margin = min(obs.joint_limit_margin.items(), key=lambda kv: kv[1])
    if margin < margin_min:
        arm = "left" if joint.startswith("left") else "right" if joint.startswith("right") else "both"
        return AnomalyEvent(
            scenario_class="R18", phase=phase, t=obs.t, arm=arm,
            detail={"joint": joint, "margin": margin, "margin_min": margin_min},
        )
    return None


def detect_r19(scenario: ScenarioConfig, phase: Phase, obs: Observation) -> AnomalyEvent | None:
    """Lift failed: gripper(s) report closed but the object isn't rising
    with them. The adapter reports this directly via
    `obs.extra["lift_stalled"]` since it already tracks object height
    over the LIFT phase."""
    an = scenario.anomaly
    if an is None or an.scenario_class != "R19" or phase != Phase.LIFT:
        return None
    lift_stalled = _required_extra(obs, "R19", phase, "lift_stalled")
    if lift_stalled:
        return AnomalyEvent(
            scenario_class="R19", phase=phase, t=obs.t, arm=an.contact_arm,
            detail={"object_vel": obs.object_vel},
        )
    return None


DetectorFn = Callable[[ScenarioConfig, Phase, Observation], AnomalyEvent | None]

DETECTORS: dict[str, DetectorFn] = {
    "R11": detect_r11,
    "R12": detect_r12,
    "R13": detect_r13,
    "R14": detect_r14,
    "R15": detect_r15,
    "R16": detect_r16,
    "R17": detect_r17,
    "R18": detect_r18,
    "R19": detect_r19,
}


def run_detectors(scenario: ScenarioConfig, phase: Phase, obs: Observation) -> AnomalyEvent | None:
    if scenario.anomaly is None:
        return None
    detector = DETECTORS.get(scenario.anomaly.scenario_class)
    if detector is None:
        return None
    return detector(scenario, phase, obs)


def run_secondary_detector(scenario: ScenarioConfig, phase: Phase, obs: Observation) -> AnomalyEvent | None:
    """F02: a second, distinct anomaly firing while already recovering
    from the primary one. episode_runner only calls this once
    recovery_count > 0 — see docs/scenario_taxonomy_v0.1.md §3."""
    if scenario.secondary_anomaly is None:
        return None
    detector = DETECTORS.get(scenario.secondary_anomaly.scenario_class)
    if detector is None:
        return None
    # detectors read scenario.anomaly, not secondary_anomaly — swap them
    # in for the duration of this call so the same functions apply.
    primary, scenario.anomaly = scenario.anomaly, scenario.secondary_anomaly
    try:
        return detector(scenario, phase, obs)
    finally:
        scenario.anomaly = primary
