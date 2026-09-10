"""EpisodeRunner: the one control loop shared by every scenario.

    RESET -> PREGRASP -> APPROACH -> CONTACT_CHECK -> GRASP -> LIFT -> HOLD
          -> MOVE_TO_TARGET -> PLACE -> RELEASE -> SUCCESS
    (any phase) -> anomaly detected -> RETREAT -> resume_phase -> ...
    recovery exhausted -> FAILURE

This file must never grow a per-scenario branch ("if scenario_id ==
'R11': ..."). Everything scenario-specific lives in ScenarioConfig
(config), detectors.py (which anomaly, if any, can fire) and
recovery.py (how to respond) — see docs/scenario_taxonomy_v0.1.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from episode_gen.detectors import run_detectors, run_secondary_detector
from episode_gen.dataset_writer import EpisodeResult, EpisodeWriter
from episode_gen.fsm import NOMINAL_NEXT, TERMINAL_PHASES, Phase
from episode_gen.recovery import RecoveryManager
from episode_gen.scenario import ScenarioConfig
from episode_gen.sim_adapter import SimAdapter


@dataclass
class EpisodeRunConfig:
    out_dir: str
    episode_index: int
    task_index: int = 0
    max_steps: int = 2000  # hard safety cap independent of adapter.check_timeout


@dataclass
class EpisodeTrace:
    result: EpisodeResult
    phase_log: list[str]


class EpisodeRunner:
    def __init__(self, adapter: SimAdapter) -> None:
        self.adapter = adapter

    def run(self, scenario: ScenarioConfig, run_cfg: EpisodeRunConfig) -> EpisodeTrace:
        # recovery mutates approach_angle/grasp_offset on retry — never let
        # that leak into the caller's base config (batch_generate reuses it).
        scenario = scenario.clone()

        writer = EpisodeWriter(
            out_dir=run_cfg.out_dir,
            episode_index=run_cfg.episode_index,
            task_index=run_cfg.task_index,
            task_text=scenario.task_text,
            scenario_id=scenario.scenario_id,
            control_hz=scenario.control_hz,
        )
        recovery_mgr = RecoveryManager()

        obs = self.adapter.reset(scenario)
        phase = Phase.RESET
        resume_after_retreat = Phase.PREGRASP
        phase_log: list[str] = [phase.value]
        failure_reason: str | None = None
        steps = 0

        while phase not in TERMINAL_PHASES and steps < run_cfg.max_steps:
            steps += 1
            obs, action = self.adapter.step(phase, scenario)
            images = self.adapter.render_cameras()

            writer.append_frame(
                images=images,
                state=obs.qpos,
                action=action,
                timestamp=obs.t,
                diagnostics={
                    "scenario_id": scenario.scenario_id,
                    "phase": phase.value,
                    "contacts": sorted(obs.contacts),
                    "object_pose": obs.object_pos,
                    "object_velocity": obs.object_vel,
                    "left_contact": any(c[0].startswith("left_arm") for c in obs.contacts),
                    "right_contact": any(c[0].startswith("right_arm") for c in obs.contacts),
                    "left_grasp_error": obs.extra.get("left_grasp_alignment_error", 0.0),
                    "right_grasp_error": obs.extra.get("right_grasp_alignment_error", 0.0),
                    "recovery_count": recovery_mgr.recovery_count,
                    "config": scenario.to_dict(),
                    "seed": scenario.seed,
                },
            )

            if self.adapter.check_timeout(obs, scenario):
                failure_reason = "F03_timeout"
                phase = Phase.FAILURE
                phase_log.append(phase.value)
                break

            if phase != Phase.RETREAT:
                primary_event = run_detectors(scenario, phase, obs)

                # F02: a second, distinct anomaly firing while already
                # recovering from the first one is a compounding failure,
                # not something to retry — see taxonomy §3.
                if primary_event is None and scenario.secondary_anomaly is not None and recovery_mgr.recovery_count > 0:
                    secondary_event = run_secondary_detector(scenario, phase, obs)
                    if secondary_event is not None:
                        assert scenario.anomaly is not None
                        failure_reason = (
                            f"F02_secondary_anomaly:{secondary_event.scenario_class}"
                            f"_during_recovery_of:{scenario.anomaly.scenario_class}"
                        )
                        phase = Phase.FAILURE
                        phase_log.append(phase.value)
                        break

                if primary_event is not None:
                    action_result = recovery_mgr.handle(primary_event, scenario)
                    phase = action_result.next_phase
                    phase_log.append(phase.value)
                    if phase == Phase.FAILURE:
                        failure_reason = f"F01_recovery_exhausted:{primary_event.scenario_class}"
                        break
                    resume_after_retreat = action_result.resume_phase
                    continue

            if phase == Phase.RETREAT:
                phase = resume_after_retreat
                phase_log.append(phase.value)
                continue

            if phase == Phase.RELEASE:
                if self.adapter.check_place_success(obs, scenario):
                    phase = Phase.SUCCESS
                    phase_log.append(phase.value)
                continue

            if obs.extra.get("phase_complete", True):
                phase = NOMINAL_NEXT.get(phase, Phase.FAILURE)
                phase_log.append(phase.value)

        if phase not in TERMINAL_PHASES:
            failure_reason = failure_reason or "F03_timeout"
            phase = Phase.FAILURE
            phase_log.append(phase.value)

        success = phase == Phase.SUCCESS
        result = writer.finalize(
            success=success,
            failure_reason=None if success else failure_reason,
            recovery_count=recovery_mgr.recovery_count,
        )
        return EpisodeTrace(result=result, phase_log=phase_log)
