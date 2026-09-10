"""Closed-loop validation of the framework's control flow (not physics):

R11 = premature table contact during APPROACH -> detector fires ->
RecoveryManager sends the episode through RETREAT -> resumes APPROACH
with adjusted params -> succeeds. This is the first full closed loop
the teacher's memo asked to prioritize (docs/scenario_taxonomy_v0.1.md
§2, §6). Runs against MockSimAdapter so it needs no Isaac Sim.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from episode_gen.batch_generate import generate_batch
from episode_gen.episode_runner import EpisodeRunConfig, EpisodeRunner
from episode_gen.scenario import ScenarioConfig
from episode_gen.sim_adapter import MockSimAdapter

R11_YAML = "configs/scenarios/r11_premature_table_contact.yaml"
N01_YAML = "configs/scenarios/n01_nominal.yaml"


class TestR11RecoveryLoop(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, scenario: ScenarioConfig):
        adapter = MockSimAdapter()
        runner = EpisodeRunner(adapter)
        run_cfg = EpisodeRunConfig(out_dir=str(self.tmp / scenario.scenario_id), episode_index=0, task_index=0)
        return runner.run(scenario, run_cfg)

    def test_r11_single_recovery_then_success(self) -> None:
        scenario = ScenarioConfig.from_yaml(R11_YAML)
        trace = self._run(scenario)

        self.assertTrue(trace.result.success, f"expected success, phase_log={trace.phase_log}")
        self.assertIsNone(trace.result.failure_reason)
        self.assertEqual(trace.result.recovery_count, 1, f"phase_log={trace.phase_log}")
        self.assertEqual(trace.phase_log.count("RETREAT"), 1)
        self.assertEqual(trace.phase_log[-1], "SUCCESS")

    def test_frames_satisfy_dataset_contract(self) -> None:
        scenario = ScenarioConfig.from_yaml(R11_YAML)
        trace = self._run(scenario)
        out_dir = Path(trace.result.out_dir)

        episode_lines = (out_dir / "episode.jsonl").read_text().strip().splitlines()
        diag_lines = (out_dir / "diagnostics.jsonl").read_text().strip().splitlines()
        self.assertEqual(len(episode_lines), trace.result.num_frames)
        self.assertEqual(len(diag_lines), trace.result.num_frames)

        required_keys = (
            "observation.images.front", "observation.images.left_wrist",
            "observation.images.right_wrist", "observation.state", "action",
            "timestamp", "frame_index", "episode_index", "index", "task_index",
        )
        leak_keys = ("scenario_id", "phase", "contacts", "object_pose", "recovery_count")

        prev_ts = None
        for i, (ep_line, diag_line) in enumerate(zip(episode_lines, diag_lines)):
            frame = json.loads(ep_line)
            for key in required_keys:
                self.assertIn(key, frame)
            self.assertEqual(len(frame["observation.state"]), 16)
            self.assertEqual(len(frame["action"]), 16)
            self.assertEqual(frame["frame_index"], i)
            for key in leak_keys:
                self.assertNotIn(key, frame, "diagnostics field leaked into policy frame")

            if prev_ts is not None:
                self.assertAlmostEqual(frame["timestamp"] - prev_ts, 1.0 / 30.0, delta=0.003)
            prev_ts = frame["timestamp"]

            diag = json.loads(diag_line)
            self.assertEqual(diag["scenario_id"], "R11")

        meta = json.loads((out_dir / "episode_meta.json").read_text())
        self.assertTrue(meta["episode_success"])
        self.assertEqual(meta["recovery_count"], 1)

    def test_n01_nominal_has_no_recovery(self) -> None:
        scenario = ScenarioConfig.from_yaml(N01_YAML)
        trace = self._run(scenario)
        self.assertTrue(trace.result.success)
        self.assertEqual(trace.result.recovery_count, 0)
        self.assertNotIn("RETREAT", trace.phase_log)

    def test_recovery_exhausted_becomes_failure(self) -> None:
        scenario = ScenarioConfig.from_yaml(R11_YAML)
        assert scenario.anomaly is not None
        scenario.anomaly.max_recovery_attempts = 0  # zero budget -> must fail immediately
        trace = self._run(scenario)
        self.assertFalse(trace.result.success)
        assert trace.result.failure_reason is not None
        self.assertTrue(trace.result.failure_reason.startswith("F01_recovery_exhausted"))
        self.assertEqual(trace.phase_log[-1], "FAILURE")

    def test_all_r_family_closures_recover_and_succeed(self) -> None:
        """R11-R14 each fire their detector exactly once and recover
        (docs/scenario_taxonomy_v0.1.md §2, "首批实现")."""
        for path in (
            "configs/scenarios/r11_premature_table_contact.yaml",
            "configs/scenarios/r12_wrong_surface_contact.yaml",
            "configs/scenarios/r13_single_side_grasp.yaml",
            "configs/scenarios/r14_arm_arm_collision.yaml",
        ):
            with self.subTest(path=path):
                scenario = ScenarioConfig.from_yaml(path)
                trace = self._run(scenario)
                self.assertTrue(trace.result.success, f"{path}: phase_log={trace.phase_log}")
                self.assertEqual(trace.result.recovery_count, 1, f"{path}: phase_log={trace.phase_log}")
                self.assertEqual(trace.phase_log.count("RETREAT"), 1, f"{path}: phase_log={trace.phase_log}")

    def test_batch_generate_produces_valid_r11_variants(self) -> None:
        out_dir = self.tmp / "generated_r11"
        paths = generate_batch(R11_YAML, str(out_dir), n=8, seed=42)
        self.assertEqual(len(paths), 8)

        successes = 0
        for p in paths:
            cfg = ScenarioConfig.from_yaml(p)
            self.assertEqual(cfg.family, "R")
            assert cfg.anomaly is not None
            self.assertEqual(cfg.anomaly.scenario_class, "R11")
            trace = self._run(cfg)
            if trace.result.success:
                successes += 1
        self.assertEqual(successes, 8, "all sampled single-fault R11 variants should recover")


if __name__ == "__main__":
    unittest.main()
