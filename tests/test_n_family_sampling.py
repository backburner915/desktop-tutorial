"""Sanity-checks episode_gen/n_family_sampling.py: every N02-N10 generator
produces valid, family=N, anomaly-free ScenarioConfigs that complete
successfully through MockSimAdapter with zero recovery — these are
"normal" variants by taxonomy definition, so a fault firing would mean
the generator (or the FSM) is broken, not that the episode is genuinely
anomalous.
"""

from __future__ import annotations

import random
import shutil
import tempfile
import unittest
from pathlib import Path

from episode_gen.episode_runner import EpisodeRunConfig, EpisodeRunner
from episode_gen.n_family_sampling import GENERATORS
from episode_gen.scenario import ScenarioConfig
from episode_gen.sim_adapter import MockSimAdapter

N01_YAML = "configs/scenarios/n01_nominal.yaml"


class TestNFamilySampling(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.base = ScenarioConfig.from_yaml(N01_YAML)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, scenario: ScenarioConfig):
        adapter = MockSimAdapter()
        runner = EpisodeRunner(adapter)
        run_cfg = EpisodeRunConfig(out_dir=str(self.tmp / scenario.scenario_id), episode_index=0)
        return runner.run(scenario, run_cfg)

    def test_each_generator_produces_n_valid_configs(self) -> None:
        rng = random.Random(0)
        for n_id, generator in GENERATORS.items():
            with self.subTest(n_id=n_id):
                configs = generator(self.base, rng, 5)
                self.assertEqual(len(configs), 5)
                ids = {c.scenario_id for c in configs}
                self.assertEqual(len(ids), 5, "scenario_ids must be unique within a batch")
                for cfg in configs:
                    self.assertEqual(cfg.family, "N")
                    self.assertIsNone(cfg.anomaly)

    def test_each_generator_completes_without_recovery(self) -> None:
        rng = random.Random(1)
        for n_id, generator in GENERATORS.items():
            with self.subTest(n_id=n_id):
                configs = generator(self.base, rng, 3)
                for cfg in configs:
                    trace = self._run(cfg)
                    self.assertTrue(trace.result.success, f"{cfg.scenario_id}: phase_log={trace.phase_log}")
                    self.assertEqual(trace.result.recovery_count, 0, f"{cfg.scenario_id}: phase_log={trace.phase_log}")

    def test_n09_perimeter_points_are_deterministic_and_bounded(self) -> None:
        rng = random.Random(2)
        configs = GENERATORS["N09"](self.base, rng, 16)
        for cfg in configs:
            dx = cfg.object_x - self.base.object_x
            dy = cfg.object_y - self.base.object_y
            self.assertAlmostEqual(max(abs(dx), abs(dy)), 0.025, places=6)

    def test_n08_distribution_roughly_matches_80_10_10(self) -> None:
        rng = random.Random(3)
        configs = GENERATORS["N08"](self.base, rng, 500)
        counts = {"synchronous": 0, "left_leads": 0, "right_leads": 0}
        for cfg in configs:
            counts[cfg.coordination_mode] += 1
            if cfg.coordination_mode == "synchronous":
                self.assertEqual(cfg.phase_offset, 0.0)
            else:
                self.assertTrue(0.0 <= cfg.phase_offset <= 0.060)
        self.assertGreater(counts["synchronous"], 350)  # ~80% of 500, generous margin
        self.assertGreater(counts["left_leads"], 20)
        self.assertGreater(counts["right_leads"], 20)


if __name__ == "__main__":
    unittest.main()
