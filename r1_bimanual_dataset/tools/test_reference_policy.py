"""Regression checks for retired reference execution and fail-closed recording."""
import json
import tempfile
import unittest
from pathlib import Path
from r1_bimanual_dataset.core.reference_policy import require_candidate, reject_legacy_execution

class ReferencePolicyTest(unittest.TestCase):
    def test_klt_aliases_cannot_be_selected(self):
        for name in ['small_KLT.usd','D:/assets/KLT_Bin/small_KLT.usd',None]:
            with self.assertRaises(RuntimeError):require_candidate(name)
    def test_candidates_require_explicit_selection(self):
        require_candidate('004_sugar_box_physics.usd')
        require_candidate('003_cracker_box_physics.usd')
    def test_legacy_executor_cannot_write(self):
        with self.assertRaisesRegex(RuntimeError,'INVALID_LEGACY_EXECUTOR'):reject_legacy_execution()
    def test_recorder_refuses_before_creating_directories(self):
        from types import SimpleNamespace
        from r1_bimanual_dataset.core.episode_recorder import EpisodeRecorder
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'must_not_exist'
            recorder=EpisodeRecorder(output,SimpleNamespace(scenario_id=1),{'cameras':{}},None,10)
            with self.assertRaisesRegex(RuntimeError,'INVALID_LEGACY_EXECUTOR'):recorder.start()
            self.assertFalse(output.exists())
    def test_retired_probe_does_not_overwrite_evidence(self):
        import subprocess,sys,hashlib
        root=Path(__file__).resolve().parents[2]
        path=root/'r1_bimanual_dataset/reports/small_klt_reference_probe_report.json'
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        process=subprocess.run([sys.executable,str(root/'r1_bimanual_dataset/tools/run_physical_klt_reference_probe.py')],capture_output=True,text=True)
        self.assertEqual(process.returncode,2,process.stderr)
        self.assertIn('INVALID_ASSET_COLLISION_MISMATCH',process.stdout)
        self.assertEqual(before,hashlib.sha256(path.read_bytes()).hexdigest())
    def test_frozen_audit_cannot_claim_pass(self):
        root=Path(__file__).resolve().parents[2]
        audit=json.loads((root/'reports/small_klt_physics_audit.json').read_text(encoding='utf-8'))
        self.assertEqual(audit['result'],'INVALID_ASSET_COLLISION_MISMATCH')
        self.assertFalse(audit['safe_for_data_reference'])
        registry=json.loads((root/'reports/invalid_reference_registry.json').read_text(encoding='utf-8'))
        self.assertIsNone(registry['active_reference'])
        self.assertFalse(registry['dataset_generation_enabled'])

if __name__=='__main__':unittest.main()
