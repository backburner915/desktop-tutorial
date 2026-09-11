"""Fail-closed policy for retired references and unqualified dataset writers."""
from pathlib import Path

INVALID_KLT = "INVALID_ASSET_COLLISION_MISMATCH"
RAW_ROOT = Path("D:/r1_bimanual_dataset_raw")


def require_candidate(asset):
    if not asset:
        raise RuntimeError("CANDIDATE_REQUIRED: select an explicitly audited asset")
    if "klt" in str(asset).casefold():
        raise RuntimeError(INVALID_KLT + ": KLT is retained for regression evidence only")


def reject_legacy_execution():
    raise RuntimeError("INVALID_LEGACY_EXECUTOR: Reference Gates A–E have not qualified a replacement; no episode may be written")


def require_physical_dataset_mode(scene_cfg):
    """Allow only the explicitly aligned, real-physics rollout path.

    The old unconditional guard is intentionally kept for retired runners.
    The new T03 dataset runner must opt into physical_rollout and must never
    use scripted grasp attachment.
    """
    if not bool(scene_cfg.get("physical_rollout", False)):
        reject_legacy_execution()
    if bool(scene_cfg.get("scripted_grasp_attachment", False)):
        raise RuntimeError("INVALID_SCRIPTED_ATTACHMENT: physical dataset must use real physics")
