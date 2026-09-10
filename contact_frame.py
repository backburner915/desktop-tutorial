"""Contact-frame construction from measured finger collision surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9 or not np.isfinite(norm):
        raise ValueError("contact-frame vector is degenerate")
    return value / norm


def build_contact_frames(geometry_report: dict, static_report: dict, *, pregrasp_distance_m: float = 0.08) -> dict:
    """Create two independent station frames without a link6-local offset.

    The long object axis separates the two stations.  Each station retains an
    independently measured hand closing axis and inward pad plane.  The
    returned frames are nominal planning data only; ``validated`` remains
    false until a physical contact trial confirms both pad surfaces.
    """
    dimensions = np.asarray(geometry_report["collision_bbox"]["dimensions_m"], dtype=np.float64)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all():
        raise ValueError("candidate dimensions are unavailable")
    long_axis_index = int(np.argmax(dimensions))
    long_axis = np.eye(3, dtype=np.float64)[long_axis_index]
    object_center = np.asarray(static_report["target"]["bbox_min_m"], dtype=np.float64)
    object_center += np.asarray(static_report["target"]["bbox_max_m"], dtype=np.float64)
    object_center /= 2.0
    station_separation = float(dimensions[long_axis_index] * 0.50)
    if station_separation <= 0.0:
        raise ValueError("object does not provide two stations along its long axis")
    stations = {}
    side_names = ("left", "right")
    for index, side in enumerate(side_names):
        g = geometry_report["grippers"][side]
        closing_axis = _unit(np.asarray(g["closing_axis"], dtype=np.float64))
        station_sign = 1.0 if index == 0 else -1.0
        station_center = object_center + station_sign * long_axis * station_separation / 2.0
        # The hand approaches its own station from outside the object along its
        # measured closing axis.  The sign is chosen independently per hand;
        # no shared local position or link6 offset is assumed.
        approach_normal = station_sign * closing_axis
        grasp_position = station_center - approach_normal * (
            float(dimensions[np.argsort(dimensions)[0]]) / 2.0 + 0.002
        )
        pregrasp_position = grasp_position - approach_normal * float(pregrasp_distance_m)
        stations[side] = {
            "station_center_m": station_center.tolist(),
            "grasp_position_m": grasp_position.tolist(),
            "pregrasp_position_m": pregrasp_position.tolist(),
            "closing_axis_world": closing_axis.tolist(),
            "approach_normal_world": approach_normal.tolist(),
            "inner_pad_plane": g["inner_planes"],
            "effective_opening_m": g["effective_opening_m"],
            "contact_frame_source": g["method"],
        }
    station_distance = float(np.linalg.norm(np.asarray(stations["left"]["station_center_m"]) - np.asarray(stations["right"]["station_center_m"])))
    return {
        "frame_version": "r1-contact-frame-v1",
        "object_center_m": object_center.tolist(),
        "object_dimensions_m": dimensions.tolist(),
        "object_long_axis_index": long_axis_index,
        "object_long_axis_world": long_axis.tolist(),
        "station_distance_m": station_distance,
        "stations": stations,
        "validated": False,
        "validation_required": "dynamic independent pad contact trial; geometry alone does not certify grasp",
    }


def write_contact_frame(geometry_path: str | Path, static_path: str | Path, output_path: str | Path) -> dict:
    geometry = json.loads(Path(geometry_path).read_text(encoding="utf-8"))
    static = json.loads(Path(static_path).read_text(encoding="utf-8"))
    report = build_contact_frames(geometry, static)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
