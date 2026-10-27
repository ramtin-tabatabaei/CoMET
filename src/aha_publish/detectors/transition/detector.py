"""Position measurements shared by clean calibration and live detection."""

import numpy as np


def _safe_array(val, length):
    if val is None:
        return np.zeros(length)
    arr = np.asarray(val, dtype=float).flatten()
    out = np.zeros(length)
    out[:min(len(arr), length)] = arr[:length]
    return out


def obs_to_row(
    obs,
    step,
    waypoint=None,
    waypoint_pose=None,
    ttm_waypoint_pose=None,
):
    gripper_pose = _safe_array(getattr(obs, 'gripper_pose', None), 7)
    gripper_xyz = gripper_pose[:3].astype(float)

    live_waypoint_distance = np.nan
    if waypoint_pose is not None:
        waypoint_arr = np.asarray(waypoint_pose, dtype=float).flatten()
        if len(waypoint_arr) >= 3:
            live_waypoint_distance = float(
                np.linalg.norm(gripper_xyz - waypoint_arr[:3])
            )

    ttm_waypoint_distance = np.nan
    if ttm_waypoint_pose is not None:
        ttm_arr = np.asarray(ttm_waypoint_pose, dtype=float).flatten()
        if len(ttm_arr) >= 3:
            ttm_waypoint_distance = float(
                np.linalg.norm(gripper_xyz - ttm_arr[:3])
            )

    waypoint_distance = (
        ttm_waypoint_distance
        if np.isfinite(ttm_waypoint_distance)
        else live_waypoint_distance
    )
    waypoint_distance_source = (
        'ttm_recalculated'
        if np.isfinite(ttm_waypoint_distance)
        else 'expected_live_waypoint'
    )
    return {
        'step': step,
        'waypoint': waypoint,
        'waypoint_distance': waypoint_distance,
        'waypoint_distance_source': waypoint_distance_source,
        'live_waypoint_distance': live_waypoint_distance,
        '_gripper_xyz': gripper_xyz.copy(),
    }


def update_deltas(logs):
    for i in range(1, len(logs)):
        curr = logs[i]
        prev = logs[i - 1]
        curr['gripper_motion'] = float(
            np.linalg.norm(curr['_gripper_xyz'] - prev['_gripper_xyz'])
        )
