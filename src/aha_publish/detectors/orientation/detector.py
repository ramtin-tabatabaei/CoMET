"""Orientation measurements shared by clean calibration and live detection."""

import math

import numpy as np


def _safe_array(val, length):
    if val is None:
        return np.zeros(length)
    arr = np.asarray(val, dtype=float).flatten()
    out = np.zeros(length)
    out[:min(len(arr), length)] = arr[:length]
    return out


def _normalize_quat(q):
    q = np.asarray(q, dtype=float).flatten()
    if len(q) < 4:
        return np.asarray([0.0, 0.0, 0.0, 1.0])
    q = q[:4]
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0])
    return q / norm


def _quat_angle(a, b):
    if a is None or b is None:
        return np.nan
    qa = _normalize_quat(a)
    qb = _normalize_quat(b)
    dot = abs(float(np.dot(qa, qb)))
    dot = max(-1.0, min(1.0, dot))
    return float(2.0 * math.acos(dot))


def _pose_quaternion(pose):
    arr = np.asarray(pose, dtype=float).flatten()
    if len(arr) >= 7:
        return _normalize_quat(arr[3:7])
    return None


def obs_to_row(
    obs,
    step,
    waypoint=None,
    waypoint_pose=None,
    ttm_waypoint_pose=None,
):
    gripper_pose = _safe_array(getattr(obs, 'gripper_pose', None), 7)
    gripper_quat = _pose_quaternion(gripper_pose)
    live_waypoint_quat = _pose_quaternion(waypoint_pose)
    live_waypoint_angle = _quat_angle(gripper_quat, live_waypoint_quat)
    ttm_waypoint_quat = _pose_quaternion(ttm_waypoint_pose)
    ttm_waypoint_angle = _quat_angle(gripper_quat, ttm_waypoint_quat)

    return {
        'step': step,
        'waypoint': waypoint,
        'live_waypoint_angle': live_waypoint_angle,
        'ttm_waypoint_angle': ttm_waypoint_angle,
        '_gripper_quat': gripper_quat.copy(),
    }


def update_deltas(logs):
    for i in range(1, len(logs)):
        curr = logs[i]
        prev = logs[i - 1]
        curr['gripper_orientation_delta'] = _quat_angle(
            curr['_gripper_quat'], prev['_gripper_quat']
        )
