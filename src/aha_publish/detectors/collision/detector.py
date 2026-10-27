"""Joint-torque telemetry and the calibrated De Luca momentum observer."""

from aha_publish import paths
import json
import os

import numpy as np

# Frames observed before thresholds are locked.
# Steps before this mark are never flagged and feed the baseline.
WARMUP_STEPS = 5

DEFAULT_CALIBRATION_ROOT = os.getenv('AHA_CALIBRATION_ROOT', str(paths.CALIBRATION_DIR))
DEFAULT_CONSECUTIVE_COLLISION_FRAMES = 1

# --- Method 3: generalized-momentum observer (De Luca et al., IROS 2006) ------
# "Collision Detection and Safe Reaction with the DLR-III Lightweight
# Manipulator Arm." A model-based residual r that estimates the external
# (collision) joint torque from proprioception alone, WITHOUT joint acceleration:
#
#   p = M(q) qd                                  (generalized momentum)
#   r = K_I [ p - integral(tau + C^T qd - g + r) dt - p(0) ]
#
# so that r_dot = K_I (tau_ext - r): each r_i is a first-order low-pass estimate
# of the true external torque at joint i. A collision is declared when the
# residual magnitude clears a (per-task-calibratable) threshold. The Panda
# dynamic model M, C^T qd, g is provided self-contained in panda_dynamics.py
# (validated to machine precision against roboticstoolbox's Panda). Because the
# simulator's true dynamics differ slightly from the nominal model, the clean
# residual is a small non-zero offset; the threshold is set adaptively above the
# warmup/clean baseline so a collision must exceed it.
#
# Observer gain K_I (1/s), diagonal, applied to every joint. Higher = faster
# tracking / tighter residual but noisier.
DEFAULT_MOMENTUM_GAIN = float(os.getenv('AHA_COLLISION_MOMENTUM_GAIN', '25.0'))
# Control-step duration used to integrate the observer (shared AHA convention).
DEFAULT_MOMENTUM_DT = float(os.getenv('AHA_SIM_DT', '0.05'))


def momentum_observer_version():
    version = os.getenv('AHA_COLLISION_MOMENTUM_VERSION', '1').strip()
    if version != '1':
        raise ValueError('Only momentum observer version 1 is supported.')
    return version


# Minimum residual magnitude (Nm) that ever counts as a collision, so quiet
# waypoints with a near-zero clean residual still need a real physical torque.
DEFAULT_RESIDUAL_FLOOR = float(os.getenv('AHA_COLLISION_RESIDUAL_FLOOR', '3.0'))
# Headroom multiplier on the residual when locking the method-3 threshold from
# the WARMUP envelope: thr = max(FLOOR, RESIDUAL_K * warmup_base). This is the
# scalar fallback path only (no calibrated per-joint stats for the task). The
# calibrated per-joint path takes its headroom entirely from
# DEFAULT_RESIDUAL_JOINT_MULT -- applying K there too meant two stacked
# multipliers and a threshold that no longer matched the per-joint weights.
DEFAULT_RESIDUAL_K = float(os.getenv('AHA_COLLISION_RESIDUAL_K', '1.1'))


def _residual_joint_mult(spec):
    """Parse the per-joint threshold multiplier vector (7 comma-separated floats)."""
    try:
        vals = [float(v) for v in str(spec).split(',')]
    except ValueError:
        vals = []
    if len(vals) != 7 or any(v <= 0.0 for v in vals):
        return np.ones(7)
    return np.asarray(vals, dtype=float)


# Per-joint multiplier applied on top of the calibrated threshold vector:
#     thr_i = max(FLOOR, clean_max_i) * MULT_i
# The joints are not equally informative. Measured over the collision runs of
# basketball_in_hoop / beat_the_buzz / change_channel / close_jar, j0, j2 and j5
# cross their calibrated threshold in nearly every burst -- including bursts the
# VLM did not confirm -- so at 1.0 they carry most of the false positives, while
# the wrist joints j4/j6 only ever cross on real contacts. Replaying the recorded
# traces of those tasks (25 labelled bursts, MIN_JOINTS=1) against the previous
# vector [5, 1.1, 5, 1.1, 1.1, 2, 1.1]:
#     all 1.0 : TP=13 FP=7 FN=3   65% precision / 81% recall
#     previous: TP=13 FP=4 FN=3   76% precision / 81% recall
# i.e. three false positives removed at no cost in detections. j5 is flat over
# 1.8-2.5 (identical results); 3.0 is a cliff -- basketball_in_hoop drops from 3
# true positives to 1, because its surviving detections peak on j5 at 2.5-3.4x.
# NOTE: fitted in-sample on those four tasks; a leave-one-task-out check showed
# per-joint multipliers do not transfer well across tasks, and the 5x on j0/j2
# was derived from close_jar's false positives -- it costs basketball_in_hoop 2
# of its 5 detections. Treat as a starting point and re-check on new tasks. Set
# all-ones to disable. Only applies to the calibrated per-joint path, not the
# warmup scalar fallback.
# Joint 3 (plot r3, zero-based index 2) now uses 1.1 instead of 5.0.
# The historical metrics above do not evaluate this updated vector.
DEFAULT_RESIDUAL_JOINT_MULT = _residual_joint_mult(
    os.getenv('AHA_COLLISION_RESIDUAL_JOINT_MULT', '5.0,1.1,1.1,1.1,1.1,2.0,1.1'))
# The observer's residual has a transient while its integral catches the initial
# momentum; these first frames are marked ``_mo_settling`` and excluded from the
# warmup baseline so they cannot inflate the fallback threshold.
DEFAULT_RESIDUAL_SETTLE = int(os.getenv('AHA_COLLISION_RESIDUAL_SETTLE', '3'))
# How many joints must simultaneously exceed their per-joint residual threshold to
# declare a collision. A real end-effector collision loads the whole kinematic
# chain, so several joints cross at once; requiring >= this many rejects a lone
# noisy joint. Applies only when per-joint calibrated thresholds are available
# (the scalar warmup fallback keeps the single-condition rule).
DEFAULT_RESIDUAL_MIN_JOINTS = int(os.getenv('AHA_COLLISION_RESIDUAL_MIN_JOINTS', '1'))

from . import panda_dynamics as _panda_dyn


def _safe_array(val, length):
    if val is None:
        return np.zeros(length)
    arr = np.asarray(val, dtype=float).flatten()
    out = np.zeros(length)
    out[:min(len(arr), length)] = arr[:length]
    return out


def _safe_float(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def obs_to_row(obs, step):
    joint_forces = _safe_array(getattr(obs, 'joint_forces', None), 7)
    joint_positions = _safe_array(getattr(obs, 'joint_positions', None), 7)
    joint_velocities = _safe_array(getattr(obs, 'joint_velocities', None), 7)
    grip_forces = _safe_array(getattr(obs, 'gripper_touch_forces', None), 6)
    gripper_open = _safe_float(getattr(obs, 'gripper_open', None))
    total_f = float(
        np.linalg.norm(grip_forces[0:3]) + np.linalg.norm(grip_forces[3:6])
    )
    return {
        'step': step,
        'is_holding': total_f > 0.1 and gripper_open < 0.5,
        'torque_norm': float(np.linalg.norm(joint_forces)),
        'torque_delta': 0.0,
        'collision': False,
        'collision_reason': '',
        'collision_score': 0.0,
        # Momentum-observer (collision method 3) fields; filled by
        # update_momentum_observer once positions/velocities are available.
        'residual_norm': 0.0,
        'residual_score': 0.0,
        'residual_crossed': False,
        '_joint_forces': joint_forces.copy(),
        '_joint_positions': joint_positions.copy(),
        '_joint_velocities': joint_velocities.copy(),
    }


def update_deltas(logs):
    for i in range(1, len(logs)):
        raw_delta = float(
            np.linalg.norm(
                logs[i]['_joint_forces'] - logs[i - 1]['_joint_forces']
            )
        )
        # Saved torque diagnostics record increases in the overall magnitude.
        rising = logs[i]['torque_norm'] > logs[i - 1]['torque_norm']
        logs[i]['torque_delta'] = raw_delta if rising else 0.0

    update_momentum_observer(logs)


def update_momentum_observer(logs, gain=None, dt=None):
    """Integrate the De Luca (IROS 2006) generalized-momentum residual (method 3).

    For each row, using q, qd (joint_positions / joint_velocities) and the
    measured joint torque tau (joint_forces):

        p_k          = M(q_k) qd_k
        integrand_k  = tau_k + C(q_k, qd_k)^T qd_k - g(q_k) + r_{k-1}
        acc_k        = acc_{k-1} + integrand_k * dt
        r_k          = K_I ( p_k - p_0 - acc_k )

    r_0 = 0 by construction (acc_0 = 0). The recursion is causal and carries its
    running state (p_0, acc, r) on each row, so it advances incrementally as new
    frames arrive and survives the live detector trimming old rows from the front
    of `logs`. residual_norm = ||r||_2, residual_score = max_i |r_i| (Nm); the
    per-joint max is what the De Luca thresholds compare against.
    """
    gain = DEFAULT_MOMENTUM_GAIN if gain is None else float(gain)
    dt = DEFAULT_MOMENTUM_DT if dt is None else float(dt)
    momentum_observer_version()
    if not np.isfinite(gain) or gain <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Momentum observer gain and dt must be finite and positive')
    if not logs:
        return

    # Find the last row that already carries observer state; resume from there so
    # only new frames are integrated (and truncation of old rows is harmless).
    seed = None
    start = 0
    for i in range(len(logs) - 1, -1, -1):
        if logs[i].get('_mo_computed'):
            seed = logs[i]
            if seed.get('_mo_config', ('1', gain, dt)) != ('1', gain, dt):
                raise ValueError('Momentum observer configuration changed; start a fresh log')
            start = i + 1
            break

    if seed is None:
        # First-ever frame: p_0 anchors the observer, residual is zero.
        first = logs[0]
        q0 = first['_joint_positions']
        qd0 = first['_joint_velocities']
        p0 = _panda_dyn.mass_matrix(q0) @ qd0
        acc = np.zeros(_panda_dyn.N)
        r = np.zeros(_panda_dyn.N)
        _store_residual(first, p0, acc, r)
        first['_mo_config'] = ('1', gain, dt)
        first['_mo_settling'] = int(first.get('step', 0)) < DEFAULT_RESIDUAL_SETTLE
        start = 1
        seed = first

    p0 = seed['_mo_p0']
    acc = seed['_mo_acc']
    r = seed['mo_residual']
    for i in range(start, len(logs)):
        row = logs[i]
        q = row['_joint_positions']
        qd = row['_joint_velocities']
        tau = row['_joint_forces']
        g = _panda_dyn.gravity_torque(q)
        ct_qd = _panda_dyn.coriolis_transpose_qd(q, qd)
        p = _panda_dyn.mass_matrix(q) @ qd
        integrand = tau + ct_qd - g + r
        acc = acc + integrand * dt
        r = gain * (p - p0 - acc)
        _store_residual(row, p0, acc, r)
        row['_mo_config'] = ('1', gain, dt)
        # The absolute frame index (step) marks the observer settle window; the
        # first DEFAULT_RESIDUAL_SETTLE frames carry the startup transient.
        row['_mo_settling'] = int(row.get('step', i)) < DEFAULT_RESIDUAL_SETTLE


def _store_residual(row, p0, acc, r):
    row['_mo_p0'] = p0
    row['_mo_acc'] = acc
    row['mo_residual'] = r
    row['_mo_computed'] = True
    row['residual_norm'] = float(np.linalg.norm(r))
    row['residual_score'] = float(np.max(np.abs(r))) if len(r) else 0.0


def robust_threshold(values, minimum, scale=6.0):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return minimum
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return max(float(minimum), median + scale * 1.4826 * mad)


def default_detector_settings():
    return {'threshold_overrides': {},
            'consecutive_collision_frames': DEFAULT_CONSECUTIVE_COLLISION_FRAMES}


def freeze_thresholds(baseline, threshold_overrides=None, residual_stats=None):
    """Compute detection thresholds once from the warmup baseline.

    residual_stats: optional clean-run per-joint residual max (from
    load_residual_stats). When given, the method-3 threshold is the De Luca
    per-joint vector max(FLOOR, clean_max_i) * MULT_i; otherwise it falls
    back to a scalar taken from the (settle-trimmed) warmup residual envelope.
    """
    # Method-3 (momentum observer). Preferred: per-joint thresholds calibrated
    # from a clean run. Fallback: the warmup residual envelope, dropping the
    # observer's first-frame startup transients so they do not inflate it.
    residual_thr_vec = None
    if residual_stats is not None:
        residual_thr_vec = residual_threshold_vector(residual_stats)
        residual_thr = float(np.max(residual_thr_vec))
    else:
        settled = [r for r in baseline
                   if not r.get('_mo_settling', False)]
        residual_scores = [r.get('residual_score', 0.0)
                           for r in (settled or baseline)]
        residual_base = robust_threshold(residual_scores, minimum=0.0)
        residual_thr = max(DEFAULT_RESIDUAL_FLOOR, DEFAULT_RESIDUAL_K * residual_base)

    thr = {'residual_thr': residual_thr, 'residual_thr_vec': residual_thr_vec}
    if threshold_overrides and 'residual_thr' in threshold_overrides:
        thr['residual_thr'] = float(threshold_overrides['residual_thr'])
    # Log the per-joint vector, not just its max: the scalar alone cannot say
    # which calibration (or which multipliers) a run actually used, which makes
    # after-the-fact analysis of a trace guesswork.
    if thr.get('residual_thr_vec') is not None:
        vec_txt = ('  residual_vec=['
                   + ' '.join(f'{v:.1f}' for v in thr['residual_thr_vec'])
                   + ']  joint_mult=['
                   + ' '.join(f'{v:g}' for v in DEFAULT_RESIDUAL_JOINT_MULT)
                   + f']  min_joints={DEFAULT_RESIDUAL_MIN_JOINTS}')
    else:
        vec_txt = '  residual_vec=none (warmup scalar fallback)'
    print(
        f"\n  [thresholds locked at step {WARMUP_STEPS}]  "
        f"residual={thr.get('residual_thr', float('nan')):.2f}"
        f"{vec_txt}\n"
    )
    return thr


# --- Method-3 residual thresholds: per-task calibration from a clean run -------
# A clean episode records the per-joint residual envelope; the threshold is
#     thr_i = max(RESIDUAL_FLOOR, clean_max_i) * MULT_i
# and a collision is any joint with |r_i| > thr_i. Falls back to the warmup
# baseline when no calibration file exists.
DEFAULT_RESIDUAL_STATS_DIR = os.getenv(
    'AHA_RESIDUAL_STATS_DIR',
    os.path.join(DEFAULT_CALIBRATION_ROOT, 'residual_stats'),
)


def residual_stats_path(task_name, stats_dir=None):
    stats_dir = stats_dir or DEFAULT_RESIDUAL_STATS_DIR
    return os.path.join(stats_dir, f'{task_name}_residual_stats.json')


def load_residual_stats(task_name, stats_dir=None):
    """Return the clean per-joint residual max (np.ndarray) or None if absent."""
    path = residual_stats_path(task_name, stats_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if (
            data.get('observer_version') != momentum_observer_version()
            or data.get('gain') != DEFAULT_MOMENTUM_GAIN
            or data.get('dt') != DEFAULT_MOMENTUM_DT
        ):
            return None
        vals = data.get('per_joint_max')
        return np.asarray(vals, dtype=float) if vals else None
    except (ValueError, TypeError, OSError):
        return None


def residual_threshold_vector(clean_per_joint_max, floor=None, mult=None):
    """De Luca per-joint thresholds from the clean-run residual envelope.

        thr_i = max(FLOOR, clean_max_i) * MULT_i

    mult: per-joint multiplier applied after the floor, re-weighting the joints
    by how informative they actually are (see DEFAULT_RESIDUAL_JOINT_MULT). It is
    now the ONLY headroom over the clean envelope: the former global K factor
    stacked on top of it, so every joint carried two multipliers at once and the
    per-joint weights could not be read off the resulting threshold.
    DEFAULT_RESIDUAL_K survives only for the scalar warmup fallback below, which
    has no per-joint vector to weight.
    """
    floor = DEFAULT_RESIDUAL_FLOOR if floor is None else float(floor)
    mult = DEFAULT_RESIDUAL_JOINT_MULT if mult is None else np.asarray(mult, dtype=float)
    mx = np.asarray(clean_per_joint_max, dtype=float).ravel()
    thr = np.maximum(floor, mx)
    return thr * mult if mult.shape == thr.shape else thr


def check_collision(row, thr):
    """Compare the momentum residual with the calibrated per-joint band."""
    residual_thr = float(thr.get('residual_thr', DEFAULT_RESIDUAL_FLOOR))
    residual_score = float(row.get('residual_score', 0.0))
    residual_thr_vec = thr.get('residual_thr_vec')
    r_vec = row.get('mo_residual')
    if residual_thr_vec is not None and r_vec is not None:
        n_joints_over = int(np.sum(np.abs(r_vec) > residual_thr_vec))
        crossed = n_joints_over >= DEFAULT_RESIDUAL_MIN_JOINTS
    else:
        n_joints_over = int(residual_score > residual_thr)
        crossed = residual_score > residual_thr
    row['residual_joints_over'] = n_joints_over
    row['residual_crossed'] = bool(crossed)
    row['residual_threshold'] = residual_thr
    row['collision_score'] = residual_score / residual_thr if residual_thr > 0 else 0.0
    return bool(crossed), 'momentum_residual' if crossed else ''
