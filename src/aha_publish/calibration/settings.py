"""Validate one tuning file and carry its parameters into live detector runs."""
import json
import math
from pathlib import Path
from aha_publish import paths

DEFAULT_SETTINGS = paths.PROJECT_ROOT / 'config/thresholds.json'

_POSE_SETTINGS = frozenset({'arrival_floor', 'arrival_scale'})
_SETTING_KEYS = {
    'transition': _POSE_SETTINGS,
    'orientation': _POSE_SETTINGS,
    'collision': frozenset({
        'momentum_version', 'momentum_gain',
        'time_step', 'residual_floor', 'residual_joint_multipliers',
        'residual_warmup_multiplier', 'residual_settle_steps', 'residual_min_joints',
    }),
    'slip': frozenset({'grip_k', 'grip_floor', 'grip_ceiling', 'min_force'}),
}


def load_settings(filename=DEFAULT_SETTINGS):
    config = json.loads(Path(filename).read_text())
    if not isinstance(config, dict) or config.keys() != _SETTING_KEYS.keys():
        raise ValueError('Threshold config must contain transition, orientation, collision, and slip.')
    for group, values in config.items():
        if not isinstance(values, dict) or values.keys() != _SETTING_KEYS[group]:
            raise ValueError(f'Unexpected or missing settings in {group}. See config/thresholds.json.')
        for key, value in values.items():
            if key == 'residual_joint_multipliers':
                if (not isinstance(value, list) or len(value) != 7
                        or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in value)):
                    raise ValueError('collision.residual_joint_multipliers must contain seven finite, positive numbers')
                continue
            if key == 'grip_ceiling' and value is None:
                continue
            elif type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{group}.{key} must be a finite, nonnegative number')
            if key in ('arrival_floor', 'arrival_scale') and value <= 0:
                raise ValueError(f'{group}.{key} must be positive')
            if key in ('momentum_gain', 'time_step', 'residual_floor', 'residual_warmup_multiplier') and value <= 0:
                raise ValueError(f'{group}.{key} must be positive')
            if key == 'residual_settle_steps' and type(value) is not int:
                raise ValueError('collision.residual_settle_steps must be a nonnegative integer')
    c = config['collision']
    if type(c['momentum_version']) is not int or c['momentum_version'] != 1:
        raise ValueError('collision.momentum_version must be 1')
    if type(c['residual_min_joints']) is not int or not 1 <= c['residual_min_joints'] <= 7:
        raise ValueError('collision.residual_min_joints must be an integer from 1 to 7')
    slip = config['slip']
    if slip['grip_ceiling'] is not None and slip['grip_ceiling'] < slip['grip_floor']:
        raise ValueError('slip.grip_ceiling must be null or >= grip_floor')
    return config


def compute_arguments(config):
    args = []
    for group, prefix in [('transition', 'trans'), ('orientation', 'ori')]:
        for key, value in config[group].items():
            args += [f'--{prefix}-{key.replace("_", "-")}', str(value)]
    slip = config['slip']
    for key, flag in [('grip_k', 'grip-k'), ('grip_floor', 'grip-floor'), ('grip_ceiling', 'grip-ceiling'), ('min_force', 'grip-min-force')]:
        args += ['--' + flag, str(slip[key]) if slip[key] is not None else 'inf']
    return args


def detector_environment(config):
    c, s = config['collision'], config['slip']
    return {
        'AHA_COLLISION_MOMENTUM_VERSION': str(c['momentum_version']),
        'AHA_COLLISION_MOMENTUM_GAIN': str(c['momentum_gain']),
        'AHA_SIM_DT': str(c['time_step']),
        'AHA_COLLISION_RESIDUAL_FLOOR': str(c['residual_floor']),
        'AHA_COLLISION_RESIDUAL_JOINT_MULT': ','.join(map(str, c['residual_joint_multipliers'])),
        'AHA_COLLISION_RESIDUAL_K': str(c['residual_warmup_multiplier']),
        'AHA_COLLISION_RESIDUAL_SETTLE': str(c['residual_settle_steps']),
        'AHA_COLLISION_RESIDUAL_MIN_JOINTS': str(c['residual_min_joints']),
        'AHA_TRANSITION_ARRIVAL_THRESHOLD': str(config['transition']['arrival_floor']),
        'AHA_ORIENTATION_ARRIVAL_THRESHOLD': str(config['orientation']['arrival_floor']),
        'AHA_SLIP_METHOD': '1',
        'AHA_SLIP_GRIP_K': str(s['grip_k']),
        'AHA_SLIP_GRIP_FLOOR': str(s['grip_floor']),
        'AHA_SLIP_GRIP_CEILING': str(s['grip_ceiling']) if s['grip_ceiling'] is not None else 'inf',
    }


def runtime_environment(env, tasks, dry_run=False):
    selected = None
    for task in tasks:
        manifest = paths.CALIBRATION_DIR / 'manifests' / f'{task}.json'
        if dry_run:
            continue
        if not manifest.exists():
            raise ValueError(f'No calibration manifest for {task}. Run calibration/main.py first.')
        data = json.loads(manifest.read_text())
        settings = data['runtime_environment']
        # Read older manifests without forwarding their retired method selector.
        if settings.pop('AHA_COLLISION_METHOD', '3') != '3':
            raise ValueError(f'{task}: calibration methods 1 and 2 were removed; recalibrate with method 3.')
        if selected is not None and selected != settings:
            raise ValueError('Selected tasks use different calibration settings; run them separately or recalibrate together.')
        selected = settings
        for artifact in data['artifacts']:
            if not (paths.CALIBRATION_DIR / artifact).is_file():
                raise ValueError(f'Missing calibration artifact: {artifact}; recompute this task.')
    env.update(selected or {})
    return env
