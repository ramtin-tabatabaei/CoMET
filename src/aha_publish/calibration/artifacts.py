"""Complete the offline grip baseline and save reproducible per-task manifests."""
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import numpy as np
from aha_publish import paths
from .compute_thresholds import load_episodes, write_json
from .settings import detector_environment


def validate_raw(raw_dir, task, minimum=1):
    files = sorted((Path(raw_dir) / task).glob('ep*.csv'))
    if len(files) < minimum:
        raise ValueError(f'{task}: need {minimum} clean episode CSVs in {raw_dir}; found {len(files)}')
    required = {'waypoint', 'local_step', 'path_done', 'distance_m', 'angle_rad', 'grip_force', 'grip_force_drop'}
    for filename in files:
        if not filename.stem[2:].isdigit():
            raise ValueError(f'Expected ep<number>.csv: {filename}')
        with filename.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if not required <= set(reader.fieldnames or []):
                raise ValueError(f'{filename}: missing columns {sorted(required - set(reader.fieldnames or []))}')
            rows = list(reader)
        if not rows:
            raise ValueError(f'{filename}: empty clean episode')
        for index, row in enumerate(rows, 2):
            if None in row or any(row.get(column) is None for column in required):
                raise ValueError(f'{filename}:{index}: malformed clean-data row')
        for column in ('distance_m', 'angle_rad', 'grip_force', 'grip_force_drop'):
            found = False
            for row in rows:
                value = row[column]
                if value == '':
                    continue
                try:
                    number = float(value)
                except ValueError:
                    raise ValueError(f'{filename}: invalid {column}: {value!r}') from None
                if not math.isfinite(number):
                    raise ValueError(f'{filename}: non-finite {column}')
                found = True
            if not found:
                raise ValueError(f'{filename}: no measurements for {column}')
        for column in ('distance_m', 'angle_rad'):
            if not any(row['waypoint'] != '' and row['path_done'] == '0'
                       and row[column] != '' for row in rows):
                raise ValueError(f'{filename}: no in-motion measurements for {column}')
    return files


def grip_stats(task, episodes, min_force):
    metrics = {}
    for metric in ('grip_force', 'grip_force_drop'):
        per_episode = []
        for i, rows in enumerate(episodes):
            values = [float(r[metric]) for r in rows if r.get(metric) not in (None, '') and float(r.get('grip_force') or 0) > min_force]
            if not values:
                continue
            values = np.asarray(values, dtype=float)
            per_episode.append({'episode': i, 'stats': {'count': int(values.size), 'mean': float(values.mean()), 'std': float(values.std()), 'min': float(values.min()), 'max': float(values.max())}})
        averages = {key: float(np.mean([e['stats'][key] for e in per_episode])) for key in ('count', 'mean', 'std', 'min', 'max')} if per_episode else {}
        metrics[metric] = {'average_of_episode_stats': averages, 'episodes': per_episode}
    return {'task': task, 'episodes_completed': len(episodes), 'episodes_requested': len(episodes), 'metrics': metrics,
            'filter': {'min_grip_force_for_stats': min_force}}


def finalize(task, raw_dir, config):
    episodes = load_episodes(task, Path(raw_dir))
    grip_file = paths.CALIBRATION_DIR / 'grip_force_stats' / f'{task}_success_grip_force_stats.json'
    write_json(grip_file, grip_stats(task, episodes, config['slip']['min_force']))
    from .residuals import compute
    files = [f'transition_arrival_stats/{task}.json', f'orientation_arrival_stats/{task}.json',
             f'grip_force_stats/{grip_file.name}', compute(task, raw_dir, config['collision'])]
    for name in files:
        if not (paths.CALIBRATION_DIR / name).is_file():
            raise ValueError(f'Calibration did not produce {name}; check the clean telemetry.')
    write_json(paths.CALIBRATION_DIR / 'manifests' / f'{task}.json', {
        'schema_version': 1, 'task': task, 'created_at': datetime.now(timezone.utc).isoformat(),
        'episodes': len(episodes), 'parameters': config,
        'runtime_environment': detector_environment(config), 'artifacts': files,
        'notes': ['No held-force samples means the slip detector uses its existing fallback.'],
    })
