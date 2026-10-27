"""Compute arrival thresholds and slip reports from recorded clean episodes.

Transition uses the median clean arrival error; orientation uses the maximum.
Each threshold is max(arrival_floor, arrival_scale * clean_arrival_error).
De Luca collision thresholds and the grip baseline are finalized by artifacts.py.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np

from aha_publish import paths
from .settings import load_settings


def load_episodes(task, raw_dir):
    """Return clean episodes in numeric order, preserving each frame's order."""
    episodes = []
    files = sorted((Path(raw_dir) / task).glob('ep*.csv'),
                   key=lambda path: int(path.stem[2:]))
    for filename in files:
        with filename.open(newline='') as stream:
            episodes.append(list(csv.DictReader(stream)))
    return episodes


def per_waypoint_arrivals(episodes, signal_col):
    """Keep the worst last in-motion error per waypoint in each episode.

    Repeated waypoint occurrences have separate histories. A waypoint change or
    a reset of local_step starts a new occurrence.
    """
    arrivals = {}
    for rows in episodes:
        episode_values = {}
        current_wp, last_local, last_value = None, None, None

        def finish_segment():
            if last_value is not None:
                episode_values.setdefault(current_wp, []).append(last_value)

        for row in rows:
            wp = row.get('waypoint')
            value = row.get(signal_col)
            if wp in (None, '') or value in (None, '') or row.get('path_done') == '1':
                continue
            wp = int(wp)
            local_step = int(float(row.get('local_step') or 0))
            if current_wp is not None and (wp != current_wp or local_step <= last_local):
                finish_segment()
            current_wp, last_local, last_value = wp, local_step, float(value)
        finish_segment()
        for wp, values in episode_values.items():
            arrivals.setdefault(wp, []).append(max(values))
    return arrivals


def waypoint_thresholds(episodes, signal_col, parameters):
    stats = {}
    for wp, values in sorted(per_waypoint_arrivals(episodes, signal_col).items()):
        array = np.asarray(values, dtype=float)
        baseline = float(np.median(array) if parameters['arrival_stat'] == 'median'
                         else array.max())
        stats[str(wp)] = {
            'mean': round(float(array.mean()), 5),
            'median': round(float(np.median(array)), 5),
            'std': round(float(array.std()), 5),
            'max': round(float(array.max()), 5),
            'threshold': round(max(parameters['arrival_floor'],
                                   parameters['arrival_scale'] * baseline), 5),
            'n': int(array.size),
        }
    return stats


def slip_threshold(episodes, k, floor, ceiling, min_force):
    episode_stats = []
    for rows in episodes:
        values = [float(row['grip_force']) for row in rows
                  if row.get('grip_force') not in (None, '')
                  and float(row['grip_force']) > min_force]
        if values:
            episode_stats.append((float(np.mean(values)), float(np.std(values))))
    if not episode_stats:
        return None
    mean, std = np.mean(episode_stats, axis=0)
    threshold = min(ceiling, max(floor, mean - k * std))
    return {'mean': round(float(mean), 5), 'std': round(float(std), 5),
            'threshold': round(float(threshold), 5), 'n_episodes': len(episode_stats)}


def pose_stats_json(task, signal_name, arrival_stats, parameters, n_episodes):
    return {
        'task': task,
        'signal': signal_name,
        'n_clean_episodes': n_episodes,
        'arrival_stats_by_waypoint': arrival_stats,
        'arrival_floor': parameters['arrival_floor'],
        'arrival_scale': parameters['arrival_scale'],
        'source': 'compute_thresholds.py (offline, from calibration clean_data)',
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def build_params(args):
    return tuple({
        'arrival_floor': getattr(args, prefix + '_arrival_floor'),
        'arrival_scale': getattr(args, prefix + '_arrival_scale'),
        'arrival_stat': statistic,
    } for prefix, statistic in [('trans', 'median'), ('ori', 'max')])


def main():
    defaults = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root', type=Path, default=paths.CALIBRATION_DIR)
    parser.add_argument('--raw-dir', type=Path,
                        help='Defaults to <out-root>/clean_data.')
    parser.add_argument('--task', action='append', default=[])
    for group, prefix in [('transition', 'trans'), ('orientation', 'ori')]:
        for key, value in defaults[group].items():
            parser.add_argument(f'--{prefix}-{key.replace("_", "-")}', type=float, default=value)
    for key, flag in [('grip_k', 'grip-k'), ('grip_floor', 'grip-floor'),
                      ('grip_ceiling', 'grip-ceiling'), ('min_force', 'grip-min-force')]:
        value = defaults['slip'][key]
        parser.add_argument('--' + flag, type=float,
                            default=float('inf') if value is None else value)
    args = parser.parse_args()
    output = args.out_root.expanduser().resolve()
    raw_dir = args.raw_dir.expanduser().resolve() if args.raw_dir else output / 'clean_data'
    tasks = args.task or sorted(folder.name for folder in raw_dir.glob('*')
                               if folder.is_dir() and any(folder.glob('ep*.csv')))
    if not tasks:
        parser.error(f'No clean episodes under {raw_dir}; collect telemetry first.')

    # Validate every selected task before replacing any baseline files.
    from .artifacts import validate_raw
    for task in tasks:
        validate_raw(raw_dir, task)

    trans, orientation = build_params(args)
    reports = {'transition': [], 'orientation': [], 'slip': []}
    print(f'compute_thresholds: {len(tasks)} task(s) from {raw_dir} -> {output}')
    for task in tasks:
        episodes = load_episodes(task, raw_dir)
        for name, column, signal, parameters in (
            ('transition', 'distance_m', 'distance (m)', trans),
            ('orientation', 'angle_rad', 'angle (rad)', orientation),
        ):
            stats = waypoint_thresholds(episodes, column, parameters)
            write_json(output / f'{name}_arrival_stats' / f'{task}.json',
                       pose_stats_json(task, signal, stats, parameters, len(episodes)))
            for wp, values in stats.items():
                reports[name].append({
                    'task': task, 'waypoint': wp,
                    'arrival_threshold': values['threshold'],
                    'arrival_median_clean': values['median'],
                    'arrival_max_clean': values['max'],
                    'n_episodes': len(episodes),
                })
        slip = slip_threshold(episodes, args.grip_k, args.grip_floor,
                              args.grip_ceiling, args.grip_min_force)
        if slip:
            reports['slip'].append({'task': task, 'metric': 'grip_force',
                                    'k': args.grip_k, **slip})
        print(f'  {task}: {len(episodes)} episodes')

    report_dir = output / 'threshold_report'
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in reports.items():
        filename = report_dir / f'{name}.csv'
        if not rows:
            filename.unlink(missing_ok=True)
            continue
        with filename.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        thresholds = [row['arrival_threshold' if name != 'slip' else 'threshold']
                      for row in rows]
        print(f'  {name}: threshold min/median/max='
              f'{min(thresholds):.4f}/{statistics.median(thresholds):.4f}/{max(thresholds):.4f}')
    print(f'Reports saved in {report_dir}')


if __name__ == '__main__':
    main()
