"""Collect successful clean episodes for the shared calibration workflow.

Each episode saves unified pose/grip telemetry and De Luca residual maxima.
Thresholds and runtime manifests are produced by the compute action.
"""

from src import paths
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np

from src.detectors.collision import detector as CD
from src.detectors.orientation import detector as OD
from src.detectors.transition import detector as TD
from src.detectors.slip import detector as SD
from . import residuals
from .artifacts import validate_raw
from .compute_thresholds import write_json
from .parallel import (CollectionProgress, SimulatorWorkers, collection_jobs,
                       default_workers, elapsed_text, emit_event, merge_task_episodes)

FAILGEN_ROOT = paths.FAILGEN_ROOT
CONFIGS_PATH = paths.CONFIGS_DIR
COPPELIA = paths.COPPELIASIM_ROOT
DEFAULT_CALIBRATION_ROOT = paths.CALIBRATION_DIR
CALIBRATION_ROOT = DEFAULT_CALIBRATION_ROOT
RAW_DIR = CALIBRATION_ROOT / 'clean_data'


def configure_output_root(out_root):
    global CALIBRATION_ROOT, RAW_DIR
    CALIBRATION_ROOT = Path(out_root).expanduser().resolve()
    RAW_DIR = CALIBRATION_ROOT / 'clean_data'


def configure_env(headless=True):
    os.environ["COPPELIASIM_ROOT"] = str(COPPELIA)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(COPPELIA)
    if headless:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    parts = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    if str(COPPELIA) not in parts:
        parts.append(str(COPPELIA))
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
        if os.environ.get("AHA_ALL_CLEAN_ENV_READY") != "1":
            os.environ["AHA_ALL_CLEAN_ENV_READY"] = "1"
            os.execvpe(sys.executable, [sys.executable, '-m',
                        'src.calibration.calibrate_all_clean', *sys.argv[1:]], os.environ)


def available_tasks():
    return sorted(p.stem for p in CONFIGS_PATH.glob("*.yaml"))


def waypoint_index_from_point(point):
    try:
        name = point._waypoint.get_name()
        if name.startswith("waypoint"):
            return int(name[len("waypoint"):])
    except Exception:
        return None
    return None


def waypoint_pose(index):
    if index is None:
        return None
    try:
        from pyrep.objects.object import Object

        obj = Object.get_object(f"waypoint{int(index)}")
        # Cartesian-path waypoints: the object's pose is the path frame origin
        # (~mid-path), but the arm drives to the path END. Resolve to the last
        # sample (relative_distance=1.0), converting Euler->quat via a scratch
        # dummy so the result matches get_pose()'s [x,y,z,qx,qy,qz,qw] layout.
        try:
            from pyrep.const import ObjectType
            from pyrep.objects.cartesian_path import CartesianPath

            if obj.get_type() == ObjectType.PATH:
                position, euler = CartesianPath(
                    obj.get_handle()).get_pose_on_path(1.0)
                from pyrep.objects.dummy import Dummy

                scratch = Dummy.create()
                try:
                    scratch.set_position(list(position))
                    scratch.set_orientation(list(euler))
                    return list(scratch.get_pose())
                finally:
                    scratch.remove()
        except Exception:
            pass
        return obj.get_pose()
    except Exception:
        return None




RAW_HEADER = [
    'task', 'episode', 'waypoint', 'local_step', 'global_step', 'path_done',
    'distance_m', 'angle_rad', 'grip_force', 'grip_force_drop',
]


def write_raw_csv(path, task, episode, pose_rows, grip_logs):
    """Align pose/grip telemetry by frame, leaving missing signals blank."""
    frames = {}
    for row in grip_logs:
        frames[int(row['step'])] = {
            'grip_force': row.get('grip_force', ''),
            'grip_force_drop': row.get('grip_force_drop', ''),
        }
    for row in pose_rows:
        frames.setdefault(row['global_step'], {}).update(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_HEADER)
        writer.writeheader()
        for step, row in sorted(frames.items()):
            writer.writerow(dict(row, task=task, episode=episode, global_step=step))


def run_task(task, episodes, max_attempts, show_sim=False):
    if str(FAILGEN_ROOT) not in sys.path:
        sys.path.insert(0, str(FAILGEN_ROOT))
    from failgen.env_wrapper import FailGenEnvWrapper

    configure_env(headless=not show_sim)
    # Collection replaces baseline files; an old manifest must not authorize a
    # partially refreshed task if collection fails or compute has not run yet.
    (CALIBRATION_ROOT / 'manifests' / f'{task}.json').unlink(missing_ok=True)
    # Remove this task's previous raw dumps so a re-run never mixes episodes.
    raw_task_dir = RAW_DIR / task
    if raw_task_dir.exists():
        for old in raw_task_dir.glob("ep*.csv"):
            old.unlink()
        for old in raw_task_dir.glob("ep*.residual.json"):
            old.unlink()

    env_wrapper = FailGenEnvWrapper(
        task_name=task,
        headless=not show_sim,
        record=False,
        save_data=False,
        no_failures=True,
        save_path=str(CALIBRATION_ROOT / 'backend_data'),
        save_keyframes_only=True,
        max_failure_attempts=max_attempts,
    )

    completed = 0
    target = max(1, int(episodes))
    attempts = 0
    # Some demos fail to produce a successful trajectory; keep retrying until the
    # task reaches `target` successful clean episodes (capped so a task that can
    # never succeed cannot loop forever).
    attempt_cap = target * max(3, int(max_attempts))

    try:
        while completed < target and attempts < attempt_cap:
            attempts += 1
            episode = completed
            state = {
                "waypoint": None,
                "original_pose": None,
                "current_pose": None,
                "next_waypoint": 0,
                "global_step": 0,
                "local_step": 0,
            }
            pose_rows = []
            grip_logs = []
            torque_logs = []

            original_step = env_wrapper.on_env_step
            original_waypoint = env_wrapper.on_env_waypoint
            original_waypoint_end = env_wrapper.on_env_waypoint_end

            def reset_waypoint(idx):
                state["waypoint"] = idx
                state["local_step"] = 0

            def feed(obs, path_done=False):
                idx = state["waypoint"]
                if idx is None:
                    return
                global_step = state["global_step"]
                local_step = state["local_step"]
                state["global_step"] += 1
                state["local_step"] += 1

                current_pose = state["current_pose"]
                if current_pose is None:
                    current_pose = waypoint_pose(idx)
                original_pose = state["original_pose"]

                o_row = OD.obs_to_row(
                    obs, global_step, waypoint=idx, waypoint_pose=current_pose,
                    ttm_waypoint_pose=original_pose)
                # Match the live arrival check against the original waypoint pose.
                angle = float(o_row.get("ttm_waypoint_angle", float("nan")))
                if not np.isfinite(angle):
                    angle = float(o_row.get("live_waypoint_angle", float("nan")))
                t_row = TD.obs_to_row(
                    obs, global_step, waypoint=idx, waypoint_pose=current_pose,
                    ttm_waypoint_pose=original_pose)
                dist = float(t_row.get("live_waypoint_distance", float("nan")))
                pose_rows.append({
                    'waypoint': idx, 'local_step': local_step,
                    'global_step': global_step, 'path_done': int(bool(path_done)),
                    'distance_m': dist if np.isfinite(dist) else '',
                    'angle_rad': angle if np.isfinite(angle) else '',
                })

            def patched_waypoint(point):
                idx = waypoint_index_from_point(point)
                if idx is None:
                    idx = state["next_waypoint"]
                state["next_waypoint"] = max(state["next_waypoint"], int(idx) + 1)
                reset_waypoint(idx)
                state["original_pose"] = waypoint_pose(idx)
                original_waypoint(point)
                state["current_pose"] = waypoint_pose(idx)

            def patched_step(obs):
                original_step(obs)
                step = state["global_step"]

                s_row = SD.obs_to_row(obs, step)
                s_row["task"] = task
                s_row["episode"] = episode
                grip_logs.append(s_row)
                # Calibration saves force/drop only; computing these needs just
                # the preceding frame, not another scan of the whole episode.
                SD.update_deltas(grip_logs[-2:])

                c_row = CD.obs_to_row(obs, step)
                c_row["task"] = task
                c_row["episode"] = episode
                torque_logs.append(c_row)
                CD.update_deltas(torque_logs)

                feed(obs, path_done=False)

            def patched_waypoint_end(point):
                idx = waypoint_index_from_point(point)
                if state["waypoint"] is None:
                    reset_waypoint(idx)
                original_waypoint_end(point)
                try:
                    obs = env_wrapper._task_env.get_observation()
                    feed(obs, path_done=True)
                except Exception:
                    pass
                state["waypoint"] = None
                state["original_pose"] = None
                state["current_pose"] = None

            env_wrapper.on_env_waypoint = patched_waypoint
            env_wrapper.on_env_step = patched_step
            env_wrapper.on_env_waypoint_end = patched_waypoint_end

            emit_event('episode_started', task=task, episode=episode)
            try:
                env_wrapper.reset()
                # Wire the waypoint-START hook so per-frame in-motion callbacks
                # know which waypoint is active; without it feed() drops every
                # in-motion frame (state["waypoint"] stays None) and only the
                # path_done arrival frame survives -> empty per-waypoint baselines.
                demo = env_wrapper.get_success(
                    callable_each_waypoint=patched_waypoint)
            finally:
                env_wrapper.on_env_step = original_step
                env_wrapper.on_env_waypoint = original_waypoint
                env_wrapper.on_env_waypoint_end = original_waypoint_end

            if demo is None:
                emit_event('retry', task=task, attempt=attempts, completed=completed)
                continue

            residuals.write_episode(RAW_DIR / task / f"ep{episode}.csv",
                                    task, episode, torque_logs, CD)
            write_raw_csv(RAW_DIR / task / f"ep{episode}.csv",
                          task, episode, pose_rows, grip_logs)
            completed += 1
            emit_event('episode_done', task=task, completed=completed,
                       attempt=attempts, frames=len(torque_logs))
    finally:
        env_wrapper.shutdown()

    if completed < target:
        raise RuntimeError(f"{task}: only {completed}/{target} clean episodes "
                           f"after {attempts} attempts (cap {attempt_cap})")

    validate_raw(RAW_DIR, task, minimum=target)
    residuals.load_envelopes(RAW_DIR, task, residuals.observer_parameters(CD))
    return {'task': task, 'episodes_completed': completed}


def clean_episode_count(task):
    """Number of clean raw-dump episodes already on disk for a task
    (<out-root>/clean_data/<task>/ep*.csv)."""
    d = RAW_DIR / task
    if not d.exists():
        return 0
    return sum(1 for p in d.glob("ep*.csv") if p.stem[2:].isdigit())


def task_done(task, episodes):
    """Collection is reusable when raw data and residual sidecars are complete."""
    try:
        validate_raw(RAW_DIR, task, minimum=episodes)
        _, count = residuals.load_envelopes(RAW_DIR, task, residuals.observer_parameters(CD))
        return count >= episodes
    except (ValueError, OSError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", default=[],
                        help="Task to calibrate. Omit for all configured tasks.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--workers", type=int, default=default_workers(),
                        help="Parallel simulators, including episodes of the same task. Default: auto, up to 8 based on CPU/RAM. Use 1 for sequential collection.")
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument(
        "--out-root",
        default=str(DEFAULT_CALIBRATION_ROOT),
        help=(
            f"Root folder for all calibration outputs. Default: {DEFAULT_CALIBRATION_ROOT}"
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--show-sim", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.episodes < 1 or args.workers < 1 or args.max_attempts < 1:
        parser.error('--episodes, --workers, and --max-attempts must be positive')

    configure_output_root(args.out_root)
    configure_env(headless=not args.show_sim)
    tasks = list(dict.fromkeys(args.task or available_tasks()))
    if not tasks:
        parser.error('No tasks found; supply --task and check the configured paths.')
    if args.worker:
        if len(tasks) != 1:
            parser.error('An internal worker requires exactly one task')
        run_task(tasks[0], args.episodes, args.max_attempts, args.show_sim)
        return 0

    if args.force:
        skipped = []
        tasks_to_run = tasks
    else:
        # Reuse complete clean telemetry unless --force requests fresh episodes.
        counts = {task: clean_episode_count(task) for task in tasks}
        skipped = [task for task in tasks
                   if counts[task] >= args.episodes and task_done(task, args.episodes)]
        tasks_to_run = [task for task in tasks if task not in set(skipped)]
    if not tasks_to_run:
        print(f'\nCalibration | reused {len(skipped)} complete task baseline(s); no collection needed.\n', flush=True)
        return 0

    CALIBRATION_ROOT.mkdir(parents=True, exist_ok=True)
    logs = CALIBRATION_ROOT / 'logs' / f'collect-{uuid4().hex[:8]}'
    logs.mkdir(parents=True)
    for task in tasks_to_run:
        (CALIBRATION_ROOT / 'manifests' / f'{task}.json').unlink(missing_ok=True)
    failed = []
    workers = SimulatorWorkers()
    start = time.monotonic()
    with TemporaryDirectory(prefix='.collect-', dir=CALIBRATION_ROOT) as staging:
        jobs = collection_jobs(tasks_to_run, args.episodes, args.workers, staging, logs)
        concurrency = min(args.workers, len(jobs))
        progress = CollectionProgress(tasks_to_run, args.episodes, concurrency, logs, len(skipped))
        task_jobs = {task: [job for job in jobs if job.task == task] for task in tasks_to_run}
        remaining = {task: len(batch) for task, batch in task_jobs.items()}
        failed_tasks = set()
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(workers.run, job, args, progress): job for job in jobs}
            try:
                while futures:
                    ready, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                    progress.refresh()
                    for future in ready:
                        job = futures.pop(future)
                        try:
                            rc = future.result()
                            if rc:
                                raise RuntimeError(f'worker exited with code {rc}; see {job.log}')
                        except Exception as exc:
                            failed_tasks.add(job.task)
                            failed.append((job.task, str(exc)))
                            progress.note(f'FAILED  {job.task}: {exc}')
                        remaining[job.task] -= 1
                        if remaining[job.task] == 0 and job.task not in failed_tasks:
                            try:
                                count = merge_task_episodes(job.task, task_jobs[job.task], staging,
                                                            RAW_DIR, residuals.observer_parameters(CD))
                                progress.note(f'SAVED   {job.task}: {count} clean episodes')
                            except Exception as exc:
                                failed.append((job.task, str(exc)))
                                progress.note(f'FAILED  {job.task}: {exc}')
            except BaseException:
                workers.cancel()
                progress.finish()
                raise
        progress.finish()

    if failed:
        write_json(
            CALIBRATION_ROOT / "all_clean_calibration_failed_tasks.json",
            [{"task": task, "error": error} for task, error in failed])
    else:
        (CALIBRATION_ROOT / "all_clean_calibration_failed_tasks.json").unlink(missing_ok=True)
    outcome = f'{len({task for task, _ in failed})} failed task(s)' if failed else 'all tasks collected'
    print(f'\n  Collection finished in {elapsed_text(time.monotonic() - start)} | {outcome}\n', flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nCalibration cancelled. Detailed simulator logs were retained.', file=sys.stderr)
        raise SystemExit(130)
