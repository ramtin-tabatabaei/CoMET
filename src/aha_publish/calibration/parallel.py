"""Isolated simulator workers, episode merging, and concise collection progress."""
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

from aha_publish import paths

EVENT_PREFIX = '@@CALIBRATION '


def default_workers():
    """Leave CPU/RAM headroom for the desktop and bound simulator concurrency."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    workers = min(8, max(1, cpus // 2))
    try:
        available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        # Reserve 2 GiB for other applications, then budget 2 GiB per simulator.
        workers = min(workers, max(1, (available - 2 * 1024**2) // (2 * 1024**2)))
    except (OSError, StopIteration, ValueError):
        pass
    return workers


def elapsed_text(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:d}:{minutes:02d}:{seconds:02d}' if hours else f'{minutes:02d}:{seconds:02d}'


def emit_event(event, **details):
    # Start a fresh line even if the simulator last printed without a newline.
    print('\n' + EVENT_PREFIX + json.dumps(dict(event=event, **details)), flush=True)


@dataclass(frozen=True)
class CollectionJob:
    task: str
    shard: int
    episodes: int
    root: Path
    log: Path

    @property
    def key(self):
        return (self.task, self.shard)


def collection_jobs(tasks, episodes, workers, staging, logs):
    """Split even a single task across simulators, reusing each for a batch."""
    shards = min(episodes, math.ceil(workers / len(tasks)))
    base, extra = divmod(episodes, shards)
    return [CollectionJob(task, shard, base + (shard < extra),
                          Path(staging) / task / str(shard),
                          Path(logs) / f'{task}.worker{shard + 1}.log')
            for shard in range(shards) for task in tasks]


class CollectionProgress:
    def __init__(self, tasks, episodes, workers, logs, skipped=0, stream=None):
        self.stream = stream or sys.stdout
        self.tty = self.stream.isatty()
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.last_print = self.started
        self.total = len(tasks) * episodes
        self.episodes = episodes
        self.counts = {}
        self.width = 0
        self.status = 'Starting simulators'
        print('\nCalibration | collect clean episodes', file=self.stream)
        print(f'  Tasks: {len(tasks)}   Episodes/task: {episodes}   Parallel simulators: {workers}', file=self.stream)
        if skipped:
            print(f'  Reused: {skipped} complete task baseline(s)', file=self.stream)
        print(f'  Simulator logs: {logs}\n', file=self.stream, flush=True)
        self.refresh(force=True)

    def _clear(self):
        if self.tty and self.width:
            self.stream.write('\r' + ' ' * self.width + '\r')
            self.width = 0

    def note(self, message):
        with self.lock:
            self._clear()
            print(f'  {message}', file=self.stream, flush=True)

    def update(self, job, event):
        with self.lock:
            kind = event.get('event')
            if kind == 'episode_done':
                self.counts[job.key] = int(event['completed'])
                count = sum(value for (task, _), value in self.counts.items() if task == job.task)
                self.status = f'{job.task}: {count}/{self.episodes} clean'
            elif kind == 'retry':
                self.status = f'{job.task}: retrying an unsuccessful demo'
            elif kind == 'episode_started':
                self.status = f'{job.task}: collecting clean episodes'
            else:
                return
            self._render(force=kind == 'episode_done')

    def _render(self, force=False):
        now = time.monotonic()
        if not force and not self.tty and now - self.last_print < 30:
            return
        done = sum(self.counts.values())
        filled = int(20 * done / self.total) if self.total else 20
        bar = '#' * filled + '-' * (20 - filled)
        line = f'  [{bar}] {done}/{self.total}  {elapsed_text(now - self.started)}  {self.status}'
        if self.tty:
            width = max(20, shutil.get_terminal_size((100, 24)).columns - 1)
            line = line[:width]
            self.stream.write('\r' + line.ljust(self.width))
            self.stream.flush()
            self.width = len(line)
        else:
            print(line, file=self.stream, flush=True)
        self.last_print = now

    def refresh(self, force=False):
        with self.lock:
            self._render(force)

    def finish(self):
        with self.lock:
            self._clear()


class SimulatorWorkers:
    def __init__(self):
        self.processes = set()
        self.lock = threading.Lock()
        self.cancelled = False

    def run(self, job, args, progress):
        job.root.mkdir(parents=True, exist_ok=True)
        job.log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', '-m', 'aha_publish.calibration.calibrate_all_clean',
                   '--worker', '--task', job.task, '--episodes', str(job.episodes),
                   '--max-attempts', str(args.max_attempts), '--out-root', str(job.root)]
        if args.show_sim:
            command.append('--show-sim')
        env = os.environ.copy()
        for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[name] = '1'
        with job.log.open('w') as log:
            with self.lock:
                if self.cancelled:
                    return -signal.SIGTERM
                process = subprocess.Popen(command, cwd=paths.PROJECT_ROOT, env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, errors='replace', start_new_session=True)
                self.processes.add(process)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith(EVENT_PREFIX):
                        try:
                            event = json.loads(line[len(EVENT_PREFIX):])
                        except json.JSONDecodeError:
                            continue
                        progress.update(job, event)
                return process.wait()
            finally:
                process.stdout.close()
                if process.poll() is None:
                    self._signal(process, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._signal(process, signal.SIGKILL)
                        process.wait()
                with self.lock:
                    self.processes.discard(process)

    @staticmethod
    def _signal(process, sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    def cancel(self):
        with self.lock:
            self.cancelled = True
            processes = list(self.processes)
        for process in processes:
            self._signal(process, signal.SIGTERM)
        deadline = time.monotonic() + 5
        for process in processes:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                self._signal(process, signal.SIGKILL)
        # Descendants can still hold the log pipe open after their worker exits.
        for process in processes:
            self._signal(process, signal.SIGKILL)


def merge_task_episodes(task, jobs, staging, raw_dir, parameters):
    """Validate a complete staged baseline before replacing the previous one."""
    from .artifacts import validate_raw
    from . import residuals

    merged_root = Path(staging) / 'merged'
    destination = merged_root / task
    destination.mkdir(parents=True)
    episode = 0
    for job in sorted(jobs, key=lambda item: item.shard):
        files = validate_raw(job.root / 'clean_data', task, minimum=job.episodes)
        residuals.load_envelopes(job.root / 'clean_data', task, parameters)
        if len(files) != job.episodes:
            raise ValueError(f'{task}: worker produced {len(files)} episodes; expected {job.episodes}')
        for source in sorted(files, key=lambda path: int(path.stem[2:])):
            target = destination / f'ep{episode}.csv'
            with source.open(newline='') as incoming, target.open('w', newline='') as outgoing:
                reader = csv.DictReader(incoming)
                writer = csv.DictWriter(outgoing, fieldnames=reader.fieldnames)
                writer.writeheader()
                for row in reader:
                    row['episode'] = episode
                    writer.writerow(row)
            sidecar = json.loads(source.with_suffix('.residual.json').read_text())
            sidecar['episode'] = episode
            target.with_suffix('.residual.json').write_text(json.dumps(sidecar, indent=2) + '\n')
            episode += 1
    validate_raw(merged_root, task, minimum=sum(job.episodes for job in jobs))
    residuals.load_envelopes(merged_root, task, parameters)
    final = Path(raw_dir) / task
    final.parent.mkdir(parents=True, exist_ok=True)
    backup = Path(staging) / f'{task}.previous'
    if final.exists():
        final.rename(backup)
    try:
        destination.rename(final)
    except BaseException:
        if backup.exists():
            backup.rename(final)
        raise
    return episode
