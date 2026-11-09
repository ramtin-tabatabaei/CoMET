"""Concise terminal progress, with the full episode output retained in run.log."""
from contextvars import ContextVar
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

EPISODE_CONSOLE = ContextVar('episode_console', default=None)


def duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}:{minutes:02d}:{seconds:02d}' if hours else f'{minutes:02d}:{seconds:02d}'


@dataclass
class EpisodeDisplay:
    phase: str = 'Starting simulator'
    checkpoint: str = 'Conditions'
    tree_result: str = ''
    detector_states: dict = field(default_factory=dict)
    unknown_checks: int = 0
    check_errors: int = 0


class RunConsole:
    def __init__(self, total, stream=None):
        self.stream = stream or sys.stdout
        self.tty = self.stream.isatty()
        self.lock = threading.RLock()
        self.total = total
        self.done = 0
        self.started = time.monotonic()
        self.last_print = self.started
        self.width = 0
        self.active = {}
        self.processes = set()
        self.cancelled = False

    def _clear(self):
        if self.width:
            self.stream.write('\r' + ' ' * self.width + '\r')
            self.width = 0

    def note(self, message):
        with self.lock:
            self._clear()
            print(message, file=self.stream, flush=True)

    def start(self, key):
        with self.lock:
            if self.cancelled:
                raise RuntimeError('Run cancelled')
            self.active[key] = EpisodeDisplay()
            self.note(f'\n  RUN  {key[0]} / {key[1]}')
            self.refresh(force=True)

    def observe(self, key, line):
        with self.lock:
            display = self.active.get(key)
            if display is None:
                return
            text = line.strip()
            prefix = f'  {key[0]}/{key[1]}'
            match = re.match(r'Waypoint (\d+) of (\d+)', text)
            if match:
                current, final = map(int, match.groups())
                display.phase = f'Waypoint {current} ({current + 1}/{final + 1})'
                self.note(f'{prefix}  |  {display.phase}')
            elif text.startswith('Description:'):
                display.phase = text.partition(':')[2].strip()
            elif match := re.search(r'\[vlm\] checking (pre|post)conditions for waypoint (\d+)(?: and postconditions for waypoint (\d+))?', text):
                kind, wp, previous = match.groups()
                display.checkpoint = (f'Pre wp{wp} + post wp{previous}' if previous is not None
                                      else f'{kind.capitalize()}conditions wp{wp}')
                display.phase = f'Checking {display.checkpoint.lower()}'
            elif match := re.search(r'VLM ([A-Z+]+(?: SIMULATOR)?) CHECK: (PASS|FAIL|UNKNOWN)', text):
                kind, verdict = match.groups()
                if 'SIMULATOR' not in kind or verdict != 'PASS':
                    self.note(f'{prefix}  |  {display.checkpoint:<26} {verdict}')
                    display.unknown_checks += verdict == 'UNKNOWN'
                display.phase = f'{display.checkpoint}: {verdict}'
            elif match := re.search(r'HOLD RESULT for waypoint (\d+):.*\[([^\]]+)\]', text):
                wp, verdict = match.groups()
                self.note(f'{prefix}  |  Hold checks wp{wp:<12} {verdict.upper()}')
            elif match := re.search(r'\[move\] (planning|executing) path for waypoint (\d+)', text):
                action, wp = match.groups()
                display.phase = f'Waypoint {wp}: {action} motion'
            elif match := re.search(r'\[move\] finished waypoint (\d+) path', text):
                display.phase = f'Waypoint {match[1]}: motion complete'
            elif match := re.search(r'\[gripper\] (opening|closing) for waypoint (\d+)', text):
                display.phase = f'Waypoint {match[2]}: {match[1]} gripper'
            elif match := re.search(r'\[DETECTOR\] (\w+) (?:failure|detected) at waypoint (\d+)', text):
                self.note(f'{prefix}  |  {match[1].capitalize()} detected at wp{match[2]}')
            elif match := re.search(r'\[vlm\] auto-confirming (\w+) detection', text):
                display.phase = f'Confirming {match[1]} detection'
            elif match := re.search(r'\[vlm:[^\]]+\]\s+(slip|collision)=(YES|NO|UNKNOWN)', text):
                name, verdict = match.groups()
                self.note(f'{prefix}  |  {name.capitalize()} confirmation: {verdict}')
            elif match := re.search(r'^BEHAVIOR TREE RESULT: (\w+)', text):
                display.tree_result = match[1]
                display.phase = 'Saving telemetry and plots'
            elif match := re.match(r'(orientation|transition|slip|collision)\s+(DISABLED|NOT EVALUATED|skipped)', text):
                display.detector_states[match[1]] = match[2].upper()
            elif text.startswith('Could not start live detectors:'):
                display.detector_states.update({name: 'NOT MONITORED' for name in
                                               ('orientation', 'transition', 'slip', 'collision')})
                self.note(f'{prefix}  |  {text}')
            elif (text.startswith('- not_satisfied:') or '[uncertain]' in text
                  or re.search(r'\[vlm\].*(?:check|confirmation) failed:', text)):
                if 'check failed:' in text:
                    display.check_errors += 1
                self.note(f'{prefix}  |  {text}')
            elif text.startswith(('WARNING:', 'ERROR:')):
                self.note(f'{prefix}  |  {text}')

    def refresh(self, force=False):
        with self.lock:
            if not self.active:
                return
            now = time.monotonic()
            if not force and not self.tty and now - self.last_print < 30:
                return
            key, display = next(reversed(self.active.items()))
            other = f' (+{len(self.active) - 1} active)' if len(self.active) > 1 else ''
            line = f'  [{self.done}/{self.total}] {duration(now - self.started)}  {key[0]}/{key[1]} | {display.phase}{other}'
            if self.tty:
                width = max(20, shutil.get_terminal_size((100, 24)).columns - 1)
                line = line[:width]
                self.stream.write('\r' + line.ljust(self.width))
                self.stream.flush()
                self.width = len(line)
            else:
                print(line, file=self.stream, flush=True)
            self.last_print = now

    def complete(self, key, rows=None, error=None):
        with self.lock:
            display = self.active.pop(key, EpisodeDisplay())
            self.done += 1
            if error is not None:
                self.note(f'\n  ERROR  [{self.done}/{self.total}] {key[0]} / {key[1]}\n    {error}')
                return
            elapsed = float(rows[0]['elapsed']) if rows else 0
            verdict = f' | tree {display.tree_result}' if display.tree_result else ''
            self.note(f'\n  DONE  [{self.done}/{self.total}] {key[0]} / {key[1]} | {duration(elapsed)}{verdict}')
            self.note(f'    {"Detector":<14} {"Result":<20} Waypoints')
            for row in rows or []:
                name = row['detector']
                raw_fire = str(row.get('fired', '')).lower() == 'true'
                confirmed_fire = str(row.get('fired_confirmed', raw_fire)).lower() == 'true'
                state = display.detector_states.get(name) or str(row.get('status', ''))
                if state.upper() in ('DISABLED', 'SKIPPED', 'NOT EVALUATED', 'NO_OUTPUT', 'NOT MONITORED'):
                    result = state.upper().replace('_', ' ')
                elif row.get('vlm_error'):
                    result = 'VLM ERROR'
                elif confirmed_fire:
                    result = 'DETECTED'
                elif raw_fire and str(row.get('vlm_retracted', '')).lower() == 'true':
                    result = 'RETRACTED'
                else:
                    result = 'NO FAILURE DETECTED'
                waypoints = str(row.get('fired_waypoints') or '-').replace(' ', ', ')
                self.note(f'    {name.capitalize():<14} {result:<20} {waypoints}')
            if rows and rows[0].get('prepost_checks') not in (None, ''):
                checks = int(rows[0]['prepost_checks'])
                failed = int(rows[0].get('prepost_fails') or 0)
                if checks:
                    self.note(f'    Condition checkpoints: {checks} checked, {failed} failed')
            if display.unknown_checks or display.check_errors:
                self.note(f'    Condition checks: {display.unknown_checks} undecided, {display.check_errors} errors')

    def cancel(self):
        with self.lock:
            self.cancelled = True
            processes = list(self.processes)
        for process in processes:
            stop_process(process)
        self.note('\nRun cancelled. Partial episode logs were retained.')


def stop_process(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def capture_episode(command, env, cwd, log_path, timeout):
    """Tail a subprocess's raw log for display without rewriting its contents."""
    context = EPISODE_CONSOLE.get()
    console, key = context if context else (None, None)
    log_path = Path(log_path)
    header = ('$ ' + shlex.join(command) + '\n\n').encode()
    with log_path.open('wb') as log:
        log.write(header)
        log.flush()
        if console is not None:
            console.lock.acquire()
        try:
            if console is not None and console.cancelled:
                raise RuntimeError('Run cancelled')
            if console is not None and key in console.active:
                # Retries replace run.log; display results must describe that
                # final attempt too, rather than retain warnings from an older one.
                console.active[key] = EpisodeDisplay()
            process = subprocess.Popen(command, env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if console is not None:
                console.processes.add(process)
        finally:
            if console is not None:
                console.lock.release()
        deadline = time.monotonic() + timeout
        timed_out = False
        pending = ''
        try:
            with log_path.open(errors='replace') as incoming:
                incoming.seek(len(header))
                while True:
                    pending += incoming.read()
                    lines = pending.split('\n')
                    pending = lines.pop()
                    if console is not None:
                        for line in lines:
                            console.observe(key, line)
                    if process.poll() is not None:
                        # Drain output written between read() and the exit check.
                        pending += incoming.read()
                        if console is not None:
                            for line in pending.splitlines():
                                console.observe(key, line)
                        break
                    if time.monotonic() >= deadline:
                        timed_out = True
                        stop_process(process)
                        log.write(b'\n[TIMEOUT]\n')
                        log.flush()
                        break
                    time.sleep(.1)
        except BaseException:
            stop_process(process)
            raise
        finally:
            if console is not None:
                with console.lock:
                    console.processes.discard(process)
    output = log_path.read_text(errors='replace')[len(header.decode()):]
    return output, process.returncode, timed_out
