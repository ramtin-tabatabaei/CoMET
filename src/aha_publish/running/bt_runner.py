"""Run one simulator episode or a batch of tasks and failure cases.

Use ``single`` for one episode and ``batch`` for selected task/failure cases.
"""

from aha_publish import paths
import argparse
import atexit
import csv
import math
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import json
import os
import re

import sys

from pathlib import Path

from aha_publish.running import run_support as EV
from aha_publish.running.plot_maker import render_plots

PROJECT_ROOT = (paths.PROJECT_ROOT)
SCRIPT_DIR = (paths.SOURCE_DIR / 'running')
FAILGEN_ROOT = (paths.FAILGEN_ROOT)
sys.path.insert(0, str(paths.PROJECT_ROOT))
sys.path.insert(0, str(paths.SOURCE_DIR))
sys.path.insert(0, str(FAILGEN_ROOT))

CONFIGS_PATH = (paths.FAILGEN_ROOT / 'failgen/configs')
PREPARED_BTS_DIR = (paths.BT_DIR)
CONDITION_DEFINITIONS_PATH = paths.PROJECT_ROOT / 'config/condition_definitions.json'

# Condition-check images are saved by the run logger.

DEFAULT_VLM_MODEL = os.getenv('OPENAI_BT_CONDITION_VLM_MODEL', os.getenv('OPENAI_MODEL', 'gpt-5.6-sol'))
VLM_CAMERA_NAMES = ('side_rgb', 'front_rgb', 'wrist_rgb')
# OpenAI image `detail` for the condition-check frames. Unset (or 'auto') keeps
# the API default, which tiles a 256x256 render into one 512 tile at ~255
# tokens; 'low' flattens every image to a flat ~85 tokens. Input tokens are
# prefilled in parallel, so this is an input-cost knob, not a latency one -- and
# 'low' throws away the few pixels in the wrist close-up that separate a clamped
# grip from an open one (see _image_to_data_url in
# aha_scripts/detectors/collision/vlm_confirm.py). Override with
# $AHA_VLM_IMAGE_DETAIL.
VLM_IMAGE_DETAIL = os.getenv('AHA_VLM_IMAGE_DETAIL', '').strip().lower()
# The wrist camera's default field of view looks past the fingertips, so the
# fingers fall outside the frame and open-vs-closed is unreadable. We widen the
# existing wrist camera (no extra camera) at startup so wrist_rgb shows both
# fingers and the gap between them. Override with $AHA_WRIST_VLM_FOV_DEG; set to
# 0 to leave the wrist camera untouched.
WRIST_VLM_FOV_DEG = float(os.getenv('AHA_WRIST_VLM_FOV_DEG', '0'))
# When no absolute FOV is requested, widen the wrist camera's native perspective
# angle by this factor so wrist_rgb is slightly wider (1.05 = 5% wider). Set to
# 1.0 to leave the native FOV untouched. Override with $AHA_WRIST_VLM_FOV_SCALE.
WRIST_VLM_FOV_SCALE = float(os.getenv('AHA_WRIST_VLM_FOV_SCALE', '1.1'))
# The wrist camera looks straight down the gripper approach axis, so the finger
# bodies sit at the very edge of the frame. A small downward pitch tilts the
# view toward the fingers so more of the gripper is visible. Override with
# $AHA_WRIST_VLM_TILT_DEG (degrees, about the camera's local X axis); set to 0
# to leave the wrist camera pointing straight down. Flip the sign if it tilts
# the wrong way.
WRIST_VLM_TILT_DEG = float(os.getenv('AHA_WRIST_VLM_TILT_DEG', '20'))

# --- VLM token-usage accounting for one run (printed once at process exit) ---
VLM_USAGE = {'calls': 0, 'input_tokens': 0, 'output_tokens': 0}


def _price_per_1m(env_name):
    raw = os.getenv(env_name)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def print_vlm_usage_summary():
    u = VLM_USAGE
    if not u['calls']:
        return
    total = u['input_tokens'] + u['output_tokens']
    in_p = _price_per_1m('AHA_INPUT_PRICE_PER_1M')
    out_p = _price_per_1m('AHA_OUTPUT_PRICE_PER_1M')
    line = (
        f"\n[vlm-usage] this run: {u['calls']} VLM call(s), "
        f"{u['input_tokens']} input + {u['output_tokens']} output = {total} tokens"
    )
    if in_p is not None and out_p is not None:
        cost = u['input_tokens'] / 1_000_000 * in_p + u['output_tokens'] / 1_000_000 * out_p
        line += f"; cost ${cost:.4f} (at ${in_p:g}/${out_p:g} per 1M in/out)"
    else:
        line += "; cost n/a (set AHA_INPUT_PRICE_PER_1M and AHA_OUTPUT_PRICE_PER_1M to price it)"
    print(line)


atexit.register(print_vlm_usage_summary)

_GRIPPER_CAPTURE_META = {}


GRIPPER_ACTION_MAX_STEPS = int(os.getenv('AHA_GRIPPER_ACTION_MAX_STEPS', '30'))
# Opening is unobstructed, so let it run until the gripper is completely open
# (amount 1.0). The 30-step cap above is meant for closing onto an object, where
# actuate() may never report done; reusing it for opening could stop the gripper
# only partially open.
GRIPPER_OPEN_MAX_STEPS = int(os.getenv('AHA_GRIPPER_OPEN_MAX_STEPS', '200'))
PATH_ACTION_MAX_STEPS = int(os.getenv('AHA_PATH_ACTION_MAX_STEPS', '1000'))

VLM_SYSTEM_PROMPT = (
    "You are a robotics task-verification auditor for an RLBench robot simulation. "
    "You inspect camera images and decide whether behavior-tree conditions are "
    "visually satisfied at a specific checkpoint. A precondition checkpoint is "
    "before the robot moves for that waypoint. A postcondition checkpoint is "
    "after the waypoint path and any gripper action for that waypoint have "
    "completed. Use only observable visual evidence from the images. Do not use "
    "simulator state values, telemetry, numeric measurements, or hidden state. "
    "An object name ending in a numeric suffix "
    "means there are multiple instances of that object type. The number does "
    "not require a particular instance: either matching instance is acceptable. "
    "In contrast, a positional or ordinal word at the START of an object name "
    "(bottom, top, middle, upper, lower, left, right, front, back, near, far, "
    "first, second, third, fourth, last) DOES pick out one specific instance "
    "among several identical-looking siblings, and you must verify you are "
    "looking at that exact one. Locate all the sibling instances first, order "
    "them along the axis the word names (top/bottom and upper/lower by height, "
    "left/right by the horizontal axis in the view, front/back and near/far by "
    "depth from the robot base, ordinals counting from the top or from the "
    "nearest side unless the definition says otherwise), then judge the "
    "condition only against the instance the name selects. Decide which "
    "instance the robot is manipulating from the front and side RGB views "
    "(front_rgb and side_rgb): they show the gripper's height, depth, and "
    "lateral position against the whole set of siblings. The wrist view is too "
    "close to show the siblings it is not pointing at, so use it only to "
    "confirm that choice, never to make it. If the images do not "
    "let you tell which sibling is which, mark the condition uncertain rather "
    "than judging a different instance. "
    "Focus on the object the robot gripper is holding or interacting with, "
    "rather than another matching object elsewhere in the scene. For placement "
    "or release checks, follow that manipulated object to its destination. "
    "Use the three statuses precisely: mark a condition not_satisfied ONLY when "
    "the images clearly show it is violated; if the relevant evidence is occluded, "
    "out of frame, or ambiguous, mark it uncertain (not not_satisfied) instead of "
    "guessing. Return strict JSON only."
)

CAMERAS = [
    "default",
    "_cam_front",
    "_cam_over_shoulder_left",
    "_cam_over_shoulder_right",
    "_cam_overhead",
    "_cam_wrist",
    "side",
    "cam_cinematic_placeholder",
]

CAMERA_LABELS = {
    "default": "Default simulator camera",
    "_cam_front": "Front camera",
    "_cam_over_shoulder_left": "Left shoulder camera",
    "_cam_over_shoulder_right": "Right shoulder camera",
    "_cam_overhead": "Overhead camera",
    "_cam_wrist": "Wrist camera (robot hand)",
    "side": "Detector side camera",
    "cam_cinematic_placeholder": "Cinematic view",
}

SIDE_CAMERA_NAME = 'aha_side_camera'
SIDE_CAMERA_POSITION = (-0.547, -0.803, 1.209)  # nudged ~0.45 m forward along the look axis
SIDE_CAMERA_ORIENTATION_DEG = (-100, 38.3, -180.0)  # panned left


class LiveCameraView:
    def __init__(self, camera, title, refresh_sec=0.05, initial_frame=None):
        import matplotlib.pyplot as plt
        import numpy as np

        self.plt = plt
        self.np = np
        self.camera = camera
        self.refresh_sec = refresh_sec
        self.latest_frame = None
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8.5, 7.2))
        self.fig.canvas.manager.set_window_title(title)
        self.ax.set_title(title)
        self.ax.axis('off')
        if initial_frame is None:
            initial_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        else:
            initial_frame = np.asarray(initial_frame, dtype=np.uint8)
            self.latest_frame = initial_frame.copy()
        self.image_artist = self.ax.imshow(initial_frame)
        self.plt.ion()
        self.fig.show()
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def update(self):
        frame = capture_camera_rgb(self.camera)
        self.latest_frame = frame.copy()
        self.image_artist.set_data(frame)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)
        return self.latest_frame


def widen_wrist_camera(scene, fov_deg=WRIST_VLM_FOV_DEG, fov_scale=WRIST_VLM_FOV_SCALE):
    """Widen the existing wrist camera's field of view so wrist_rgb is slightly
    wider (no extra camera is created). An explicit ``fov_deg`` sets the
    perspective angle absolutely; otherwise the native angle is scaled by
    ``fov_scale`` (1.05 = 5% wider). Returns the applied FOV in degrees, or None
    if it was left unchanged."""
    wrist_cam = getattr(scene, '_cam_wrist', None)
    if wrist_cam is None:
        return None
    # PyRep VisionSensor perspective angles are in DEGREES.
    if fov_deg and fov_deg > 0:
        wrist_cam.set_perspective_angle(fov_deg)
        return fov_deg
    if fov_scale and fov_scale != 1.0:
        try:
            current = float(wrist_cam.get_perspective_angle())
        except Exception:
            return None
        new_fov = current * fov_scale
        wrist_cam.set_perspective_angle(new_fov)
        return new_fov
    return None


def tilt_wrist_camera(scene, tilt_deg=WRIST_VLM_TILT_DEG):
    """Pitch the wrist camera down by ``tilt_deg`` degrees (about its own local
    X axis) so wrist_rgb shows more of the gripper body instead of looking
    straight past the fingertips. The camera is parented to the gripper, so a
    rotation in its own frame persists as the arm moves. Returns the applied
    tilt, or None if it was left unchanged."""
    import numpy as np

    if not tilt_deg:
        return None
    wrist_cam = getattr(scene, '_cam_wrist', None)
    if wrist_cam is None:
        return None
    # PyRep Object.rotate takes Euler angles in RADIANS, applied about the
    # object's own reference frame.
    wrist_cam.rotate([float(np.radians(tilt_deg)), 0.0, 0.0])
    return tilt_deg


def get_camera_handle(scene, camera_name):
    from pyrep.backend import sim

    if camera_name == "default":
        return sim.simGetObjectHandle("DefaultCamera")
    if camera_name == "side":
        return get_or_create_side_camera(scene).get_handle()
    if camera_name.startswith("_cam"):
        return getattr(scene, camera_name).get_handle()
    return sim.simGetObjectHandle(camera_name)


def get_or_create_side_camera(scene):
    existing = getattr(scene, '_aha_side_camera', None)
    if existing is not None:
        return existing

    import numpy as np
    from pyrep.const import RenderMode
    from pyrep.objects.vision_sensor import VisionSensor

    side_camera = VisionSensor.create([256, 256])
    side_camera.set_name(SIDE_CAMERA_NAME)
    side_camera.set_render_mode(RenderMode.OPENGL3)
    side_camera.set_position(list(SIDE_CAMERA_POSITION))
    side_camera.set_orientation(np.radians(SIDE_CAMERA_ORIENTATION_DEG))
    scene._aha_side_camera = side_camera
    return side_camera


def capture_camera_rgb(camera):
    import numpy as np

    # capture_rgb() returns floats in [0, 1], but a freshly created sensor that
    # has not rendered yet can return NaN/inf or out-of-range values. Sanitize
    # before scaling so we never overflow the multiply or cast invalid values.
    frame = np.asarray(camera.capture_rgb(), dtype=np.float64)
    frame = np.nan_to_num(frame, nan=0.0, posinf=1.0, neginf=0.0)
    frame = np.clip(frame, 0.0, 1.0)
    return (frame * 255.0).astype(np.uint8)


def camera_frame_has_content(frame):
    import numpy as np

    if frame is None:
        return False
    arr = np.asarray(frame)
    return arr.size > 0 and (float(arr.max()) > 8.0 or float(arr.mean()) > 2.0)


def attach_side_rgb(obs, side_camera):
    """Give this observation its OWN fresh side-camera frame.

    The detector VLM falls back to a single live side-camera capture when an
    observation has no ``side_rgb`` (vlm_confirm._source_to_image), which collapses
    the per-step image sequence to one moment. The standalone detector harness
    avoids this by stamping a fresh ``side_rgb`` on every observation; we do the
    same here so the recent-observation buffer shows distinct timepoints.
    """
    if obs is None or side_camera is None:
        return obs
    try:
        setattr(obs, 'side_rgb', capture_camera_rgb(side_camera))
    except Exception:
        pass
    return obs


def warm_up_camera_sensor(scene, camera, steps=12):
    """Step the renderer until a newly created vision sensor has a real frame."""
    latest_frame = None
    for _ in range(max(1, steps)):
        try:
            scene.pyrep.step()
        except Exception:
            try:
                scene.step()
            except Exception:
                pass
        try:
            refresh_camera_sensors(scene)
        except Exception:
            pass
        try:
            latest_frame = capture_camera_rgb(camera)
            if camera_frame_has_content(latest_frame):
                return latest_frame
        except Exception:
            pass
    return latest_frame


def sync_default_camera(scene, camera_name):
    """Make the visible CoppeliaSim GUI camera match the selected RLBench camera."""
    from pyrep.backend import sim

    if camera_name == "default":
        return
    src = get_camera_handle(scene, camera_name)
    if sim.lib.simAdjustView(-1, src, 8, sim.ffi.NULL) >= 0:
        return
    dst = sim.simGetObjectHandle("DefaultCamera")
    pos = sim.simGetObjectPosition(src, -1)
    rot = sim.simGetObjectOrientation(src, -1)
    sim.simSetObjectPosition(dst, -1, pos)
    sim.simSetObjectOrientation(dst, -1, rot)


def refresh_camera_sensors(scene):
    """Refresh RLBench vision sensors that use explicit handling."""
    try:
        return scene.get_observation()
    except Exception:
        return None


def pyrep_step_with_camera(scene, camera_name):
    scene.pyrep.step()
    refresh_camera_sensors(scene)
    if camera_name == "default":
        return
    try:
        sync_default_camera(scene, camera_name)
    except Exception:
        pass


def env_step_with_camera(env_wrapper, scene, camera_name):
    scene.step()
    obs = refresh_camera_sensors(scene)
    if obs is not None:
        env_wrapper.on_env_step(obs)
    if camera_name != "default":
        try:
            sync_default_camera(scene, camera_name)
        except Exception:
            pass
    return obs


def env_pyrep_step_with_camera(env_wrapper, scene, camera_name):
    scene.pyrep.step()
    obs = refresh_camera_sensors(scene)
    if obs is not None:
        env_wrapper.on_env_step(obs)
    if camera_name != "default":
        try:
            sync_default_camera(scene, camera_name)
        except Exception:
            pass
    return obs


def get_all_tasks():
    files = sorted(CONFIGS_PATH.glob('*.yaml'))
    return [path.stem for path in files]


def load_task_config(task_name):
    import yaml

    path = CONFIGS_PATH / f'{task_name}.yaml'
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def load_yaml_descriptions(task_name):
    config = load_task_config(task_name)
    descriptions = {}
    for st in config.get('sub-tasks', []):
        processes = st.get('processes', [])
        desc_list = st.get('task_description', [''])
        if len(processes) >= 2:
            wp_from = int(processes[0].replace('waypoint', ''))
            descriptions[wp_from] = desc_list[0] if desc_list else ''
    return descriptions


def pick_task():
    tasks = get_all_tasks()
    return pick_from_list('AVAILABLE TASKS', tasks)


def pick_failure(config):
    from aha_publish.simulation.failures import get_available_failures

    failures = get_available_failures(config)
    print("\n" + "=" * 80)
    print("  CHOOSE FAILURE TYPE")
    print("=" * 80)
    print("  [ 0] No failure")
    for i, failure in enumerate(failures):
        wps = failure.get('waypoints', [])
        print(f"  [{i + 1:>2}] {failure.get('type', '?'):<20} waypoints: {wps}")
    print("=" * 80)
    while True:
        try:
            choice = int(input("\nEnter failure number: "))
            if choice == 0:
                return None, -1
            if 1 <= choice <= len(failures):
                failure = failures[choice - 1]
                ftype = failure.get('type')
                if ftype == 'wrong_sequence_v2':
                    return ftype, -1
                return ftype, pick_failure_waypoint(failure.get('waypoints', []))
        except ValueError:
            pass
        print("Invalid choice.")


def pick_failure_waypoint(waypoints):
    if not waypoints:
        return -1
    print("\n" + "=" * 60)
    print("  WAYPOINTS FOR THIS FAILURE")
    print("=" * 60)
    for i, waypoint in enumerate(waypoints):
        print(f"  [{i + 1:>2}] Waypoint {waypoint}")
    print("=" * 60)
    while True:
        try:
            choice = int(input("\nEnter waypoint number: "))
            if 1 <= choice <= len(waypoints):
                return int(waypoints[choice - 1])
        except ValueError:
            pass
        print("Invalid choice.")


def resolve_failure_choice(config, args):
    if args.failure is None:
        return pick_failure(config)

    if args.failure in ("", "none", "None", "NO_FAIL", "no_failure"):
        return None, -1

    from aha_publish.simulation.failures import get_available_failures

    failures = get_available_failures(config)
    valid_types = [failure.get('type') for failure in failures]
    if args.failure not in valid_types:
        raise ValueError(
            f"Failure '{args.failure}' is not available for this task. "
            f"Available: {', '.join(str(item) for item in valid_types)}"
        )

    matching = next(failure for failure in failures if failure.get('type') == args.failure)
    waypoint = args.failure_waypoint
    if args.failure == 'wrong_sequence_v2':
        # Whole-sequence shuffle: no single injection waypoint.
        return args.failure, -1
    if waypoint is None:
        waypoint = pick_failure_waypoint(matching.get('waypoints', []))
    return args.failure, int(waypoint)


def configure_failure(env_wrapper, failtype, waypoint):
    from failgen.fail_wrong_object import WrongObjectFailure
    from failgen.fail_sequence_v2 import WrongSequenceV2Failure

    if not failtype:
        for fail_obj in env_wrapper.manager._failures:
            fail_obj.set_enabled(False)
        print("Failure mode: none")
        return

    has_failtype = False
    target_fail_obj = None
    for fail_obj in env_wrapper.manager._failures:
        if fail_obj.failure_type == failtype:
            fail_obj.set_enabled(True)
            has_failtype = True
            target_fail_obj = fail_obj
        else:
            fail_obj.set_enabled(False)

    if not has_failtype:
        raise RuntimeError(f"Failure type '{failtype}' not found for this task.")

    if (
        target_fail_obj
        and failtype not in (WrongObjectFailure.FAILURE_TYPE, WrongSequenceV2Failure.FAILURE_TYPE)
        and waypoint >= 0
    ):
        target_fail_obj.change_waypoint_fail_name(f"waypoint{waypoint}")

    wp_label = f"waypoint {waypoint}" if waypoint >= 0 else "configured waypoint"
    print(f"Failure mode: {failtype} at {wp_label}")


def wait_for_space(prompt):
    """Wait for one space or Enter key press."""
    print(prompt, end='', flush=True)
    if not sys.stdin.isatty():
        try:
            input()
        except EOFError:
            print("  [auto-close]")
        return

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in (' ', '\n', '\r'):
                print("  [accepted]")
                return
            if ch.lower() == 'q':
                print("\nStopped by user.")
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def ask_yes_no(prompt, default=False):
    suffix = " [Y/n] " if default else " [y/N] "
    if not sys.stdin.isatty():
        return default
    raw = input(prompt + suffix).strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes")


def pick_from_list(title, items, label_fn=str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    for i, item in enumerate(items):
        print(f"  [{i + 1:>2}] {label_fn(item)}")
    print("=" * 60)
    while True:
        try:
            choice = int(input("\nEnter number: ")) - 1
            if 0 <= choice < len(items):
                return items[choice]
        except ValueError:
            pass
        print("Invalid choice.")


def prepared_bt_candidates(task_name):
    pattern = f'{task_name}.bt_conditions.json'
    return sorted(PREPARED_BTS_DIR.rglob(pattern))


def pick_bt_conditions_path(task_name):
    candidates = prepared_bt_candidates(task_name)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return pick_from_list(
            'AVAILABLE PREPARED BT CONDITION FILES',
            candidates,
            lambda path: str(path.relative_to(PROJECT_ROOT)),
        )

    all_condition_files = sorted(PREPARED_BTS_DIR.rglob('*.bt_conditions.json'))
    if not all_condition_files:
        raise FileNotFoundError(f'No .bt_conditions.json files found in {PREPARED_BTS_DIR}')

    print(f"\nNo prepared BT condition file found for task '{task_name}'.")
    print("Choose another prepared condition file, or press Ctrl+C to stop.")
    return pick_from_list(
        'AVAILABLE PREPARED BT CONDITION FILES',
        all_condition_files,
        lambda path: str(path.relative_to(PROJECT_ROOT)),
    )


def selected_condition_blocks(blocks):
    selected = []
    for block in blocks or []:
        if block.get('selected', True) is False:
            continue
        condition = block.get('condition') or block.get('original_condition')
        if not condition:
            continue
        selected.append(block)
    return selected


def load_bt_stages(path):
    from aha_publish.behavior_trees.failure_definitions import hold_conditions_for_saved_stage
    from aha_publish.detectors import load_primitive_policy

    primitive_policy = load_primitive_policy()
    with open(path, 'r') as f:
        data = json.load(f)

    source_name = 'review'
    stages = (data.get('review') or {}).get('stages') or []
    if not stages:
        source_name = 'generated'
        stages = (data.get('generated') or {}).get('stages') or []
    if not stages:
        raise ValueError(f'No review.stages or generated.stages found in {path}')

    normalized = []
    for index, stage in enumerate(stages):
        normalized.append({
            'stage': int(stage.get('stage', index)),
            'name': stage.get('name') or f'Stage {stage.get("stage", index)}',
            'summary': stage.get('summary', ''),
            'primitive': str(stage.get('primitive') or '').strip().lower(),
            'preconditions': selected_condition_blocks(stage.get('preconditions')),
            'postconditions': selected_condition_blocks(stage.get('postconditions')),
            'hold_conditions': selected_condition_blocks(hold_conditions_for_saved_stage(
                stage, primitive_policy=primitive_policy)),
        })
    return data, source_name, normalized


def stage_for_waypoint(stages, waypoint_index):
    for stage in stages:
        if stage['stage'] == waypoint_index:
            return stage
    if waypoint_index < len(stages):
        return stages[waypoint_index]
    return {
        'stage': waypoint_index,
        'name': f'Waypoint {waypoint_index}',
        'summary': '',
        'preconditions': [],
        'postconditions': [],
    }


def condition_text(block):
    text = block.get('condition') or block.get('original_condition') or '(empty condition)'
    failures = [
        link.get('failure')
        for link in block.get('failure_links', [])
        if link.get('failure')
    ]
    if failures:
        return f"{text}  [{', '.join(failures)}]"
    return text


def print_condition_section(title, blocks):
    print(title)
    if not blocks:
        print("  - (none)")
        return
    for block in blocks:
        print(f"  - {condition_text(block)}")


def condition_strings(blocks):
    return [
        block.get('condition') or block.get('original_condition') or ''
        for block in blocks or []
        if block.get('condition') or block.get('original_condition')
    ]


def condition_dedupe_key(condition):
    key = re.sub(r'\s+', '', condition.strip().lower())
    key = key.replace('==true', '=true').replace('==false', '=false')
    return key


def dedupe_labeled_condition_groups(groups):
    labeled_conditions = []
    skipped_conditions = []
    seen = set()
    for label, blocks in groups:
        for condition in condition_strings(blocks):
            key = condition_dedupe_key(condition)
            labeled = f"{label}: {condition}"
            if key in seen:
                skipped_conditions.append(labeled)
                continue
            seen.add(key)
            labeled_conditions.append(labeled)
    return labeled_conditions, skipped_conditions


def print_preconditions(stage, waypoint_index):
    print("\n" + "-" * 70)
    print(f"PRE-CHECK before waypoint {waypoint_index}: {stage['name']}")
    print_condition_section("Preconditions:", stage['preconditions'])
    print("-" * 70)


def print_postconditions(current_stage, waypoint_index, final_waypoint_index):
    print("\n" + "-" * 70)
    print(f"POST-CHECK after waypoint {waypoint_index}: {current_stage['name']}")
    print_condition_section("Postconditions:", current_stage['postconditions'])
    if waypoint_index >= final_waypoint_index:
        print()
        print("No next waypoint preconditions; this was the final waypoint.")
    print("-" * 70)


def hold_block_detector(block):
    return str(block.get('detector') or block.get('failure') or '').strip().lower()


def print_hold_conditions(stage, waypoint_index):
    """Announce the hold conditions the detectors monitor live during a waypoint."""
    blocks = stage.get('hold_conditions') or []
    if not blocks:
        return
    print("\n" + "-" * 70)
    print(f"HOLD CHECK during waypoint {waypoint_index}: {stage['name']}  "
          f"(monitored in real time by detectors)")
    print("Hold conditions:")
    for block in blocks:
        det = hold_block_detector(block)
        print(f"  - {condition_text(block)}  [{det} detector]")
    print("-" * 70)


def print_hold_results(stage, waypoint_index, monitor):
    """Report each hold condition's live result for the waypoint just finished."""
    blocks = stage.get('hold_conditions') or []
    if not blocks:
        return
    fired = monitor.fired_at(waypoint_index) if monitor is not None else set()
    active = monitor.active_at(waypoint_index) if monitor is not None else set()
    any_fired = bool(fired)
    print("\n" + "-" * 70)
    unevaluated = monitor.unevaluated_at(waypoint_index) if monitor is not None else set()
    verdict = "FAILED" if any_fired else ("NOT EVALUATED" if unevaluated else "held")
    print(f"HOLD RESULT for waypoint {waypoint_index}: {stage['name']}  [{verdict}]")
    for block in blocks:
        cond = condition_text(block)
        det = hold_block_detector(block)
        if monitor is None or det not in active:
            status = "not monitored"
        elif det in monitor.unevaluated_at(waypoint_index):
            status = "not evaluated — no valid target measurements"
        elif det in fired:
            status = f"VIOLATED — {det} detected"
        else:
            status = "OK"
        print(f"  - {cond:<34} {status}")
    print("-" * 70)


def waypoint_extension(wp):
    # Prefer the RLBench Waypoint's tracked extension (get_ext): failures such as
    # GraspFailure clear it via clear_ext() to drop the gripper command. Fall back
    # to the raw sim string only if get_ext is unavailable.
    try:
        return wp.get_ext()
    except Exception:
        pass
    try:
        return wp._waypoint.get_extension_string()
    except Exception:
        return ""


def print_waypoint_header(wp, waypoint_index, final_waypoint_index, stage, yaml_descs):
    # No waypoint pose here: this was the last place that read a waypoint dummy
    # from the sim, and it was only ever printed. Reference poses come from the
    # chain spec; the composed values are logged once by the [wp-chain] banner.
    ext = waypoint_extension(wp)
    default_label = yaml_descs.get(waypoint_index, "")

    print("\n" + "=" * 70)
    print(f"Waypoint {waypoint_index} of {final_waypoint_index}")
    print(f"  Action:      {ext if ext else '(just move)'}")
    print(f"  Description: {default_label or stage['summary'] or '(none)'}")
    print("=" * 70)
    return ext


def capture_checkpoint_observation(scene, side_camera=None):
    obs = scene.get_observation()
    if side_camera is not None:
        try:
            setattr(obs, 'side_rgb', capture_camera_rgb(side_camera))
        except Exception:
            pass
    return obs


def load_vlm_helpers():
    import importlib.util

    helper_path = (paths.SOURCE_DIR / 'detectors/collision/vlm_confirm.py')
    spec = importlib.util.spec_from_file_location('aha_collision_vlm_confirm', helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_openai_credential_from_local_files():
    """Use explicitly configured credentials."""
    return os.environ.get("OPENAI_API_KEY")


def make_openai_client():
    try:
        import openai
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python package 'openai'. Install it or run in an environment "
            "where the OpenAI SDK is available."
        ) from exc

    api_key = load_openai_credential_from_local_files()
    if not api_key:
        raise RuntimeError(
            "Missing OpenAI credentials. Set OPENAI_API_KEY in the shell that "
            "launches this script."
        )
    return openai.OpenAI(api_key=api_key)


def model_supports_reasoning(model):
    """True for OpenAI models that accept a `reasoning` effort parameter."""
    name = (model or '').lower()
    return name.startswith(('gpt-5', 'o1', 'o3', 'o4'))


def _flatten_condition_text(data):
    """Long text fields may be written as a JSON array of lines so the file stays
    readable; join each array back into a single space-separated string so the
    model sees exactly the same text it would from a plain string value."""
    def join(value):
        return ' '.join(str(part) for part in value) if isinstance(value, list) else value

    if not isinstance(data, dict):
        return data
    for key in ('general_guidance', 'fallback_definition'):
        if key in data:
            data[key] = join(data[key])
    for entry in (data.get('predicates') or {}).values():
        if isinstance(entry, dict) and 'definition' in entry:
            entry['definition'] = join(entry['definition'])
    return data


def load_condition_definitions(path):
    """Load the per-predicate condition definitions JSON (single source of truth).

    Long text fields (general_guidance, fallback_definition, and each predicate's
    definition) may be written either as a plain string or as a JSON array of
    lines; array values are joined with spaces at load time, so the file can be
    kept readable without changing what the model is shown."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return _flatten_condition_text(json.load(f))
    except FileNotFoundError:
        print(f"WARNING: condition definitions file not found: {path}")
        return {}
    except json.JSONDecodeError as exc:
        print(f"WARNING: could not parse condition definitions {path}: {exc}")
        return {}


_CONDITION_LABEL_RE = re.compile(r'^\s*(?:pre|post)\b[^:]*:\s*', re.IGNORECASE)


def strip_condition_label(condition):
    """Remove a boundary label prefix like 'post waypoint 0: ' from a condition."""
    return _CONDITION_LABEL_RE.sub('', condition or '')


def condition_predicate_name(condition):
    """Leading identifier of a condition, e.g. object_in_gripper(x) == True -> object_in_gripper."""
    match = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)', strip_condition_label(condition))
    return match.group(1) if match else ''


def filter_conditions_for_vlm(conditions, allow=None):
    'Keep only explicitly selected predicate names; an empty selection checks every condition.'
    result = list(conditions or [])
    if allow:
        result = [c for c in result if condition_predicate_name(c).lower() in allow]
    return result


def condition_predicate_args(condition):
    """Arguments inside the first parentheses, e.g. on(sponge, desk) -> ['sponge', 'desk']."""
    match = re.search(r'\(([^)]*)\)', strip_condition_label(condition))
    if not match or not match.group(1).strip():
        return []
    return [arg.strip() for arg in match.group(1).split(',') if arg.strip()]


_CONDITION_VALUE_RE = re.compile(r'^\s*[A-Za-z_][A-Za-z0-9_]*\s*={1,2}\s*([A-Za-z_][A-Za-z0-9_]*)\s*$')


def condition_value_name(condition):
    """Right-hand value of a `predicate = Value` condition, e.g. 'gripper_condition = Open' -> 'Open'.

    Returns '' for the usual `predicate(args) == True` form, which has parentheses
    and is keyed on the predicate name alone. Used so that a state-style predicate
    can carry a separate definition per value (gripper_condition = Open vs = Closed)."""
    match = _CONDITION_VALUE_RE.match(strip_condition_label(condition))
    return match.group(1) if match else ''


def render_definition(template, args):
    """Substitute the condition's object args into a definition template's placeholders."""
    text = template or ''
    if '<args>' in text:
        text = text.replace('<args>', ', '.join(args) if args else 'the listed objects')
    distinct = []
    for token in re.findall(r'<[^>]+>', text):
        if token not in distinct:
            distinct.append(token)
    for token, value in zip(distinct, args):
        text = text.replace(token, value)
    return text


def lookup_predicate_definition(definitions, predicate):
    predicates = (definitions or {}).get('predicates', {})
    if predicate in predicates:
        return predicates[predicate]
    for key, value in predicates.items():
        if key.lower() == predicate.lower():
            return value
    return None


def build_condition_definitions_block(conditions, definitions):
    """One rendered definition per distinct condition being checked."""
    if not definitions:
        return ''
    lines = []
    seen = set()
    for condition in conditions or []:
        if not condition or condition in seen:
            continue
        seen.add(condition)
        predicate = condition_predicate_name(condition)
        # State-style predicates may define one entry per value, keyed
        # 'gripper_condition = Open' / 'gripper_condition = Closed', so each check
        # only sees the rules for the state it is actually asserting. Fall back to
        # the bare predicate name when no value-specific entry exists.
        value = condition_value_name(condition)
        entry = None
        if value:
            entry = lookup_predicate_definition(definitions, f'{predicate} = {value}')
        if entry is None:
            entry = lookup_predicate_definition(definitions, predicate)
        args = condition_predicate_args(condition)
        if entry:
            definition = render_definition(entry.get('definition', ''), args)
        else:
            definition = (
                definitions.get('fallback_definition')
                or 'Interpret this predicate literally from the camera images.'
            )
        # For door-like objects the robot physically interacts via the handle or
        # edge — holding the handle of a door still satisfies object_in_gripper
        # for that door.
        if (predicate == 'object_in_gripper'
                and any('door' in arg.lower() for arg in args)):
            definition += (
                " NOTE: this object is a door or door panel. The robot grasps "
                "doors by their handle or edge — if the gripper is visibly "
                "gripping the handle or edge of this door, treat "
                "object_in_gripper as True for the door itself."
            )
        lines.append(f"- {condition}\n    {definition}")
    if not lines:
        return ''
    return "# Condition definitions (how to judge each predicate)\n" + "\n".join(lines)


def guidance_section_keys_for_conditions(conditions, definitions):
    """Distinct predicate-specific guidance section names needed by the
    conditions being checked, in first-seen order.

    The predicate -> guidance-section mapping lives in the definitions file
    under 'guidance_map' (e.g. object_in_gripper -> gripper_and_grasp_guidance);
    predicates absent from the map rely on general_guidance alone."""
    guidance_map = {
        str(key).lower(): value
        for key, value in ((definitions or {}).get('guidance_map') or {}).items()
    }
    keys = []
    for condition in conditions or []:
        predicate = condition_predicate_name(condition).lower()
        section = guidance_map.get(predicate)
        if section and section not in keys:
            keys.append(section)
    return keys


def build_guidance_sections_block(conditions, definitions):
    """Render the predicate-specific guidance sections (gripper/grasp,
    alignment/orientation, spatial) that apply to the conditions in this
    checkpoint, each included once. A section may be written as a JSON array of
    lines; it is joined the same way general_guidance is."""
    sections = []
    for key in guidance_section_keys_for_conditions(conditions, definitions):
        section = (definitions or {}).get(key)
        if isinstance(section, list):
            section = ' '.join(str(part) for part in section)
        if section:
            sections.append(section)
    return "\n\n".join(sections)


def vlm_prompt_text(task_name, waypoint_index, stage, checkpoint_kind, conditions, definitions=None):
    # The stage name/summary describe the commanded action for this waypoint,
    # which is a strong prior the VLM should NOT use (it must judge from the
    # images only). They are deliberately omitted from the prompt rather than
    # fed-then-forbidden. Only the conditions to evaluate are provided.
    parts = [
        "# BT condition checkpoint",
        "conditions_to_check:\n" + json.dumps(conditions, indent=2),
    ]

    general_guidance = (definitions or {}).get('general_guidance')
    if general_guidance:
        parts.append(general_guidance)

    guidance_sections = build_guidance_sections_block(conditions, definitions)
    if guidance_sections:
        parts.append(guidance_sections)

    definitions_block = build_condition_definitions_block(conditions, definitions)
    if definitions_block:
        parts.append(definitions_block)


    parts.append(
        "For each condition, decide whether it is visually satisfied at this exact "
        "checkpoint, using the condition definitions above."
    )
    explanation_spec = (
        "`explanation` (one concise sentence)."
    )
    parts.append(
        "Return JSON with exactly these keys: "
        "`checkpoint_passed` (true if every condition is satisfied; false only if "
        "at least one condition is clearly not_satisfied; null if no condition is "
        "violated but one or more are uncertain), "
        "`conditions` (array of objects with `condition`, `status` one of "
        "`satisfied`, `not_satisfied`, `uncertain`, and `evidence`), and "
        + explanation_spec
    )
    return "\n\n".join(parts)


def extract_vlm_json(text):
    stripped = text.strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find('{')
        end = stripped.rfind('}')
        if start == -1 or end <= start:
            return {
                'checkpoint_passed': None,
                'conditions': [],
                'explanation': stripped[:300],
            }
        try:
            data = json.loads(stripped[start:end + 1])
        except json.JSONDecodeError:
            return {
                'checkpoint_passed': None,
                'conditions': [],
                'explanation': stripped[:300],
            }

    conditions = data.get('conditions') or []
    checkpoint_passed = data.get('checkpoint_passed')
    if conditions:
        statuses = [str(item.get('status', '')).lower() for item in conditions]
        violated = {'not_satisfied', 'unsatisfied', 'violated', 'failed', 'fail', 'false'}
        if any(s in violated for s in statuses):
            # At least one condition is clearly violated -> hard failure.
            checkpoint_passed = False
        elif all(s == 'satisfied' for s in statuses):
            checkpoint_passed = True
        else:
            # No violation, but one or more conditions are uncertain/unknown. Keep
            # the result three-valued (None) so verdict_to_status treats it as
            # uncertain (accepted with a warning in auto mode) instead of a false
            # failure. This is what stops "uncertain" from reading as a violation.
            checkpoint_passed = None

    return {
        'checkpoint_passed': checkpoint_passed,
        'conditions': conditions,
        'explanation': str(data.get('explanation', '')).strip(),
    }





def evaluate_sim_backed_conditions(scene, conditions):
    """All conditions are forwarded to the VLM.

    object_in_gripper is judged from the camera images only; no proprioceptive
    grasp-sensor signal is injected into the prompt.
    """
    return [], list(conditions or [])


def sim_condition_result(sim_items):
    if not sim_items:
        return None
    if any(item['status'] == 'not_satisfied' for item in sim_items):
        passed = False
    elif any(item['status'] == 'unknown' for item in sim_items):
        passed = None
    else:
        passed = True
    return {
        'checkpoint_passed': passed,
        'conditions': sim_items,
        'explanation': 'Simulator-backed condition check.',
    }


def merge_condition_results(sim_items, vlm_result):
    if not sim_items:
        return vlm_result
    if vlm_result is None:
        return sim_condition_result(sim_items)
    return {
        **vlm_result,
        'conditions': sim_items + list(vlm_result.get('conditions', [])),
    }


# ---------------------------------------------------------------------------
# Post-failure failure-type diagnosis
#
# When a run ends in a failure, the type of failure is classified
# DETERMINISTICALLY (no LLM call), twice: option 1 has the live-detector fires +
# grasp force sensor as hard evidence; option 2 is blind and reasons only from
# the VLM condition-check transcript. Both apply the same ordered decision
# procedure the earlier LLM prompt described, as plain code over the structured
# evidence: live-detector fires map straight to a type; no_grasp/slip are decided
# only when a pick was expected (an object_in_gripper condition was checked),
# using the grasp force sensor + that condition's history; and the BT condition
# predicates that ended
# not_satisfied (selected_object -> wrong_object, gripper_oriented_for ->
# orientation, aligned_with/on/inside/next_to -> translation, gripper_condition
# -> wrong_sequence) supply the rest. See
# run_failure_diagnosis / _diagnose_deterministic below.
# ---------------------------------------------------------------------------

# Canonical failure-type labels used by both diagnosis options.
DIAGNOSIS_FAILURE_TYPES = [
    'collision', 'translation', 'orientation', 'wrong_sequence',
    'wrong_object', 'no_grasp', 'slip',
]
# Grip force above this (from the slip detector's max-grip telemetry) means a
# grasp actually registered on the force sensor. Mirrors SLIP_HOLD_FLOOR (0.1).
DIAGNOSIS_GRASP_FORCE_THRESHOLD = float(os.getenv('AHA_DIAGNOSIS_GRASP_FORCE', '0.1'))


def _diagnosis_detector_evidence(ctx):
    """Option-1 hard evidence: which live detectors fired (and at which
    waypoints), whether a grasp ever registered on the force sensor, and what the
    gripper is holding at the moment the run stopped."""
    fired = {}
    max_grip = None
    monitor = getattr(ctx, 'monitor', None)
    detectors = getattr(monitor, 'detectors', {}) if monitor is not None else {}
    for name, det in detectors.items():
        waypoints = sorted(getattr(det, 'fired_waypoints', None) or set())
        if waypoints:
            fired[name] = waypoints
    slip_det = detectors.get('slip')
    if slip_det is not None:
        try:
            max_grip = float(getattr(slip_det, 'diag', {}).get('max_grip'))
        except (TypeError, ValueError):
            max_grip = None
    held_now = []
    try:
        for obj in ctx.robot.gripper.get_grasped_objects():
            held_now.append(obj.get_name())
    except Exception:
        held_now = []
    grasp_ever = (None if max_grip is None
                  else bool(max_grip > DIAGNOSIS_GRASP_FORCE_THRESHOLD))
    return {
        'detectors_fired': fired,
        'max_grip_force': max_grip,
        'grasp_force_threshold': DIAGNOSIS_GRASP_FORCE_THRESHOLD,
        'grasp_ever_registered': grasp_ever,
        'currently_holding': held_now,
    }


# Condition statuses (VLM verdict strings) that count as a violated condition.
_VIOLATED_STATUSES = frozenset(
    {'not_satisfied', 'unsatisfied', 'violated', 'failed', 'fail', 'false'})

# Live-detector name -> diagnosis failure type (option-1 step 1). Mirrors the
# mapping the old prompt spelled out: the transition detector owns 'translation'.
_DETECTOR_TYPE = {
    'collision': 'collision',
    'slip': 'slip',
    'orientation': 'orientation',
    'transition': 'translation',
}

# Standalone spatial/effect predicates that evidence a wrong POSITION when they
# end not_satisfied. `\b...\(` so we match the predicate head, not a substring
# (e.g. `open(` but not the word "open" inside prose).
_TRANSLATION_PRED_RE = re.compile(
    r'\b(?:end_effector_aligned_with|aligned_with|inside|on|next_to|closed|open|'
    r'object_for_press|end_effector_in_contact_with|in_contact_with)\s*\(')


def _condition_failure_type(text):
    """Map a BT condition predicate to its diagnosis type when not_satisfied.
    Returns one of
    'wrong_object', 'orientation', 'grasp', 'sequence', 'translation', or None."""
    t = (text or '').lower()
    if 'selected_object' in t or 'object_found' in t:
        return 'wrong_object'
    if 'oriented_for' in t:                       # gripper_oriented_for(...)
        return 'orientation'
    if 'object_in_gripper' in t or 'gripper_released' in t:
        return 'grasp'                            # grasp lifecycle -> slip/no_grasp
    if 'gripper_condition' in t:                  # open/close ordering
        return 'sequence'
    if _TRANSLATION_PRED_RE.search(t):
        return 'translation'
    return None


def _grasp_lifecycle(check_history):
    """Track object_in_gripper across the run. Returns (present, ever_ok, lost,
    never_ok): whether a grasp condition was ever checked, ever satisfied, was
    satisfied then later became not_satisfied, and was checked but never
    satisfied."""
    present = ever_ok = lost = False
    for entry in check_history or []:
        for cond in entry.get('conditions', []) or []:
            text = (cond.get('condition') or '').lower()
            if 'object_in_gripper' not in text:
                continue
            present = True
            status = (cond.get('status') or '').lower()
            if status == 'satisfied':
                ever_ok = True
            elif status in _VIOLATED_STATUSES and ever_ok:
                lost = True
    return present, ever_ok, lost, (present and not ever_ok)


def _failed_condition_types(check_history):
    """(waypoint, type) for every not_satisfied condition, in transcript order.
    `type` per _condition_failure_type (None-typed conditions are dropped)."""
    out = []
    for entry in check_history or []:
        wp = entry.get('waypoint')
        for cond in entry.get('conditions', []) or []:
            if (cond.get('status') or '').lower() not in _VIOLATED_STATUSES:
                continue
            ctype = _condition_failure_type(cond.get('condition') or '')
            if ctype:
                out.append((wp, ctype))
    return out


def _diagnose_deterministic(check_history, allowed_types, evidence, stop_waypoint):
    """Deterministically classify the run-ending failure — no LLM call.

    ``evidence`` is the option-1 detector/grasp-sensor dict (see
    _diagnosis_detector_evidence); pass None for the blind option-2 pass. Applies
    the same ordered decision procedure the old LLM prompt described. Returns
    {'failure_type', 'reasoning'}."""
    def result(ftype, reasoning):
        return {'failure_type': ftype if ftype in allowed_types else 'unknown',
                'reasoning': reasoning}

    present, _grasp_ok, grasp_lost, grasp_never = _grasp_lifecycle(check_history)
    failed = _failed_condition_types(check_history)
    wrong_object_seen = any(ct == 'wrong_object' for _, ct in failed)
    sequence_seen = any(ct == 'sequence' for _, ct in failed)

    def nearest_spatial():
        """translation/orientation category of the not_satisfied condition
        nearest the waypoint where the run stopped."""
        best = None
        for wp, ctype in failed:
            if ctype not in ('translation', 'orientation'):
                continue
            key = wp if isinstance(wp, int) else -1
            if best is None or key >= best[0]:
                best = (key, ctype)
        return best[1] if best else None

    if evidence is not None:
        # --- Option 1: detector-aware. Live-detector response is authoritative. ---
        fired = evidence.get('detectors_fired') or {}
        if fired:
            stop = stop_waypoint if isinstance(stop_waypoint, int) else 0
            best_name, best_dist = None, None
            for name, wps in fired.items():
                for wp in (wps or [stop]):
                    dist = abs((wp if isinstance(wp, int) else stop) - stop)
                    if best_dist is None or dist < best_dist:
                        best_dist, best_name = dist, name
            return result(
                _DETECTOR_TYPE.get(best_name, best_name),
                f"live '{best_name}' detector fired at waypoint(s) "
                f"{sorted(fired.get(best_name) or [])}; detector response is authoritative.")
        # No detector fired: reason from conditions + grasp force sensor.
        grasp_reg = evidence.get('grasp_ever_registered')
        holding = evidence.get('currently_holding') or []
        # no_grasp only applies when the robot was actually supposed to pick an
        # object up: require an object_in_gripper condition to have been checked
        # (present) and the grasp to have never succeeded. Never infer no_grasp
        # from the force sensor alone on a task that has no pick step.
        if grasp_never:
            return result('no_grasp',
                          "no detector fired and object_in_gripper was never satisfied; "
                          "the grasp never succeeded.")
        if present and grasp_reg is False:
            # Hard evidence wins here: the grasp force sensor never registered a
            # hold, so the object was never actually gripped -> no_grasp, even if
            # the VLM's object_in_gripper briefly read satisfied then not
            # (grasp_lost). Slip is reserved for a grasp that physically happened.
            return result('no_grasp',
                          "no detector fired and the grasp force sensor never "
                          "registered a hold, so the object was never actually gripped.")
        if grasp_lost or (grasp_reg is True and not holding):
            return result('slip',
                          "no detector fired but a grasp registered on the force sensor "
                          "and the object was subsequently lost (currently holding nothing).")
        if sequence_seen:
            return result('wrong_sequence',
                          "a gripper open/close ordering condition ended not_satisfied.")
        if wrong_object_seen:
            return result('wrong_object',
                          "a selected_object/object_found condition ended not_satisfied.")
        sp = nearest_spatial()
        if sp:
            return result(sp, f"nearest-to-stop failed condition evidences {sp}.")
        return result('unknown', "no detector fired and no decisive condition failed.")

    # --- Option 2: blind (no detector/sensor access). Transcript only. ---
    if grasp_lost:
        return result('slip',
                      "an object_in_gripper condition was satisfied earlier and later "
                      "became not_satisfied.")
    if grasp_never:
        return result('no_grasp',
                      "the object_in_gripper postcondition was never satisfied after the "
                      "grasp attempt.")
    if sequence_seen:
        return result('wrong_sequence',
                      "a gripper open/close ordering condition ended not_satisfied.")
    if wrong_object_seen:
        return result('wrong_object',
                      "a selected_object/object_found condition ended not_satisfied.")
    sp = nearest_spatial()
    if sp:
        return result(sp, f"nearest-to-stop failed condition evidences {sp}.")
    return result('unknown', "no decisive condition failed in the transcript.")


def run_failure_diagnosis(ctx):
    """Deterministically classify a run-ending failure into one of
    DIAGNOSIS_FAILURE_TYPES twice: option 1 with the live-detector fires + grasp
    force sensor as evidence, and option 2 blind from only the VLM condition-check
    transcript. No LLM call is made. Both predictions are printed (parsed by the
    batch harness from run.log) and logged to the per-run VLM CSV when active.
    Stored on ``ctx.diagnosis``."""
    if not getattr(ctx, 'vlm_enabled', False) or not getattr(ctx, 'failures', None):
        return None
    check_history = getattr(ctx, 'check_history', []) or []
    evidence = _diagnosis_detector_evidence(ctx)
    stop_waypoint = ctx.failures[0].waypoint if ctx.failures else None

    print("\n" + "=" * 70)
    print("FAILURE DIAGNOSIS (post-hoc deterministic classification)")
    print("=" * 70)
    opt1 = _diagnose_deterministic(
        check_history, DIAGNOSIS_FAILURE_TYPES, evidence, stop_waypoint)
    opt2 = _diagnose_deterministic(
        check_history, DIAGNOSIS_FAILURE_TYPES, None, stop_waypoint)

    ctx.diagnosis = {'with_detectors': opt1, 'no_detectors': opt2, 'evidence': evidence}
    # These two lines are the machine-readable contract the batch harness parses.
    print(f"  [diagnosis] with_detectors: {opt1['failure_type']} | {opt1['reasoning']}")
    print(f"  [diagnosis] no_detectors: {opt2['failure_type']} | {opt2['reasoning']}")
    print("=" * 70)


    return ctx.diagnosis


def vlm_image_part(data_url):
    """One `input_image` content part, honouring $AHA_VLM_IMAGE_DETAIL."""
    part = {'type': 'input_image', 'image_url': data_url}
    if VLM_IMAGE_DETAIL in ('low', 'high', 'auto'):
        part['detail'] = VLM_IMAGE_DETAIL
    return part


def run_vlm_condition_check(obs, scene, env_wrapper, task_name, waypoint_index, stage, checkpoint_kind, conditions, model, definitions=None, reasoning_effort=None):
    if not conditions:
        return None

    helpers = load_vlm_helpers()


    prompt_text = vlm_prompt_text(
        task_name=task_name,
        waypoint_index=waypoint_index,
        stage=stage,
        checkpoint_kind=checkpoint_kind,
        conditions=conditions,
        definitions=definitions,
    )
    content = [{'type': 'input_text', 'text': prompt_text}]
    image_manifest = []

    for camera_name, data_url in helpers.collect_camera_images(
        obs,
        camera_names=VLM_CAMERA_NAMES,
        env_wrapper=env_wrapper,
    ):
        content.append({'type': 'input_text', 'text': f'Camera: {camera_name}'})
        content.append(vlm_image_part(data_url))
        image_manifest.append({'camera_name': camera_name})


    if not image_manifest:
        raise RuntimeError("No camera images were available for the VLM condition check.")



    print(
        "  [vlm] images sent: "
        + ", ".join(item['camera_name'] for item in image_manifest)
    )
    client = make_openai_client()
    create_kwargs = dict(
        model=model,
        max_output_tokens=1000,
        input=[
            {'role': 'system', 'content': VLM_SYSTEM_PROMPT},
            {'role': 'user', 'content': content},
        ],
    )
    # Lower reasoning effort is a large latency win on reasoning models. Only
    # attach it for models that support it, and fall back gracefully if the API
    # still rejects the parameter for this model.
    if reasoning_effort and model_supports_reasoning(model):
        create_kwargs['reasoning'] = {'effort': reasoning_effort}
    try:
        response = client.responses.create(**create_kwargs)
    except Exception as exc:
        if 'reasoning' in create_kwargs and 'reasoning' in str(exc).lower():
            create_kwargs.pop('reasoning')
            response = client.responses.create(**create_kwargs)
        else:
            raise

    usage = getattr(response, 'usage', None)
    in_tok = int(getattr(usage, 'input_tokens', 0) or 0)
    out_tok = int(getattr(usage, 'output_tokens', 0) or 0)
    VLM_USAGE['calls'] += 1
    VLM_USAGE['input_tokens'] += in_tok
    VLM_USAGE['output_tokens'] += out_tok
    print(
        f"  [vlm] tokens: {in_tok} in + {out_tok} out "
        f"(run total: {VLM_USAGE['input_tokens']} in + {VLM_USAGE['output_tokens']} out)"
    )

    parsed = extract_vlm_json(response.output_text)


    return {
        'checkpoint_passed': parsed['checkpoint_passed'],
        'conditions': parsed['conditions'],
        'explanation': parsed['explanation'],
    }


def print_vlm_result(checkpoint_kind, result):
    if result is None:
        return
    passed = result.get('checkpoint_passed')
    status = 'PASS' if passed is True else 'FAIL' if passed is False else 'UNKNOWN'
    print(f"\n  VLM {checkpoint_kind.upper()} CHECK: {status}")
    for item in result.get('conditions', []):
        condition = item.get('condition', '(condition)')
        item_status = item.get('status', 'unknown')
        evidence = item.get('evidence', '')
        print(f"    - {item_status}: {condition}")
        if evidence:
            print(f"      {evidence}")
    if result.get('explanation'):
        print(f"    Explanation: {result['explanation']}")
    if result.get('trace_path'):
        print(f"    Trace: {result['trace_path']}")








def maybe_run_vlm_condition_check(args, enabled, checkpoint_kind, scene, side_camera, env_wrapper, task_name, waypoint_index, stage, conditions):
    if not enabled:
        return None


    conditions = filter_conditions_for_vlm(conditions, allow=getattr(args, "vlm_predicate_filter", None))
    if not conditions:
        return None
    if args.vlm_checks == 'ask':
        label = 'preconditions' if checkpoint_kind == 'pre' else 'postconditions'
        if not ask_yes_no(
            f"\nCheck {label} for waypoint {waypoint_index} with VLM?",
            default=False,
        ):
            return None
    elif args.vlm_checks not in (checkpoint_kind, 'both'):
        return None
    if not conditions:
        return None

    print(f"\n  [vlm] checking {checkpoint_kind}conditions for waypoint {waypoint_index}...")
    sim_items, conditions = evaluate_sim_backed_conditions(scene, conditions)
    if sim_items:
        sim_result = sim_condition_result(sim_items)
        print_vlm_result(f"{checkpoint_kind} simulator", sim_result)
        if sim_result.get('checkpoint_passed') is not True:
            return sim_result
    if not conditions:
        return sim_condition_result(sim_items)
    try:
        obs = capture_checkpoint_observation(scene, side_camera)

        result = run_vlm_condition_check(
            obs=obs,
            scene=scene,
            env_wrapper=env_wrapper,
            task_name=task_name,
            waypoint_index=waypoint_index,
            stage=stage,
            checkpoint_kind=checkpoint_kind,
            conditions=conditions,
            model=args.vlm_model,
            definitions=getattr(args, 'condition_definitions_data', None),
            reasoning_effort=getattr(args, 'vlm_reasoning_effort', 'low'),
        )
        result = merge_condition_results(sim_items, result)
        print_vlm_result(checkpoint_kind, result)

        return result
    except Exception as exc:
        print(f"  [vlm] condition check failed: {exc}")
        return None


def maybe_run_vlm_boundary_check(args, enabled, scene, side_camera, env_wrapper, task_name, waypoint_index, stage, next_waypoint_index, next_stage):
    conditions, skipped_conditions = dedupe_labeled_condition_groups([
        (f"pre waypoint {next_waypoint_index}", next_stage['preconditions']),
        (f"post waypoint {waypoint_index}", stage['postconditions']),
    ])
    conditions = filter_conditions_for_vlm(conditions, allow=getattr(args, 'vlm_predicate_filter', None))
    if not enabled or not conditions:
        return None

    checkpoint_kind = f"pre{next_waypoint_index}_post{waypoint_index}"
    if args.vlm_checks == 'ask':
        if not ask_yes_no(
            (
                f"\nCheck preconditions for waypoint {next_waypoint_index} and "
                f"postconditions for waypoint {waypoint_index} with VLM?"
            ),
            default=False,
        ):
            return None
    elif args.vlm_checks not in ('pre', 'post', 'both'):
        return None

    print(
        f"\n  [vlm] checking preconditions for waypoint {next_waypoint_index} "
        f"and postconditions for waypoint {waypoint_index}..."
    )
    if skipped_conditions:
        print(
            "  [vlm] skipped repeated condition(s): "
            + "; ".join(skipped_conditions)
        )
    sim_items, conditions = evaluate_sim_backed_conditions(scene, conditions)
    if sim_items:
        sim_result = sim_condition_result(sim_items)
        print_vlm_result("PRE+POST simulator", sim_result)
        if sim_result.get('checkpoint_passed') is not True:
            return sim_result
    if not conditions:
        return sim_condition_result(sim_items)
    combined_stage = {
        'name': (
            f"{next_stage['name']} / previous postcheck: {stage['name']}"
        ),
        'summary': (
            f"Pre waypoint {next_waypoint_index}: "
            f"{next_stage.get('summary', '') or 'not provided'}\n"
            f"Post waypoint {waypoint_index}: "
            f"{stage.get('summary', '') or 'not provided'}"
        ),
    }
    try:
        obs = capture_checkpoint_observation(scene, side_camera)

        result = run_vlm_condition_check(
            obs=obs,
            scene=scene,
            env_wrapper=env_wrapper,
            task_name=task_name,
            waypoint_index=next_waypoint_index,
            stage=combined_stage,
            checkpoint_kind=checkpoint_kind,
            conditions=conditions,
            model=args.vlm_model,
            definitions=getattr(args, 'condition_definitions_data', None),
            reasoning_effort=getattr(args, 'vlm_reasoning_effort', 'low'),
        )
        result = merge_condition_results(sim_items, result)
        print_vlm_result("PRE+POST", result)

        return result
    except Exception as exc:
        print(f"  [vlm] boundary condition check failed: {exc}")
        return None


def step_sim_light(scene, camera_name):
    scene.pyrep.step()
    try:
        scene.task.step()
    except Exception:
        pass
    if camera_name == "default":
        return
    try:
        sync_default_camera(scene, camera_name)
    except Exception:
        pass


def actuate_gripper_with_camera(gripper, amount, scene, env_wrapper, camera_name, failure_active, side_camera_window=None, max_steps=GRIPPER_ACTION_MAX_STEPS, label='gripper', side_camera=None, monitor=None, monitor_waypoint=None):
    done = False
    step_count = 0
    while not done and step_count < max_steps:
        done = gripper.actuate(amount, 0.04)
        grip_obs = None
        if failure_active:
            grip_obs = env_pyrep_step_with_camera(env_wrapper, scene, camera_name)
        else:
            step_sim_light(scene, camera_name)
            if monitor is not None:
                grip_obs = refresh_camera_sensors(scene)
        if monitor is not None and grip_obs is not None:
            attach_side_rgb(grip_obs, side_camera)
            monitor.on_step(grip_obs, monitor_waypoint, True, phase="gripper")


        if step_count % 5 == 0:
            if side_camera_window is not None:
                side_camera_window.update()

        if step_count and step_count % 10 == 0:
            print(f"  [gripper] {label} still running ({step_count}/{max_steps})...", flush=True)
        step_count += 1
    if side_camera_window is not None:
        side_camera_window.update()

    return done, step_count


# ---------------------------------------------------------------------------
# Behavior-tree action primitives.
#
# These are verbatim extractions of the per-waypoint move/gripper blocks that
# used to live inline in main()'s procedural loop. They are wrapped as leaf
# actions by the py_trees behaviors in bt_tree.py (run-to-completion: one call
# executes the whole path / gripper action). Keeping the bodies identical to the
# original loop preserves the exact sim-stepping cadence, failure hooks, detector
# ticks, and motion capture.
# ---------------------------------------------------------------------------


def _apply_reference_rules(monitor, phase, i):
    """Recompute the hook-written reference waypoints for waypoint ``i``.

    On failure, only the waypoints the rules own (the report's
    rule_resolved_waypoints, i.e. the ones the source analysis flagged dynamic)
    lose their reference: their inspected reset offset is the value the hook was
    going to overwrite, so keeping it would hand the arrival detectors a
    confidently wrong target. Every other waypoint in the chain is a static
    offset the hook never touches, so it keeps the reference it already had --
    dropping those too cost tasks like place_shape_in_shape_sorter (2 hook-driven
    waypoints out of 5) their whole chain over one unevaluatable equation.
    """
    rules = getattr(monitor, 'reference_rules', None)
    if rules is None:
        return
    try:
        rules.apply(phase, monitor._original_waypoint_at(i))
    except Exception as exc:
        rules.failed = True
        chain = monitor.wp_chain
        owned = ({int(w) for w in (chain.spec.get('rule_resolved_waypoints') or [])}
                 | {int(w) for w in (chain.spec.get('dynamic_waypoints') or [])})
        dropped = [idx for idx in sorted(owned)
                   if chain.poses.pop(idx, None) is not None]
        scope = (f"waypoints {dropped}" if dropped else "no waypoint")
        print(f"  [references] NOT EVALUATED for {scope}: "
              f"inspection rule failed: {exc}")


def bt_run_move(ctx, i, wp, stage):
    """Execute one waypoint path while feeding the live detectors."""
    scene = ctx.scene
    env_wrapper = ctx.env_wrapper
    monitor = ctx.monitor
    chosen_cam = ctx.chosen_cam
    side_camera = ctx.side_camera
    side_camera_window = ctx.side_camera_window

    failure_active = ctx.failure_active

    if monitor is not None:
        print_hold_conditions(stage, i)
    print(f"  [move] starting waypoint {i}...")
    # RLBench prepares the legitimate target first; failure hooks perturb it
    # afterward. Detector references come exclusively from inspection data.
    if monitor is not None:
        if ctx.task_name in ('take_item_out_of_drawer', 'change_channel'):
            from aha_publish.simulation.waypoint_reference_rules import refresh_moving_object_reference
            from pyrep.objects.object import Object as ReferenceObject
            ref_index = monitor._original_waypoint_at(i)
            try:
                refreshed = refresh_moving_object_reference(
                    ctx.task_name, monitor.wp_chain, ref_index,
                    lambda name: ReferenceObject.get_object(name).get_pose())
                if refreshed:
                    print(f"  [references] waypoint {ref_index}: refreshed object "
                          "anchor; target fixed for this motion")
            except Exception as exc:
                print(f"  [references] NOT EVALUATED for waypoint {ref_index}: "
                      f"object anchor refresh failed: {exc}")
        _apply_reference_rules(monitor, 'start', i)
    wp.start_of_path()
    if monitor is not None:
        monitor.begin_waypoint(i)
    if failure_active and not wp.skip:
        env_wrapper.on_env_waypoint(wp)
    if not wp.skip:
        try:
            print(f"  [move] planning path for waypoint {i}...")
            path = wp.get_path()
            print(f"  [move] executing path for waypoint {i}...", flush=True)
            done = False
            step_count = 0
            while not done and step_count < PATH_ACTION_MAX_STEPS:
                paused = (
                    env_wrapper.on_env_should_pause_path(
                        path=path,
                        path_done=False,
                    )
                    if failure_active
                    else False
                )
                if not paused:
                    done = path.step()
                step_obs = None
                if failure_active:
                    step_obs = env_step_with_camera(env_wrapper, scene, chosen_cam)
                else:
                    step_sim_light(scene, chosen_cam)
                    if monitor is not None:
                        step_obs = refresh_camera_sensors(scene)
                if monitor is not None and step_obs is not None:
                    attach_side_rgb(step_obs, side_camera)
                    monitor.on_step(step_obs, i, done)


                if step_count % 5 == 0:
                    if side_camera_window is not None:
                        side_camera_window.update()

                if step_count and step_count % 25 == 0:
                    print(
                        f"  [move] waypoint {i} still executing "
                        f"({step_count}/{PATH_ACTION_MAX_STEPS})...",
                        flush=True,
                    )
                step_count += 1
            if not done:
                print(
                    "  [move] waypoint path reached step limit "
                    f"({step_count}); continuing to avoid a hang."
                )
            if failure_active:
                while env_wrapper.on_env_should_pause_path(
                    path=path,
                    path_done=True,
                ):
                    pause_obs = env_pyrep_step_with_camera(
                        env_wrapper, scene, chosen_cam
                    )
                    if monitor is not None and pause_obs is not None:
                        attach_side_rgb(pause_obs, side_camera)
                        monitor.on_step(pause_obs, i, True)

                    if side_camera_window is not None:
                        side_camera_window.update()

            print(f"  [move] finished waypoint {i} path in {step_count} sim steps.")
        except Exception as e:
            print(f"  ERROR: {e}")
    if not wp.skip:
        if monitor is not None:
            _apply_reference_rules(monitor, 'end', i)
        wp.end_of_path()
    if failure_active:
        env_wrapper.on_env_waypoint_end(wp)
    if side_camera_window is not None:
        side_camera_window.update()


def bt_run_gripper(ctx, i, wp):
    """Execute one waypoint's gripper action (open/close from the waypoint
    extension), or do nothing when the waypoint carries no gripper command.

    Reads the extension fresh: a grasp-failure hook in bt_run_move may have
    cleared it via clear_ext(), and this honors that dropped command."""
    if wp.skip:
        return
    scene = ctx.scene
    env_wrapper = ctx.env_wrapper
    monitor = ctx.monitor
    chosen_cam = ctx.chosen_cam
    side_camera = ctx.side_camera
    side_camera_window = ctx.side_camera_window

    failure_active = ctx.failure_active


    ext = waypoint_extension(wp)
    if ext:
        gripper = ctx.robot.gripper
        if 'open_gripper(' in ext:
            print(f"  [gripper] opening for waypoint {i}...", flush=True)
            # Tag this exact frame as the open-command frame in slip telemetry.
            if monitor is not None and hasattr(monitor, "note_gripper_open_command"):
                monitor.note_gripper_open_command()
            gripper.release()
            done, gripper_steps = actuate_gripper_with_camera(
                gripper=gripper,
                amount=1.0,
                scene=scene,
                env_wrapper=env_wrapper,
                camera_name=chosen_cam,
                failure_active=failure_active,
                side_camera_window=side_camera_window,
                max_steps=GRIPPER_OPEN_MAX_STEPS,
                label='open',
                side_camera=side_camera,
                monitor=monitor,
                monitor_waypoint=i,
            )
            if not done:
                print(
                    "  [gripper] open action reached step limit "
                    f"({gripper_steps}); continuing."
                )
            if side_camera_window is not None:
                side_camera_window.update()

        elif 'close_gripper(' in ext:
            print(f"  [gripper] closing for waypoint {i}...", flush=True)
            # Tag this frame as the close-command frame in slip telemetry (starts
            # the grasp window / resets the held-peak baseline for method 1).
            if monitor is not None and hasattr(monitor, "note_gripper_close_command"):
                monitor.note_gripper_close_command()
            done, gripper_steps = actuate_gripper_with_camera(
                gripper=gripper,
                amount=0.0,
                scene=scene,
                env_wrapper=env_wrapper,
                camera_name=chosen_cam,
                failure_active=failure_active,
                side_camera_window=side_camera_window,
                label='close',
                side_camera=side_camera,
                monitor=monitor,
                monitor_waypoint=i,
            )
            if not done:
                print(
                    "  [gripper] close action reached step limit "
                    f"({gripper_steps}); attempting grasp and continuing."
                )
            for g_obj in scene.task.get_graspable_objects():
                gripper.grasp(g_obj)
            if side_camera_window is not None:
                side_camera_window.update()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Interactive waypoint runner that prints BT preconditions before each "
            "move and postconditions after each waypoint."
        ),
        epilog="For multiple episodes, use: python -m aha_publish.running.bt_runner batch --help",
    )
    parser.add_argument('--task', help='RLBench/failgen task name.')
    parser.add_argument(
        '--mode',
        choices=('manual', 'auto'),
        default='manual',
        help=(
            'Behavior-tree run mode. "manual" (default) asks at each checkpoint '
            'whether to verify the conditions with the VLM (answering no accepts '
            'them as satisfied). "auto" verifies pre/post conditions with the VLM '
            'and hold conditions with the detectors automatically, no prompts.'
        ),
    )

    parser.add_argument(
        '--camera',
        choices=CAMERAS,
        help='Camera to show in the CoppeliaSim GUI.',
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run CoppeliaSim headless (no GUI window). Useful for batch/parallel '
             'runs. Defaults to a visible GUI.',
    )
    parser.add_argument(
        '--bt-conditions',
        type=Path,
        help='Path to a prepared .bt_conditions.json file.',
    )
    parser.add_argument(
        '--condition-definitions',
        type=Path,
        default=CONDITION_DEFINITIONS_PATH,
        help='Path to the per-predicate BT condition definitions JSON used in VLM prompts.',
    )


    parser.add_argument(
        '--failure',
        help="Failure type from the task config. Use 'none' for no failure.",
    )
    parser.add_argument(
        '--failure-waypoint',
        type=int,
        help='Waypoint index where the selected failure should be injected.',
    )


    parser.add_argument(
        '--vlm-checks',
        choices=('ask', 'off', 'pre', 'post', 'both'),
        default='ask',
        help=(
            'Use OpenAI VLM to check BT conditions. Default ask prompts at each '
            'pre/post checkpoint; pre/post/both run automatically.'
        ),
    )
    parser.add_argument(
        '--vlm-predicates',
        default=None,
        help='Comma-separated predicate names to restrict VLM checking to (e.g. '
             '"gripper_oriented_for,end_effector_aligned_with"). Conditions whose '
             'predicate is not listed are skipped (treated as satisfied). Default: '
             'check all conditions.',
    )



    parser.add_argument(
        '--vlm-model',
        default=DEFAULT_VLM_MODEL,
        help='OpenAI model for BT condition VLM checks. Use a faster/smaller '
             'model here to speed up checks (e.g. a mini variant).',
    )
    parser.add_argument(
        '--vlm-reasoning-effort',
        choices=('minimal', 'low', 'medium', 'high'),
        default='low',
        help='Reasoning effort for reasoning-capable models (gpt-5 / o-series). '
             'Lower is much faster; "minimal" is fastest. Ignored by models that '
             'do not support reasoning.',
    )


    parser.add_argument(
        '--vlm-diagnose-failure',
        dest='vlm_diagnose_failure',
        action='store_true',
        help='After a run-ending failure, run a text-only LLM diagnosis that '
             'classifies the failure type (collision/translation/orientation/'
             'wrong_sequence/wrong_object/no_grasp/slip) twice: once with '
             'the live-detector fires + grasp force sensor as evidence (option 1) '
             'and once blind from only the VLM condition-check transcript (option 2). '
             'On by default.',
    )
    parser.add_argument(
        '--no-vlm-diagnose-failure',
        dest='vlm_diagnose_failure',
        action='store_false',
        help='Disable the post-failure failure-type LLM diagnosis.',
    )
    parser.set_defaults(vlm_diagnose_failure=True)



    parser.add_argument(
        '--no-side-camera-window',
        action='store_true',
        help='Do not open the detector side-camera image window.',
    )
    parser.add_argument(
        '--live-detectors',
        action='store_true',
        help='Run the runtime detectors (collision/slip) live during '
             'the run and report detections (on by default).',
    )
    parser.add_argument(
        '--no-live-detectors',
        dest='live_detectors',
        action='store_false',
        help='Disable live runtime detector monitoring.',
    )
    parser.set_defaults(live_detectors=True)


    return parser.parse_args(argv)


def run_single(argv=None):
    args = parse_args(argv)

    seed_raw = os.getenv("AHA_RUN_SEED", "").strip()
    if seed_raw:
        try:
            seed = int(seed_raw)
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)
            print(f"Run seed: {seed}")
        except ValueError:
            print(f"WARNING: ignoring invalid AHA_RUN_SEED={seed_raw!r}")

    # Resolve --mode onto --vlm-checks, which the rest of the code keys off.
    # manual -> keep the interactive 'ask' default (answering no accepts the
    # conditions). auto -> verify automatically; default to 'both' unless the
    # user explicitly narrowed coverage with --vlm-checks.
    if args.mode == 'auto' and args.vlm_checks == 'ask':
        args.vlm_checks = 'both'

    if args.mode == 'auto' and 'AHA_DETECTOR_VLM_AUTO' not in os.environ:
        os.environ['AHA_DETECTOR_VLM_AUTO'] = '1'

    # Optional predicate allow-list: only these predicates are checked by the VLM.
    args.vlm_predicate_filter = (
        {p.strip().lower() for p in args.vlm_predicates.split(',') if p.strip()}
        if args.vlm_predicates
        else None
    )
    if args.vlm_predicate_filter:
        print(f"VLM predicate filter (only these checked): "
              f"{', '.join(sorted(args.vlm_predicate_filter))}")

    from failgen.env_wrapper import FailGenEnvWrapper

    # 1) Choose the task first.
    task_name = args.task or pick_task()
    task_config = load_task_config(task_name)

    # 2) Then choose the failure type, and 3) then which waypoint.
    failtype, failure_waypoint = resolve_failure_choice(task_config, args)
    failure_active = failtype is not None

    chosen_cam = args.camera or "default"
    vlm_enabled = args.vlm_checks != 'off'

    # Detector that owns the injected failure (slip->slip, collision->collision,
    # rotation_*->orientation, translation_*->transition).
    # Used to stop the run only on THAT detector's confirmed fire (cross-fires
    # from other detectors are still recorded/plotted but do not abort).
    from aha_publish.running.run_support import responsible_detector_for
    args.responsible_detector = (
        responsible_detector_for(failtype) if failure_active else None)

    args.condition_definitions_data = (
        load_condition_definitions(args.condition_definitions) if vlm_enabled else {}
    )

    yaml_descs = load_yaml_descriptions(task_name)

    # BT condition source is selected after the task/failure choices.
    bt_conditions_path = (
        args.bt_conditions.resolve()
        if args.bt_conditions
        else pick_bt_conditions_path(task_name)
    )

    bt_data, bt_source_name, bt_stages = load_bt_stages(bt_conditions_path)
    bt_task_name = bt_data.get('task_name')
    if bt_task_name and bt_task_name != task_name:
        print(
            f"\nWARNING: BT file task_name is '{bt_task_name}', "
            f"but simulator task is '{task_name}'."
        )

    print(f"\nLaunching simulator for '{task_name}' with camera '{chosen_cam}'...")
    print(f"BT mode: {args.mode}")
    print(f"Failure: {failtype if failtype else 'none'}")
    if failtype and failure_waypoint >= 0:
        print(f"Failure waypoint: {failure_waypoint}")
    print(f"VLM checks: {args.vlm_checks if vlm_enabled else 'off'}")
    if vlm_enabled:
        print(f"VLM model: {args.vlm_model}")
        if model_supports_reasoning(args.vlm_model):
            print(f"VLM reasoning effort: {args.vlm_reasoning_effort}")
        if VLM_IMAGE_DETAIL:
            print(f"VLM image detail: {VLM_IMAGE_DETAIL}")
        print(f"Condition definitions: {args.condition_definitions}")
    print(f"BT conditions: {bt_conditions_path}")
    print(f"Condition source: {bt_source_name}.stages\n")

    env_wrapper = FailGenEnvWrapper(
        task_name=task_name,
        headless=args.headless,
        record=False,
        save_data=True,
        save_path=str(paths.BACKEND_DATA_DIR / 'running'),
        save_keyframes_only=False,
    )
    configure_failure(env_wrapper, failtype, failure_waypoint)
    env_wrapper.reset()

    try:
        scene = env_wrapper._env._scene
        scene.robot.arm.set_control_loop_enabled(True)

        try:
            sync_default_camera(scene, chosen_cam)
            pyrep_step_with_camera(scene, chosen_cam)
            label = CAMERA_LABELS.get(chosen_cam, chosen_cam)
            print(f"View set to: {label}")
        except Exception as e:
            print(f"Could not set camera: {e}")

        # Speed knob: skip the side-camera vision sensor entirely when it can't
        # be used (no VLM, no side window). The sensor is rendered every physics
        # step on software GL, so creating it for pose-only detector calibration
        # is pure overhead. Gated behind AHA_NO_SIDE_CAMERA so default runs are
        # unchanged. All downstream side_camera uses are None-safe.
        _skip_side_cam = (
            os.getenv("AHA_NO_SIDE_CAMERA", "").strip() not in ("", "0", "false", "False")
            and not vlm_enabled and args.no_side_camera_window and chosen_cam != 'side')
        if _skip_side_cam:
            side_camera = None
            initial_side_frame = None
            print("Side camera SKIPPED (AHA_NO_SIDE_CAMERA: no VLM / no window)")
        else:
            side_camera = get_or_create_side_camera(scene)
            initial_side_frame = warm_up_camera_sensor(scene, side_camera)
            pos = side_camera.get_position()
            print(
                "Side camera ready "
                f"({SIDE_CAMERA_NAME}) at "
                f"x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}"
            )
            try:
                applied_fov = widen_wrist_camera(scene)
                if applied_fov:
                    print(f"Wrist camera FOV widened to {applied_fov:.0f} deg so wrist_rgb shows the gripper fingers")
            except Exception as e:
                print(f"Could not widen wrist camera FOV: {e}")
            try:
                applied_tilt = tilt_wrist_camera(scene)
                if applied_tilt:
                    print(f"Wrist camera tilted down {applied_tilt:.0f} deg so wrist_rgb shows more of the gripper")
            except Exception as e:
                print(f"Could not tilt wrist camera: {e}")


        side_camera_window = (
            None
            if args.no_side_camera_window
            else LiveCameraView(
                side_camera,
                f"{task_name} - detector side camera",
                initial_frame=initial_side_frame,
            )
        )


        robot = scene.robot
        waypoints = scene.task.get_waypoints()
        n = len(waypoints)
        print(f"Found {n} waypoints")

        monitor = None
        if getattr(args, 'live_detectors', False):
            # Detector targets come exclusively from inspection transforms and
            # optional hook equations. Scene-object roots are cached before
            # failure injection; simulator waypoint geometry is never read here.
            # The robot's CURRENT pose still comes from obs.gripper_pose.
            original_wp_poses = {}
            _wp_chain = None
            try:
                from aha_publish.simulation.waypoint_chain import WaypointChain
                from pyrep.objects.object import Object as _Object
                _chain = WaypointChain.for_task(task_name)
                if _chain is None:
                    raise FileNotFoundError(
                        f"no inspection report for {task_name}")
                _world = {}
                for _name in _chain.root_objects:
                    try:
                        if 'waypoint' in _name.lower():
                            raise ValueError('simulator waypoint reads are prohibited')
                        _world[_name] = list(
                            _Object.get_object(_name).get_pose())
                    except Exception as _exc:
                        print(f"[wp-chain] root {_name!r} unreadable: {_exc}")
                _resolved = _chain.resolve(
                    _world, variation_index=scene._variation_index)
                original_wp_poses = {int(k): list(v)
                                     for k, v in _resolved.items()}
                print(f"[wp-chain] composed {len(_resolved)} waypoint targets "
                      f"from {len(_world)} object poses read once")
                _missing = _chain.unresolved()
                if _missing:
                    print(f"[wp-chain] WARNING: waypoints {_missing} could not "
                          f"be grounded and have NO reference pose")
                _stale = _chain.spec.get("dynamic_waypoints") or []
                if _stale:
                    # init_episode repositions these, so the stored offset does
                    # not describe this episode. Correcting them meant reading
                    # the dummy, which no longer happens -- the spec has to carry
                    # corrected offsets instead.
                    print(f"[wp-chain] WARNING: {task_name} lists "
                          f"dynamic_waypoints {_stale}; their offsets are "
                          f"known-stale and are NOT corrected at run time")
                _wp_chain = _chain
            except Exception as _exc:
                print(f"[wp-chain] FAILED ({type(_exc).__name__}: {_exc}) -- no "
                      f"reference poses, detectors have nothing to compare "
                      f"against for this run")

            def _original_pose_fn(idx):
                # In-memory inspection reference, including any hook equations
                # evaluated for this execution. Never a simulator dummy read.
                return _wp_chain.pose(idx) if _wp_chain is not None else None
            try:
                from aha_publish.running.live_detectors import LiveDetectorMonitor, enabled_detectors_for_stage, stage_context_map_for_stages, target_object_map_for_stages
                # load_bt_stages applies configured primitive assignments and
                # saved selections for the remaining detectors.
                enabled_map = {
                    idx: enabled_detectors_for_stage(stage_for_waypoint(bt_stages, idx))
                    for idx in range(n)
                }
                # The object each stage manipulates, straight from that stage's
                # own conditions. The collision VLM verifier is told it, so
                # contact with THIS object reads as intended manipulation while
                # contact with anything else reads as a collision.
                target_object_map = target_object_map_for_stages(
                    bt_stages, n, stage_lookup=stage_for_waypoint)
                # The same verifier is also told the stage's primitive and its
                # pre/postconditions, so the "what is this step for" context is
                # the BT's predicates instead of the description JSON's prose.
                stage_context_map = stage_context_map_for_stages(
                    bt_stages, n, stage_lookup=stage_for_waypoint)
                if vlm_enabled:
                    # Make sure OpenAI creds are loaded so the detector VLM
                    # confirmation (OpenAI) can run on a detection.
                    try:
                        load_openai_credential_from_local_files()
                    except Exception:
                        pass
                monitor = LiveDetectorMonitor(
                    task_name,
                    original_pose_fn=_original_pose_fn,
                    enabled_map=enabled_map,
                    target_object_map=target_object_map,
                    stage_context_map=stage_context_map,
                    env_wrapper=env_wrapper,
                    vlm_enabled=vlm_enabled,
                    vlm_model=args.vlm_model,
                    n_waypoints=n,
                    failure_waypoint=failure_waypoint if failure_active else None,
                )
                monitor.wp_chain = _wp_chain
                monitor.reference_rules = None
                if _wp_chain is not None and _wp_chain.spec.get('hook_reference_rules'):
                    from aha_publish.simulation.waypoint_reference_rules import ReferenceRuleEvaluator, referenced_object_names
                    _scene_pose_cache = dict(_world)
                    def _scene_reference_pose(name):
                        if 'waypoint' in name.lower():
                            raise ValueError('simulator waypoint reads are prohibited')
                        return _scene_pose_cache[name]
                    # Capture named scene objects BEFORE episode-level failure
                    # setup. Pose equations select among these cached roots.
                    # reference_scene_objects covers what the inspection exported
                    # (SHAPEs inside the boundary root); the equations also select
                    # objects the inspection never listed -- grasp/drop-point
                    # DUMMYs, proximity sensors -- through task attributes, and a
                    # single one of those missing from the cache costs its
                    # waypoints their whole reference.
                    _hook_objects = referenced_object_names(
                        _wp_chain.spec.get('hook_reference_rules') or {}, scene.task)
                    for _name in (list(_wp_chain.spec.get('reference_scene_objects', []))
                                  + _hook_objects):
                        if 'waypoint' in _name.lower():
                            raise ValueError('waypoint cannot be a scene-reference root')
                        if _name not in _scene_pose_cache:
                            try:
                                _scene_pose_cache[_name] = list(_Object.get_object(_name).get_pose())
                            except Exception as exc:
                                print(f"  [references] scene object {_name!r} unavailable: {exc}")
                    monitor.reference_rules = ReferenceRuleEvaluator(
                        _wp_chain, scene.task, _scene_reference_pose)
                monitor.print_banner()
            except Exception as e:
                print(f"Could not start live detectors: {e}")
                monitor = None
        if len(bt_stages) != n:
            print(
                f"WARNING: BT file has {len(bt_stages)} stages, "
                f"but simulator has {n} waypoints."
            )
        print()

        if failure_active:
            env_wrapper.on_env_start(scene.task)

        # Run the episode as a real py_trees behavior tree. The tree reuses the
        # bt_run_move / bt_run_gripper / maybe_run_vlm_* primitives above; see
        # bt_tree.py for the node definitions and tree shape. bt_tree is imported
        # here (not at module top level) so the light-import contract holds: the
        # GUI can import this runner without pulling in
        # py_trees or any simulator dependency.
        import aha_publish.running.bt_tree as bt_tree

        ctx = bt_tree.BTContext(
            runner=sys.modules[__name__],
            args=args,
            scene=scene,
            env_wrapper=env_wrapper,
            robot=robot,
            waypoints=waypoints,
            n=n,
            bt_stages=bt_stages,
            task_name=task_name,
            chosen_cam=chosen_cam,
            side_camera=side_camera,
            side_camera_window=side_camera_window,
            monitor=monitor,
            yaml_descs=yaml_descs,
            vlm_enabled=vlm_enabled,
            failure_active=failure_active,
            failure_waypoint=failure_waypoint,
            responsible_detector=getattr(args, 'responsible_detector', None),
        )

        bt_tree.run_tree(ctx)


        if ctx.failures:
            print("\nStopped: the behavior tree reported a failure (see BT VERDICT above).")
        else:
            print("\nDone! Robot completed all waypoints.")
        if monitor is not None:
            monitor.print_summary()
            monitor.close()
        wait_for_space("Press Space or Enter to close the simulator...")
    finally:
        env_wrapper.shutdown()

DETECTORS = ("orientation", "transition", "slip", "collision")
LIVE_ALL = ",".join(DETECTORS)
# failgen failure `type` -> the detector responsible for catching it.
FAILTYPE_DETECTOR = {ft: det for det, fts in EV.DETECTOR_FAILTYPES.items() for ft in fts}
# Failure types no detector is expected to catch (a failed grasp means the
# object was never held, so there is nothing for the cheap detectors to see).
# Not enumerated, and dropped entirely by --rescore.
EXCLUDED_FAILTYPES = {"grasp"}
RUN_TIMEOUT = 1800
# Max attempts for a case whose injection waypoint gets skipped (NA_unreached):
# the skip is stochastic, so re-run and only finalize NA if all attempts skip.
NA_RETRIES = 5

_write_lock = threading.Lock()

RESULT_FIELDS = ["task", "case", "failure", "waypoint", "ground_truth",
                 "responsible_detector", "detector", "fired", "fired_waypoints",
                 "vlm_ran", "vlm_verdict", "vlm_retracted", "vlm_model",
                 "vlm_calls", "vlm_input_tokens", "vlm_output_tokens",
                 "vlm_reasoning_tokens", "vlm_cost_usd", "vlm_reason", "vlm_error",
                 "prepost_checks", "prepost_fails",
                 "prepost_fail_wps", "outcome", "status", "plot", "elapsed"]

VLM_AUTO_CONFIRM = re.compile(
    r"\[vlm\] auto-confirming (orientation|transition|slip|collision) detection")
VLM_CONFIRM_FAILED = re.compile(r"\[vlm\] confirmation failed:\s*(.*)")

# Pre/post-condition VLM checkpoint lines (only with --vlm-prepost):
#   [vlm] checking preconditions for waypoint 2...            (single kind)
#   [vlm] checking preconditions for waypoint 3 and postconditions for waypoint 2...
#   VLM PRE CHECK: PASS | VLM PRE+POST CHECK: FAIL | ...
# "VLM <kind> SIMULATOR CHECK" lines are sim-backed condition shortcuts printed
# for the same checkpoint: on PASS the real VLM verdict follows (that line is
# the check), on FAIL/UNKNOWN the checkpoint ends there (the simulator line IS
# the check).
PREPOST_CHECKING = re.compile(
    r"\[vlm\] checking (pre|post)conditions for waypoint (\d+)"
    r"(?: and postconditions for waypoint (\d+))?")
PREPOST_VERDICT = re.compile(
    r"VLM ([A-Z+]+(?: SIMULATOR)?) CHECK: (PASS|FAIL|UNKNOWN)")


def _prepost_fields(stdout):
    """Per-episode pre/post-condition VLM stats parsed from run.log: number of
    checkpoints whose verdict came back, number of FAILs, and (for each FAIL)
    the checkpoint's waypoint label from the preceding '[vlm] checking' line."""
    checks, fails, fail_wps = 0, 0, []
    where = ""
    for line in stdout.splitlines():
        m = PREPOST_CHECKING.search(line)
        if m:
            kind, wp, post_wp = m.groups()
            where = (f"pre{wp}+post{post_wp}" if post_wp is not None
                     else f"{kind}{wp}")
            continue
        m = PREPOST_VERDICT.search(line)
        if not m:
            continue
        kind, verdict = m.groups()
        if "SIMULATOR" not in kind or verdict != "PASS":
            checks += 1
        if verdict == "FAIL":
            fails += 1
            fail_wps.append(where or "?")
    return {"prepost_checks": checks, "prepost_fails": fails,
            "prepost_fail_wps": " ".join(fail_wps)}


# --------------------------------------------------------------------------- #
# Case enumeration
# --------------------------------------------------------------------------- #


def enumerate_cases(task):
    """[(case_name, failure_type, waypoint, responsible_detector)] for a task:
    one clean case + one per (failure type, waypoint) in the task's yaml."""
    cases = [("clean", "none", None, "")]
    by_type = EV.config_failtypes_by_wp(task)          # {type: set(wp)} from yaml
    for ftype in sorted(by_type):
        if ftype in EXCLUDED_FAILTYPES:
            continue
        det = FAILTYPE_DETECTOR.get(ftype, "")
        for wp in sorted(by_type[ftype]):
            cases.append((f"{ftype}_wp{wp}", ftype, wp, det))
    return cases


# --------------------------------------------------------------------------- #
# One episode: all detectors live, optional detector VLM
# --------------------------------------------------------------------------- #
def run_episode(task, failure, waypoint, case_dir, *, vlm_confirm='', vlm_model=None, vlm_cameras=None, prepost=False, prepost_predicates=''):
    """Run one BT episode with all detectors live. Returns
    (stdout, elapsed, drift_paths)."""
    os.makedirs(case_dir, exist_ok=True)
    drift = {d: os.path.join(case_dir, f"{d}_drift.csv")
             for d in ("orientation", "transition", "slip", "collision")}
    capture = os.path.join(case_dir, "capture.jsonl")   # replayable input stream
    for p in list(drift.values()) + [capture]:          # start each run clean
        if os.path.exists(p):
            os.remove(p)
    extra = {
        "AHA_LIVE_DETECTORS": LIVE_ALL,                 # every detector live
        # orientation/transition: fire ONLY via C1 arrival. C4 prop is gone from
        # the detectors; this also drops the diverge mode and the instant
        # target-mutation shortcut (mode=target),
        # which otherwise fires at local_step 0 before any approach telemetry
        # and leaves the injection waypoint absent from the trace plot.
        # arrival-threshold floor: 10 deg (0.1745 rad) instead of 0.3 rad, so
        # smaller orientation errors (e.g. rotation injections that only nudge
        # the live angle) can trip C1 arrival.
        "AHA_ORIENTATION_ARRIVAL_THRESHOLD": os.getenv("AHA_ORIENTATION_ARRIVAL_THRESHOLD", "0.1745"),
        # arrival-threshold floor: 0.015 m (was 0.02).
        "AHA_TRANSITION_ARRIVAL_THRESHOLD": os.getenv("AHA_TRANSITION_ARRIVAL_THRESHOLD", "0.015"),
        "AHA_ORIENTATION_DRIFT_LOG": drift["orientation"],
        "AHA_TRANSITION_DRIFT_LOG": drift["transition"],
        "AHA_SLIP_DRIFT_LOG": drift["slip"],
        "AHA_COLLISION_DRIFT_LOG": drift["collision"],
        "AHA_DETECTOR_CAPTURE": capture,                # per-step inputs for replay
    }
    if vlm_confirm:
        EV.apply_detector_vlm_env(
            extra, cameras=vlm_cameras, confirm_detectors=vlm_confirm)
    else:
        extra["AHA_DETECTOR_VLM_AUTO"] = "0"             # NO VLM
    env = EV.base_env(LIVE_ALL, "", extra)              # COPPELIA/DISPLAY setup + overrides

    # --vlm-cameras default (resolves to ""): use EACH detector's OWN built-in
    # camera set (slip = 5 incl. wrist_depth, collision = its own), matching the
    # interactive runner, which never sets this var. base_env unconditionally
    # pins the 4-cam eval default, so clear it here to actually hand control back
    # to the detectors.
    if vlm_confirm and not EV.detector_vlm_cameras(vlm_cameras):
        env.pop("AHA_VLM_CONFIRM_CAMERAS", None)

    # --vlm-checks both is needed by BOTH VLM channels; which one actually runs
    # is steered by --vlm-predicates (pre/post condition checks) and
    # AHA_VLM_CONFIRM_DETECTORS (detector fire confirmation). Without
    # --vlm-prepost the predicates are pinned to __none__ so no pre/post
    # condition call ever happens.
    if prepost:
        vlm_args = ["--vlm-checks", "both"]
        if prepost_predicates:
            vlm_args += ["--vlm-predicates", prepost_predicates]

    elif vlm_confirm:
        vlm_args = ["--vlm-checks", "both", "--vlm-predicates", "__none__"]
    else:
        vlm_args = ["--vlm-checks", "off"]
    cmd = [sys.executable, str(Path(__file__).resolve()), "single", "--task", task, "--failure", failure,
           "--headless", "--mode", "auto", *vlm_args,
           "--no-side-camera-window"]
    if vlm_model:
        cmd += ["--vlm-model", vlm_model]

    if failure != "none" and waypoint is not None:
        cmd += ["--failure-waypoint", str(waypoint)]

    t0 = time.time()
    failure_reason = None
    try:
        p = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, cwd=EV.PROJECT_ROOT,
                           capture_output=True, text=True, timeout=RUN_TIMEOUT)
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        if p.returncode:
            failure_reason = f"exit status {p.returncode}"
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        out += "\n[TIMEOUT]"
        failure_reason = "timeout"
    elapsed = round(time.time() - t0, 1)
    with open(os.path.join(case_dir, "run.log"), "w") as f:
        f.write("$ " + " ".join(cmd) + "\n\n" + out)
    if failure_reason:
        raise RuntimeError(f"BT episode failed ({failure_reason}); see {case_dir}/run.log")
    return out, elapsed, drift


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score(det, info, is_clean, responsible, inj_wp, *,
          reached=True, slip_holding=None):
    """Outcome for one (detector, episode):

      TP/TN/FP/FN     as usual, except FP is now CLEAN-episode-only.
      CF              cross-fire: a non-responsible detector fired during
                      another failure's episode. Often a genuine physical
                      consequence of the injection (e.g. a translation offset
                      ramming the table IS a collision), so it must not
                      pollute FP counts.
      NA_unreached    the episode ended before the injection waypoint started,
                      so the failure was never injected -- nothing to detect.
      NA_no_grasp     slip only: the injection hit while the robot was not
                      holding anything, so no slip could manifest.
    """
    # Score the END-TO-END pipeline (cheap detector + VLM confirmation): a fire
    # the VLM retracted is not a pipeline detection, so use the VLM-confirmed fire
    # set. This turns a retracted fire on a clean episode into TN (not FP) and a
    # wrongly-retracted real failure into FN. Falls back to the raw fires for
    # detectors/episodes with no VLM (retracted fires there don't exist).
    fired = info.get("fired_confirmed", info["fired"])
    fwps = info.get("fired_waypoints_confirmed", info["fired_waypoints"])
    if is_clean:
        return "FP" if fired else "TN"
    if det != responsible:
        return "CF" if fired else "TN"
    # "Can't-manifest" NAs take precedence: if the failure physically could not
    # occur, a fire is spurious, not a true catch, so these stay NA even if the
    # detector fired.
    if det == "slip" and slip_holding is False:
        return "NA_no_grasp"
    # A detector that fired at (or one waypoint after) the injection point
    # DETECTED the failure -> TP, even if the arm then skipped the waypoint's
    # motion (orientation's target-mode catches the corrupted goal at wp start).
    # Checked before the reached/skip branch so a genuine catch is never
    # downgraded to NA_unreached.
    hit = (inj_wp in fwps or (inj_wp + 1) in fwps) if inj_wp is not None else fired
    if hit:
        return "TP"
    if not reached:
        return "NA_unreached"
    return "FN"


def _drift_max_wp(drift):
    """Highest waypoint with any telemetry across the drift logs. The failure
    hook runs when its waypoint STARTS, so telemetry at wp >= inj_wp means the
    injection actually happened."""
    best = -1
    for p in drift.values():
        if not p or not os.path.exists(p):
            continue
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    best = max(best, int(float(r["waypoint"])))
                except (KeyError, TypeError, ValueError):
                    continue
    return best


def _finished_injection_wp(out, wp):
    """True if the arm actually executed the injected waypoint's MOTION to
    completion, i.e. stdout contains '[move] finished waypoint N path ...'.

    When the motion planner skips the injected waypoint (it prints '[move]
    planning path for waypoint N' then jumps straight to the next waypoint with
    no 'executing'/'finished path' line), the responsible orientation/transition
    detector never receives an arrival observation for that waypoint -- it never
    had a chance to judge the failure. Such an episode is scored NA_unreached
    rather than a false miss (FN). This uses the LOGICAL waypoint index (matching
    inj_wp / fired_waypoints), unlike the raw scene-object index in the drift
    logs, which RLBench renumbers per episode."""
    if wp is None:
        return True
    return re.search(rf"finished waypoint {int(wp)} path", out) is not None


def _slip_holding_at_injection(slip_csv, inj_wp):
    """True/False: was the robot holding an object when the slip injection hit
    (start of inj_wp)? Judged from slip telemetry in a window spanning the last
    frames before the waypoint starts and its first frames. None if unknown."""
    if not slip_csv or not os.path.exists(slip_csv):
        return None
    with open(slip_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    first = next((i for i, r in enumerate(rows)
                  if r.get("waypoint") not in ("", None)
                  and int(float(r["waypoint"])) == inj_wp), None)
    if first is None:
        return None
    window = rows[max(0, first - 10):first + 3]

    def _i(r, k):
        try:
            return int(float(r.get(k) or 0))
        except (TypeError, ValueError):
            return 0
    return any(_i(r, "is_holding") or _i(r, "prior_holding_streak")
               for r in window)


def episode_extras(is_clean, responsible, waypoint, out, drift):
    """(reached, slip_holding) for score(), from the episode's
    stdout + drift logs -- everything on disk, so --rescore can reuse it."""
    reached, slip_holding = True, None
    if not is_clean and waypoint is not None:
        if responsible in ("orientation", "transition"):
            # Pose-corruption failures: only a MISS on a waypoint the arm truly
            # executed to arrival is a real FN; a planner-skipped waypoint is NA.
            reached = _finished_injection_wp(out, waypoint)
        else:
            # Slip/collision injections use the telemetry-based reach test.
            reached = _drift_max_wp(drift) >= waypoint
        if responsible == "slip":
            slip_holding = _slip_holding_at_injection(drift["slip"], waypoint)
    return reached, slip_holding


# --------------------------------------------------------------------------- #
# One job = one (task, case) episode -> per-detector result rows
# --------------------------------------------------------------------------- #
def run_job(task, case, out_dir, na_retries=NA_RETRIES, *, vlm_confirm='', vlm_model=None, vlm_cameras=None, prepost=False, prepost_predicates=''):
    case_name, failure, waypoint, responsible = case
    case_dir = os.path.join(out_dir, task, case_name)
    is_clean = failure == "none"
    label = "clean" if is_clean else f"{failure}@wp{waypoint}"

    # A skipped injection waypoint (NA_unreached) is often stochastic -- the
    # motion planner sometimes skips the corrupted waypoint's path. Re-run up to
    # na_retries times to give the failure a chance to actually execute; only
    # finalize NA if every attempt skips it. A genuine catch (TP, incl.
    # orientation target-mode) or a real miss on an executed waypoint (FN) stops
    # retrying immediately -- those are not NA. Each attempt overwrites the case
    # dir, so the kept run.log/drift/raw-capture belong to the final attempt.
    attempt = 0
    while True:
        attempt += 1
        out, elapsed, drift = run_episode(task, failure, waypoint, case_dir, vlm_confirm=vlm_confirm, vlm_model=vlm_model, vlm_cameras=vlm_cameras, prepost=prepost, prepost_predicates=prepost_predicates)
        info = _add_vlm_errors({d: EV.parse_run(out, d) for d in DETECTORS}, out)
        reached, slip_holding = episode_extras(
            is_clean, responsible, waypoint, out, drift)
        if is_clean or attempt >= na_retries:
            break
        resp_outcome = score(responsible, info[responsible], is_clean,
                             responsible, waypoint, reached=reached,
                             slip_holding=slip_holding)
        if resp_outcome != "NA_unreached":
            break
        print(f"[retry] {task}/{case_name}: injection wp {waypoint} skipped "
              f"(NA_unreached), attempt {attempt}/{na_retries} -> re-running",
              flush=True)

    plots = render_plots(case_dir, task, label, drift, info,
                         failure=failure, waypoint=waypoint)

    prepost_fields = _prepost_fields(out)               # per-episode, same on every row
    rows = []
    for det in DETECTORS:
        info_det = info[det]
        rows.append({
            "task": task, "case": case_name, "failure": failure,
            "waypoint": "" if waypoint is None else waypoint,
            "ground_truth": "clean" if is_clean else "fail",
            "responsible_detector": responsible, "detector": det,
            "fired": info_det["fired"],
            "fired_waypoints": " ".join(map(str, info_det["fired_waypoints"])),
            **_vlm_row_fields(info_det),
            **prepost_fields,
            "outcome": score(det, info_det, is_clean, responsible, waypoint,
                             reached=reached, slip_holding=slip_holding),
            "status": info_det["status"], "plot": plots.get(det, ""),
            "elapsed": elapsed,
        })
    return rows


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _r(x):
    return "nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else round(x, 4)


def _vlm_row_fields(info):
    return {
        "vlm_ran": info.get("vlm_ran", ""),
        "vlm_verdict": info.get("vlm_verdict", ""),
        "vlm_retracted": info.get("retracted", ""),
        "vlm_model": info.get("vlm_model", ""),
        "vlm_calls": info.get("vlm_calls", ""),
        "vlm_input_tokens": info.get("vlm_input_tokens", ""),
        "vlm_output_tokens": info.get("vlm_output_tokens", ""),
        "vlm_reasoning_tokens": info.get("vlm_reasoning_tokens", ""),
        "vlm_cost_usd": info.get("vlm_cost_usd", ""),
        "vlm_reason": info.get("vlm_reason", ""),
        "vlm_error": info.get("vlm_error", ""),
    }


def _add_vlm_errors(info, stdout):
    """Attach detector-specific VLM errors parsed from run.log.

    A failed OpenAI call prints no final "[vlm:model] detector=YES/NO" verdict,
    so EV.parse_run correctly reports no verdict but loses the reason. Track the
    detector named by the preceding auto-confirm line and save the failure text.
    """
    pending_detector = None
    errors = {}
    for line in stdout.splitlines():
        m = VLM_AUTO_CONFIRM.search(line)
        if m:
            pending_detector = m.group(1)
            continue
        m = VLM_CONFIRM_FAILED.search(line)
        if m and pending_detector:
            errors[pending_detector] = m.group(1).strip()
            pending_detector = None
    for det, message in errors.items():
        if det in info:
            info[det]["vlm_error"] = message
            info[det]["vlm_ran"] = True
    return info


def rewrite_summaries(out_dir):
    results = os.path.join(out_dir, "results.csv")
    if not os.path.exists(results):
        return
    with open(results, newline="") as f:
        rows = list(csv.DictReader(f))
    per = {d: {k: 0 for k in ("TP", "TN", "FP", "FN", "CF", "NA")} for d in DETECTORS}
    mis = []
    for r in rows:
        d, o = r["detector"], r["outcome"]
        if d in per:
            if o.startswith("NA"):
                per[d]["NA"] += 1
            elif o in per[d]:
                per[d][o] += 1
        if o in ("FP", "FN"):
            mis.append(r)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector", "TP", "TN", "FP", "FN", "cross_fire", "NA",
                    "precision", "recall", "specificity"])
        for d in DETECTORS:
            c = per[d]
            tp, tn, fp, fn = c["TP"], c["TN"], c["FP"], c["FN"]
            prec = tp / (tp + fp) if (tp + fp) else float('nan')
            rec = tp / (tp + fn) if (tp + fn) else float('nan')
            spec = tn / (tn + fp) if (tn + fp) else float('nan')
            w.writerow([d, tp, tn, fp, fn, c["CF"], c["NA"],
                        _r(prec), _r(rec), _r(spec)])
    with open(os.path.join(out_dir, "misclassified.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        for r in mis:
            w.writerow({k: r.get(k, "") for k in RESULT_FIELDS})


def append_results(out_dir, rows):
    results = os.path.join(out_dir, "results.csv")
    with _write_lock:
        exists = os.path.exists(results)
        if exists:
            with open(results, newline="") as f:
                header = next(csv.reader(f), [])
            if header != RESULT_FIELDS:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                backup = f"{results}.schema_mismatch.{stamp}.bak"
                os.replace(results, backup)
                print(f"[run_all] results.csv schema changed; previous file -> "
                      f"{backup}", flush=True)
                exists = False
        with open(results, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            if not exists:
                w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in RESULT_FIELDS})
        rewrite_summaries(out_dir)


# --------------------------------------------------------------------------- #
# Offline re-scoring: rebuild results.csv from run.log + drift csvs on disk
# with the CURRENT scoring rules. No simulation.
# --------------------------------------------------------------------------- #
def _parse_case_name(case):
    """'clean' -> ('none', None); '<ftype>_wp<N>' -> (ftype, N); else None."""
    if case == "clean":
        return "none", None
    head, sep, tail = case.rpartition("_wp")
    if not sep or not tail.isdigit():
        return None
    return head, int(tail)


def rescore(out_dir):
    old_elapsed = {}
    results = os.path.join(out_dir, "results.csv")
    if os.path.exists(results):
        with open(results, newline="") as f:
            for r in csv.DictReader(f):
                old_elapsed[(r["task"], r["case"], r["detector"])] = r.get("elapsed", "")
        os.replace(results, results + ".bak")

    n_cases = 0
    all_rows = []
    for task in sorted(os.listdir(out_dir)):
        tdir = os.path.join(out_dir, task)
        if not os.path.isdir(tdir):
            continue
        for case in sorted(os.listdir(tdir)):
            cdir = os.path.join(tdir, case)
            runlog = os.path.join(cdir, "run.log")
            parsed = _parse_case_name(case)
            if not os.path.isfile(runlog) or parsed is None:
                continue
            failure, waypoint = parsed
            if failure in EXCLUDED_FAILTYPES:
                continue
            is_clean = failure == "none"
            responsible = "" if is_clean else FAILTYPE_DETECTOR.get(failure, "")
            with open(runlog, errors="ignore") as f:
                out = f.read()
            drift = {d: os.path.join(cdir, f"{d}_drift.csv")
                     for d in ("orientation", "transition", "slip", "collision")}
            info = _add_vlm_errors({d: EV.parse_run(out, d) for d in DETECTORS}, out)
            reached, slip_holding = episode_extras(
                is_clean, responsible, waypoint, out, drift)
            prepost_fields = _prepost_fields(out)
            n_cases += 1
            for det in DETECTORS:
                png = os.path.join(cdir, f"{det}.png")
                info_det = info[det]
                all_rows.append({
                    "task": task, "case": case, "failure": failure,
                    "waypoint": "" if waypoint is None else waypoint,
                    "ground_truth": "clean" if is_clean else "fail",
                    "responsible_detector": responsible, "detector": det,
                    "fired": info_det["fired"],
                    "fired_waypoints": " ".join(map(str, info_det["fired_waypoints"])),
                    **_vlm_row_fields(info_det),
                    **prepost_fields,
                    "outcome": score(det, info_det, is_clean, responsible, waypoint,
                                     reached=reached, slip_holding=slip_holding),
                    "status": info_det["status"],
                    "plot": png if os.path.exists(png) else "",
                    "elapsed": old_elapsed.get((task, case, det), ""),
                })

    with open(results, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    rewrite_summaries(out_dir)
    print(f"[rescore] {n_cases} episodes re-scored -> {results} "
          f"(previous file kept as results.csv.bak)")


def replot(out_dir):
    """No simulation: regenerate every detector plot from the run.log + drift
    CSVs already on disk, using the current render_plots. Reuses whatever python
    is running this (must have matplotlib -- use the `aha` conda env)."""
    n = 0
    for task in sorted(os.listdir(out_dir)):
        tdir = os.path.join(out_dir, task)
        if not os.path.isdir(tdir):
            continue
        for case in sorted(os.listdir(tdir)):
            cdir = os.path.join(tdir, case)
            runlog = os.path.join(cdir, "run.log")
            parsed = _parse_case_name(case)
            if not os.path.isfile(runlog) or parsed is None:
                continue
            failure, waypoint = parsed
            if failure in EXCLUDED_FAILTYPES:
                continue
            with open(runlog, errors="ignore") as f:
                out = f.read()
            drift = {d: os.path.join(cdir, f"{d}_drift.csv")
                     for d in ("orientation", "transition", "slip", "collision")}
            info = {d: EV.parse_run(out, d) for d in DETECTORS}
            label = "clean" if failure == "none" else f"{failure}@wp{waypoint}"
            render_plots(cdir, task, label, drift, info,
                         failure=failure, waypoint=waypoint)
            n += 1
            if n % 200 == 0:
                print(f"[replot] {n} episodes done", flush=True)
    print(f"[replot] regenerated plots for {n} episodes")


# --------------------------------------------------------------------------- #
def run_batch(argv=None):
    global RUN_TIMEOUT
    ap = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} batch",
        description='Run selected tasks and failure cases, save detector plots, and summarize results.')
    ap.add_argument("--out", default=str(paths.RUNS_DIR),
                    help="parent folder for per-task run logs, plots, and result CSVs")
    ap.add_argument("--workers", type=int, default=3,
                    help="parallel episodes (default 3; each episode is a full "
                         "CoppeliaSim instance at ~3-5 GB RAM -- with 31 GB and "
                         "no swap, >3 risks an OOM kill)")
    ap.add_argument("--redo", action="store_true",
                    help="rerun every case even if it already has results "
                         "(default: resume -- skip (task, case) pairs already "
                         "in results.csv)")
    ap.add_argument("--rescore", action="store_true",
                    help="no simulation: rebuild results.csv + summaries from "
                         "the run.log / drift csvs already on disk, applying "
                         "the current scoring rules")
    ap.add_argument("--replot", action="store_true",
                    help="no simulation: regenerate every detector plot from "
                         "the drift csvs on disk (needs matplotlib -- run with "
                         "the `aha` conda env python)")
    ap.add_argument("--task", action="append", default=None,
                    help="run only these named task(s); repeatable")
    ap.add_argument("--failures", default="",
                    help="comma-separated failure types to run, e.g. "
                         "slip,collision. Clean baseline episodes are still "
                         "included for specificity scoring. Default: all.")
    ap.add_argument("--waypoints", default="",
                    help="comma-separated injection waypoint indices to run, "
                         "e.g. 2 or 2,3. Filters failure cases only; the clean "
                         "baseline episode is still included. Default: all.")
    ap.add_argument("--no-clean", action="store_true",
                    help="skip the clean baseline episode and run only the "
                         "selected failure cases (no TN/FP specificity rows)")
    ap.add_argument("--start", type=int, default=1,
                    help="1-based index into the task list to start from (default 1)")
    ap.add_argument("--limit", type=int, default=None,
                    help="run at most this many tasks from --start")
    ap.add_argument("--timeout", type=int, default=RUN_TIMEOUT,
                    help=f"per-episode timeout seconds (default {RUN_TIMEOUT})")
    ap.add_argument("--na-retries", type=int, default=NA_RETRIES,
                    help="max attempts for a case whose injection waypoint is "
                         f"skipped before finalizing NA (default {NA_RETRIES})")
    ap.add_argument("--vlm-confirm", default="",
                    help="comma-separated live detectors whose fires should be "
                         "confirmed by the detector VLM, e.g. slip,collision. "
                         "Default blank keeps the historical no-VLM run.")
    ap.add_argument("--vlm-model",
                    default=EV.DEFAULT_DETECTOR_VLM_MODEL,
                    help="optional OpenAI model name passed through to the BT "
                         "runner for detector VLM confirmation")
    ap.add_argument("--vlm-cameras",
                    default=EV.DEFAULT_DETECTOR_VLM_CAMERAS,
                    help="AHA_VLM_CONFIRM_CAMERAS for detector VLM confirmation "
                         "(defaults from config/detector_vlm.json; use 'default' for "
                         "the detector modules' production defaults)")

    ap.add_argument("--vlm-prepost", action="store_true",
                    help="ALSO run the BT pre/post-condition VLM checks at "
                         "every waypoint boundary (--vlm-checks both). Verdicts "
                         "land in the prepost_* columns of results.csv. NOTE: a "
                         "FAIL verdict stops the BT, so waypoints after it "
                         "never run.")
    ap.add_argument("--vlm-prepost-predicates", default="",
                    help="comma-separated predicate names the pre/post VLM "
                         "checks are restricted to (passed through as "
                         "--vlm-predicates), e.g. end_effector_aligned_with. "
                         "Default: check all conditions.")

    args = ap.parse_args(argv)

    RUN_TIMEOUT = args.timeout
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    if args.rescore:
        rescore(out_dir)
        return
    if args.replot:
        replot(out_dir)
        return

    valid_vlm_confirm = {d.strip() for d in args.vlm_confirm.split(",") if d.strip()}
    unknown_vlm_confirm = valid_vlm_confirm - {"slip", "collision"}
    if unknown_vlm_confirm:
        ap.error("--vlm-confirm contains unknown detector(s): "
                 + ",".join(sorted(unknown_vlm_confirm)))
    vlm_confirm = ",".join(d for d in DETECTORS if d in valid_vlm_confirm)

    if not args.vlm_prepost and (args.vlm_prepost_predicates
                                 or ''):
        ap.error("--vlm-prepost-predicates "
                 "require --vlm-prepost")

    tasks = args.task or EV.prepared_tasks()
    if not args.task:
        tasks = tasks[args.start - 1:]
        if args.limit is not None:
            tasks = tasks[:args.limit]

    allowed_failures = {f.strip() for f in args.failures.split(",") if f.strip()}
    if allowed_failures:
        configured_failures = set()
        for task in tasks:
            configured_failures.update(EV.config_failtypes_by_wp(task))
        unknown_failures = allowed_failures - configured_failures - {"none"}
        if unknown_failures:
            ap.error("--failures contains unknown/unconfigured failure type(s): "
                     + ",".join(sorted(unknown_failures)))

    try:
        allowed_waypoints = {int(w) for w in args.waypoints.split(",") if w.strip()}
    except ValueError:
        ap.error("--waypoints must be comma-separated integers")

    def _selected_cases(task):
        cases = enumerate_cases(task)
        if allowed_failures:
            cases = [case for case in cases
                     if case[1] == "none" or case[1] in allowed_failures]
        if allowed_waypoints:
            cases = [case for case in cases
                     if case[1] == "none" or case[2] in allowed_waypoints]
        if args.no_clean:
            cases = [case for case in cases if case[1] != "none"]
        return cases

    # Resume: a (task, case) pair reaches results.csv only after its episode
    # fully finished and parsed, so pairs already there are safe to skip and
    # half-run episodes (killed mid-flight) get rerun automatically. For --redo,
    # start a fresh result CSV rather than appending duplicate rows to the old
    # one; per-case files are still overwritten by run_episode().
    done_pairs = set()
    results_path = os.path.join(out_dir, "results.csv")
    if args.redo and os.path.exists(results_path):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup = f"{results_path}.redo.{stamp}.bak"
        os.replace(results_path, backup)
        print(f"[run_all] --redo: previous results.csv -> {backup}", flush=True)
    if not args.redo and os.path.exists(results_path):
        with open(results_path, newline="") as f:
            for r in csv.DictReader(f):
                done_pairs.add((r["task"], r["case"]))

    cases_by_task = {t: _selected_cases(t) for t in tasks}
    jobs = [(t, c) for t in tasks for c in cases_by_task[t]
            if (t, c[0]) not in done_pairs]
    skipped = sum(len(cases_by_task[t]) for t in tasks) - len(jobs)
    print(f"[run_all] out={out_dir} tasks={len(tasks)} episodes={len(jobs)} "
          f"(skipped {skipped} already-done) workers={args.workers} "
          f"failures={','.join(sorted(allowed_failures)) if allowed_failures else 'all'} "
          f"(all detectors live, "
          f"{'VLM confirm=' + vlm_confirm + ', cameras=' + (EV.detector_vlm_cameras(args.vlm_cameras) or 'per-detector default') + ', model=' + args.vlm_model if vlm_confirm else 'no detector VLM'}, "
          f"{'prepost VLM=' + (args.vlm_prepost_predicates or 'all predicates') if args.vlm_prepost else 'no prepost VLM'})\n",
          flush=True)

    done = 0
    failed_jobs = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_job, t, c, out_dir, args.na_retries, vlm_confirm=vlm_confirm, vlm_model=args.vlm_model, vlm_cameras=args.vlm_cameras, prepost=args.vlm_prepost, prepost_predicates=args.vlm_prepost_predicates): (t, c)
                for t, c in jobs}
        for fut in as_completed(futs):
            t, c = futs[fut]
            done += 1
            try:
                rows = fut.result()
            except Exception as exc:                    # keep the batch alive
                failed_jobs += 1
                print(f"[{done}/{len(jobs)}] {t}/{c[0]} ERROR: {exc}", flush=True)
                continue
            append_results(out_dir, rows)
            fired = [r["detector"] for r in rows if str(r["fired"]) == "True"]
            print(f"[{done}/{len(jobs)}] {t}/{c[0]} "
                  f"fired={','.join(fired) or 'none'} ({rows[0]['elapsed']}s)", flush=True)

    rewrite_summaries(out_dir)
    print(f"\n[run_all] done. results -> {os.path.join(out_dir, 'results.csv')}")
    print(f"[run_all] summary -> {os.path.join(out_dir, 'summary.csv')}")
    print(f"[run_all] misclassified -> {os.path.join(out_dir, 'misclassified.csv')}")

    if failed_jobs:
        raise SystemExit(f"{failed_jobs} BT episode(s) failed; inspect the case run.log files.")


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == 'batch':
        return run_batch(arguments[1:])
    if arguments and arguments[0] == 'single':
        arguments = arguments[1:]
    return run_single(arguments)


if __name__ == '__main__':
    main()
