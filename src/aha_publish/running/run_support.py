"""Batch runtime settings, task discovery, simulator environment, and log parsing."""

from aha_publish import paths

import json
import os

import re

PROJECT_ROOT = str(paths.PROJECT_ROOT)

_VLM_DEFAULTS = json.loads((paths.PROJECT_ROOT / 'config/detector_vlm.json').read_text())
DEFAULT_DETECTOR_VLM_CAMERAS = ','.join(_VLM_DEFAULTS['cameras'])
DEFAULT_DETECTOR_VLM_MODEL = os.getenv(
    'OPENAI_DETECTOR_VLM_MODEL', os.getenv('OPENAI_MODEL', _VLM_DEFAULTS['model']))

BT_DIR = str(paths.BT_DIR)

CFG_DIR = str(paths.FAILGEN_ROOT / 'failgen/configs')

COPPELIA = str(paths.COPPELIASIM_ROOT)

ALL_DETECTORS = ("orientation", "transition", "slip", "collision")

DETECTOR_FAILTYPES = {
    "transition": ("translation_x", "translation_y", "translation_z", "translation",
                   "combined_transition"),
    "orientation": ("rotation_x", "rotation_y", "rotation_z", "no_rotation",
                    "combined_orientation"),
    "slip": ("slip", "grasp"),
    "collision": ("collision",),
}

SUMMARY_DETECTED = {n: re.compile(rf"{n}\s+DETECTED {n} x(\d+)") for n in ALL_DETECTORS}

SUMMARY_OK = {n: re.compile(rf"{n}\s+ok — no {n}") for n in ALL_DETECTORS}

SUMMARY_DISABLED = {n: re.compile(rf"{n}\s+DISABLED") for n in ALL_DETECTORS}

SUMMARY_SKIPPED = {n: re.compile(rf"{n}\s+skipped") for n in ALL_DETECTORS}

FIRE_LINE = re.compile(
    r"\[DETECTOR\] (ORIENTATION|TRANSITION|SLIP|COLLISION) "
    r"(?:failure|detected) at waypoint (\d+)")

VLM_VERDICT = re.compile(
    r"\[vlm:[^\]]*\]\s+(orientation|transition|slip|collision)=(YES|NO|UNKNOWN)")

VLM_MODEL = re.compile(
    r"\[vlm:([^\]]+)\]\s+(orientation|transition|slip|collision)=")

VLM_REASON = re.compile(
    r"\[vlm:[^\]]*\]\s+(orientation|transition|slip|collision)="
    r"(?:YES|NO|UNKNOWN)\s+(.*?)\s+cameras=")

VLM_TOKENS = re.compile(
    r"\[vlm:(orientation|transition|slip|collision)\]\s+tokens:\s+"
    r"(\d+)\s+in\s+\+\s+(\d+)\s+out"
    r"(?:\s+\(reasoning:\s+(\d+)\))?")

RETRACT_LINE = re.compile(r"\[detector:(\w+)\] detection RETRACTED")


OPENAI_PRICE_PER_1M = {
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.5": (5.00, 30.00),
}


def prepared_tasks():
    return sorted(f[:-len(".bt_conditions.json")]
                  for f in os.listdir(BT_DIR) if f.endswith(".bt_conditions.json"))


def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def config_failtypes_by_wp(task):
    """{failure_type: set(waypoints)} from the task's failgen config."""
    path = os.path.join(CFG_DIR, f"{task}.yaml")
    if not os.path.exists(path):
        return {}
    cfg = load_yaml(path)
    by_type = {}
    for f in cfg.get("failures", []) or []:
        ty = f.get("type")
        if not ty:
            continue
        for wp in f.get("waypoints", []) or []:
            by_type.setdefault(ty, set()).add(int(wp))
    return by_type


def detector_vlm_cameras(value=None):
    value = (value or '').strip()
    if value.lower() == 'default':
        return ''
    return value or DEFAULT_DETECTOR_VLM_CAMERAS


def apply_detector_vlm_env(env, *, cameras=None, confirm_detectors=None):
    """Pass confirmation settings to detector subprocesses."""
    env['AHA_DETECTOR_VLM_AUTO'] = '1'
    if confirm_detectors is not None:
        env['AHA_VLM_CONFIRM_DETECTORS'] = confirm_detectors
    camera_names = detector_vlm_cameras(cameras)
    if camera_names:
        env['AHA_VLM_CONFIRM_CAMERAS'] = camera_names
    return env


def base_env(detector, cameras, extra=None):
    env = dict(os.environ)
    env["COPPELIASIM_ROOT"] = COPPELIA
    env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + ":" + COPPELIA
    env["QT_QPA_PLATFORM_PLUGIN_PATH"] = COPPELIA
    env.pop("QT_QPA_PLATFORM", None)            # need a real GL context
    env["DISPLAY"] = os.environ.get("DISPLAY", ":1")
    env["AHA_LIVE_DETECTORS"] = detector        # isolate one detector
    env["AHA_FAIL_DEBUG"] = "1"
    apply_detector_vlm_env(
        env, cameras=cameras, confirm_detectors=None,
        )
    if extra:
        env.update(extra)
    return env


def _price_per_1m(model):
    input_override = os.getenv("AHA_VLM_INPUT_PRICE_PER_1M", "").strip()
    output_override = os.getenv("AHA_VLM_OUTPUT_PRICE_PER_1M", "").strip()
    if input_override and output_override:
        try:
            return float(input_override), float(output_override)
        except ValueError:
            pass

    key = (model or "").strip().lower()
    return OPENAI_PRICE_PER_1M.get(key, (0.0, 0.0))


def parse_run(stdout, detector):
    """Parse fire waypoints, the VLM verdict, retraction and the summary status."""
    fired_wps = sorted(int(wp) for name, wp in FIRE_LINE.findall(stdout)
                       if name.lower() == detector)
    verdicts = [v for n, v in VLM_VERDICT.findall(stdout) if n == detector]
    vlm_verdict = verdicts[-1] if verdicts else ""    # last verdict in the run
    models = [m.strip() for m, n in VLM_MODEL.findall(stdout) if n == detector]
    vlm_model = models[-1] if models else ""
    reasons = [r.strip() for n, r in VLM_REASON.findall(stdout) if n == detector]
    vlm_reason = reasons[-1] if reasons else ""       # reason for the last verdict
    token_rows = [
        (int(input_tokens), int(output_tokens), int(reasoning_tokens or 0))
        for n, input_tokens, output_tokens, reasoning_tokens
        in VLM_TOKENS.findall(stdout)
        if n == detector
    ]
    vlm_calls = len(token_rows)
    vlm_input_tokens = sum(row[0] for row in token_rows)
    vlm_output_tokens = sum(row[1] for row in token_rows)
    vlm_reasoning_tokens = sum(row[2] for row in token_rows)
    input_price, output_price = _price_per_1m(vlm_model)
    vlm_cost_usd = (
        (vlm_input_tokens * input_price + vlm_output_tokens * output_price)
        / 1_000_000.0
        if vlm_model and token_rows and (input_price or output_price)
        else 0.0
    )
    retracted = any(n == detector for n in RETRACT_LINE.findall(stdout))
    # VLM-confirmed fires: a "detection RETRACTED" line overturns the most recent
    # unretracted fire of the same detector (retraction is printed right after the
    # fire+verdict it refers to). Walking fire/retract events in document order
    # yields the fires that SURVIVED VLM confirmation, so downstream scoring can
    # judge the end-to-end pipeline (detector + VLM) rather than the raw detector.
    events = [(m.start(), "fire", int(m.group(2)))
              for m in FIRE_LINE.finditer(stdout) if m.group(1).lower() == detector]
    events += [(m.start(), "retract", None)
               for m in RETRACT_LINE.finditer(stdout) if m.group(1) == detector]
    events.sort()
    surviving = []
    for _, kind, wp in events:
        if kind == "fire":
            surviving.append(wp)
        elif surviving:                       # retract kills the latest live fire
            surviving.pop()
    fired_wps_confirmed = sorted(set(surviving))

    status = "ran"
    if SUMMARY_DETECTED[detector].search(stdout):
        status = "ran"
    elif SUMMARY_OK[detector].search(stdout):
        status = "ran"
    elif SUMMARY_DISABLED[detector].search(stdout):
        status = "disabled"
    elif SUMMARY_SKIPPED[detector].search(stdout):
        status = "skipped"
    elif not stdout.strip():
        status = "no_output"
    return {"fired_waypoints": fired_wps, "fired": bool(fired_wps),
            "fired_waypoints_confirmed": fired_wps_confirmed,
            "fired_confirmed": bool(fired_wps_confirmed),
            "vlm_verdict": vlm_verdict, "vlm_ran": bool(verdicts),
            "vlm_model": vlm_model, "vlm_calls": vlm_calls,
            "vlm_input_tokens": vlm_input_tokens,
            "vlm_output_tokens": vlm_output_tokens,
            "vlm_reasoning_tokens": vlm_reasoning_tokens,
            "vlm_cost_usd": vlm_cost_usd,
            "vlm_reason": vlm_reason,
            "retracted": retracted, "status": status,
            "stdout_tail": stdout[-1500:]}




def responsible_detector_for(failure):
    """Use the same injection-type ownership as batch case enumeration."""
    return next((name for name, failures in DETECTOR_FAILTYPES.items()
                 if failure in failures), None)


def parse_vlm_events(stdout):
    """Checkpoint verdicts and detector confirmations from the standard run log."""
    checking = re.compile(r"\[vlm\] checking (pre|post)conditions for waypoint (\d+)"
                          r"(?: and postconditions for waypoint (\d+))?")
    verdict = re.compile(r"VLM ([A-Z+]+(?: SIMULATOR)?) CHECK: (PASS|FAIL|UNKNOWN)")
    confirm = re.compile(r"\[vlm:[^\]]+\]\s+(slip|collision)=(YES|NO|UNKNOWN)\b")
    events, fires = [], {}
    checkpoint = None
    for line in (stdout or "").splitlines():
        match = checking.search(line)
        if match:
            kind, waypoint, previous = match.groups()
            checkpoint = ('prepost' if previous is not None else kind, int(waypoint))
            continue
        match = FIRE_LINE.search(line)
        if match:
            name, waypoint = match.groups()
            fires[name.lower()] = int(waypoint)
        match = verdict.search(line)
        if match and checkpoint is not None:
            kind, value = match.groups()
            if 'SIMULATOR' not in kind or value != 'PASS':
                events.append({'channel': 'condition', 'event_kind': checkpoint[0],
                               'waypoint': checkpoint[1], 'verdict': value})
        match = confirm.search(line)
        if match:
            name, value = match.groups()
            if name in fires:
                events.append({'channel': 'detector', 'event_kind': name,
                               'waypoint': fires[name], 'verdict': value})
    return events
