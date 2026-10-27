'Build A1–A4 checkpoint samples and frame sidecars from recorded BT runs.'


from aha_publish import paths
from aha_publish.running.run_support import parse_vlm_events


import argparse


import csv


import re


import sys


from collections import defaultdict


from pathlib import Path


DEFAULT_BT_ROOT = (paths.RUNS_DIR)


DEFAULT_OUT = (paths.SCORES_DIR / 'detection_samples.csv')


METHODS = ("A1", "A2", "A3", "A4")


DETECTORS = ("collision", "orientation", "slip", "transition")


RESPONSIBLE = {
    "slip": "slip",
    "collision": "collision",
    "transition": "transition",
    "combined_transition": "transition",
    "orientation": "orientation",
    "combined_orientation": "orientation",
    "rotation": "orientation",
    "translation": "transition",
    "grasp": "",
    "wrong_object": "",
    "wrong_sequence": "",
    "wrong_sequence_v2": "",
}


CONFIRMED_DETECTORS = ("collision", "slip")


FAILTYPE_TO_TYPE = {
    "slip": "Slip",
    "grasp": "Grasp",
    "collision": "collision",
    "wrong_sequence_v2": "Wrong Sequence",
    "combined_transition": "Wrongtransition",
    "combined_orientation": "WrongOrientation",
    "wrong_object": "WrongObjectSelection",
    "wrong_sequence": "Wrong Sequence",
    "transition": "Wrongtransition",
    "translation": "Wrongtransition",
    "orientation": "WrongOrientation",
    "rotation": "WrongOrientation",
}


TYPE_REMAP = {
    "ObjectNotFound": "Wrong Sequence",
}


DETECTOR_PRECEDENCE = {
    "collision": 0,
    "slip": 1,
    "transition": 2,
    "orientation": 2,
}


DETECTOR_TO_TYPE = {
    "collision": "collision",
    "slip": "Slip",
    "transition": "Wrongtransition",
    "orientation": "WrongOrientation",
}


DETECTOR_VOCABULARY = frozenset(DETECTOR_TO_TYPE.values())


CONDITION_TAG_TO_TYPE = {
    "ObjectNotFound": "ObjectNotFound",
    "WrongObjectSelection": "WrongObjectSelection",
    "WrongPosition": "Wrongtransition",
    "WrongOrientation": "WrongOrientation",
    "GraspStateFailure": "Grasp",
    "ReleaseStateFailure": "Wrong Sequence",
    "ExecutionSequenceMismatch": "Wrong Sequence",
    "ContactStateFailure": "Wrongtransition",
}


CHECK_HEADER_RE = re.compile(
    r"^(?P<kind>PRE|POST)-CHECK\s+(?:before|after)\s+waypoint\s+(?P<wp>\d+):",
    re.MULTILINE)


GRIP_PREDICATE = "object_in_gripper"


ALIGNMENT_PREDICATE = "end_effector_aligned_with"


SELECTION_PREDICATE = "selected_object"


CHECK_VERDICT_RE = re.compile(
    r"^\s+VLM\s+(?:PRE\+POST|PRE|POST)(?:\s+SIMULATOR)?\s+CHECK:\s+(?P<verdict>\w+)\s*$",
    re.MULTILINE)


CONDITION_TAG_RE = re.compile(
    r"^\s+-\s+(?P<pred>.+?)\s+\[(?P<tag>[A-Za-z_]+)\]\s*$", re.MULTILINE)


NOT_SATISFIED_RE = re.compile(
    r"^\s+-\s+not_satisfied:\s+(?:(?:pre|post)\s+waypoint\s+\d+:\s*)?"
    r"(?P<pred>.+?)\s*$", re.MULTILINE)


DETECTOR_FIRE_RE = re.compile(
    r"\[DETECTOR\]\s+(ORIENTATION|TRANSITION|SLIP|COLLISION)\s+"
    r"(?:failure|detected)\s+at waypoint\s+(\d+)\s+\(step\s+(\d+)\)",
    re.IGNORECASE,
)


DETECTOR_SUMMARY_RE = re.compile(
    r"^\s*(orientation|transition|slip|collision)\s+"
    r"DETECTED\s+\1\s+x(\d+)\s+\(first:\s+step\s+(\d+),\s+waypoint\s+(\d+),",
    re.IGNORECASE,
)


WAYPOINT_OF_RE = re.compile(r"^Waypoint\s+(\d+)\s+of\s+(\d+)\s*$", re.MULTILINE)


GLOBAL_STEP_LOGS = ("slip_trace.csv", "slip_drift.csv", "collision_drift.csv")


LOCAL_STEP_DETECTORS = frozenset({"orientation", "transition"})


def _int(value):
    try:
        if value in ("", None):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def split_case(case: str) -> tuple[str, int | None]:
    """'collision_wp3' -> ('collision', 3); 'wrong_sequence_v2' -> (name, None)."""
    if case == "clean":
        return "clean", None
    match = re.match(r"(.+)_wp(\d+)$", case)
    if match:
        return match.group(1), int(match.group(2))
    return case, None


def add_alarm(alarms: dict, wp: int, frame: int | None, ftype: str,
              rank: int = 99) -> None:
    """Record one alarm at `wp`, keeping every distinct type that fired there.

    The alarm frame is the earliest fire at that waypoint -- when a reactive
    system would actually stop.  Types accumulate instead of overwriting: nothing
    that fired is thrown away, and `alarm_type` orders them by DETECTOR_PRECEDENCE
    so the root cause is named first.
    """
    alarm = alarms.setdefault(wp, {"frame": frame, "types": []})
    if frame is not None and (alarm["frame"] is None or frame < alarm["frame"]):
        alarm["frame"] = frame
    if not any(t["type"] == ftype for t in alarm["types"]):
        alarm["types"].append({"type": ftype, "rank": rank,
                               "frame": frame if frame is not None else 1 << 30})


def alarm_type(alarm: dict) -> str:
    """The alarm's label: every type it named, root cause first.

    Ordered by DETECTOR_PRECEDENCE, then by which fired first.  A collision that
    fires alongside the transition/orientation deviation it caused therefore leads
    the label, and the downstream types stay visible behind it.
    """
    ordered = sorted(alarm["types"], key=lambda t: (t["rank"], t["frame"]))
    return "|".join(t["type"] for t in ordered)


def waypoint_bounds(run_dir: Path, n_wp: int = 0) -> dict[int, tuple[int, int]]:
    """waypoint -> (first_global_step, last_global_step).

    Waypoints the planner skipped have no telemetry; they inherit a zero-length
    span at the previous waypoint's end so a checkpoint alarm there still gets a
    monotone frame.
    """
    bounds: dict[int, list[int]] = {}
    for name in GLOBAL_STEP_LOGS:
        for row in read_csv(run_dir / name):
            wp, step = _int(row.get("waypoint")), _int(row.get("step"))
            if wp is None or step is None:
                continue
            span = bounds.setdefault(wp, [step, step])
            span[0] = min(span[0], step)
            span[1] = max(span[1], step)
    out = {wp: (lo, hi) for wp, (lo, hi) in bounds.items()}
    if not out:
        return out
    last = 0
    for wp in range(max(n_wp, max(out) + 1)):
        if wp in out:
            last = out[wp][1]
        else:
            out[wp] = (last, last)
    return out


def parse_detector_fires(text: str) -> dict[str, dict[int, int]]:
    """detector -> {waypoint: first firing step}."""
    fires: dict[str, dict[int, int]] = {name: {} for name in DETECTORS}

    def add(name, waypoint, step):
        name = name.lower()
        wp, st = _int(waypoint), _int(step)
        if name not in fires or wp is None:
            return
        if wp not in fires[name] or (st is not None and st < fires[name][wp]):
            fires[name][wp] = st if st is not None else fires[name].get(wp)

    for match in DETECTOR_FIRE_RE.finditer(text or ""):
        add(match.group(1), match.group(2), match.group(3))
    for line in (text or "").splitlines():
        match = DETECTOR_SUMMARY_RE.search(line)
        if match:
            add(match.group(1), match.group(4), match.group(3))
    return fires


def condition_stage(row: dict) -> int | None:
    """The waypoint a condition check reports on.

    'pre'     @ w -> checked before w ran, so it reports on w-1 (-1 at w=0);
    'prepost' @ w -> post-conditions of w-1 and pre-conditions of w  -> w-1;
    'post'    @ w -> the final check after w finished                -> w.
    """
    wp = _int(row.get("waypoint"))
    if wp is None:
        return None
    kind = (row.get("event_kind") or "").strip().lower()
    return wp if kind == "post" else wp - 1


def condition_alarm(row: dict) -> bool:
    return (str(row.get("failure_detected") or "").strip().lower() == "true"
            or str(row.get("verdict") or "").strip().upper() in ("FAIL", "FAILED"))


def slip_holding_at_injection(run_dir: Path, inj_wp: int) -> bool | None:
    rows = read_csv(run_dir / "slip_trace.csv") or read_csv(run_dir / "slip_drift.csv")
    if not rows:
        return None
    first = next((i for i, r in enumerate(rows) if _int(r.get("waypoint")) == inj_wp),
                 None)
    if first is None:
        return None
    window = rows[max(0, first - 10):first + 3]
    return any(_int(r.get("is_holding")) or _int(r.get("prior_holding_streak"))
               for r in window)


def episode_na_reason(run_text: str, run_dir: Path, failure: str, inj_wp,
                      responsible: str, fires: dict) -> str:
    """Why this episode cannot be scored, '' when it is scorable.

    Mirrors bt_runner.score(): a failure that physically could
    not manifest is Not-Applicable, not a missed detection.
    """
    if failure == "clean" or inj_wp is None:
        return ""
    if any(wp in (inj_wp, inj_wp + 1) for wp in fires.get(responsible, {})):
        return ""                      # a genuine catch is never downgraded
    if responsible == "slip":
        if "SLIPPING NOW" not in run_text:
            return "NA_no_slip"
        if slip_holding_at_injection(run_dir, inj_wp) is False:
            return "NA_no_grasp"
    if responsible in ("orientation", "transition"):
        if not re.search(rf"finished waypoint {inj_wp} path", run_text):
            return "NA_unreached"
    elif responsible:
        bounds = waypoint_bounds(run_dir)
        if bounds and max(bounds) < inj_wp:
            return "NA_unreached"
    return ""


def _norm_predicate(text: str) -> str:
    """Predicate text as a comparison key: no case, no spaces, no trailing 'OK'."""
    return re.sub(r"\s+", "", re.sub(r"\s+OK$", "", text or "")).lower()


def condition_tags(run_text: str) -> dict[str, str]:
    """Predicate -> the BT's failure tag, from the PRE/POST-CHECK listings.

    The detector hold-conditions carry lowercase tags ([collision], [slip], ...)
    and are excluded: they belong to A2, not to A1.
    """
    tags = {}
    for match in CONDITION_TAG_RE.finditer(run_text or ""):
        tag = match.group("tag")
        if tag in CONDITION_TAG_TO_TYPE:
            tags.setdefault(_norm_predicate(match.group("pred")), tag)
    return tags


def grasp_and_transport_stages(run_text: str) -> tuple[int | None, set[int]]:
    """(grasp stage, transport stages), from where object_in_gripper is required.

    The BT asserts object_in_gripper as a POSTcondition of the stage that closes
    the gripper and as a PREcondition of every stage that carries the object.  So
    the grasp primitive is the stage that must end holding without having started
    holding, and the transport primitives are the ones that must start holding.
    """
    pre: dict[int, set[str]] = defaultdict(set)
    post: dict[int, set[str]] = defaultdict(set)
    headers = list(CHECK_HEADER_RE.finditer(run_text or ""))
    for index, match in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(run_text)
        block = run_text[match.end():end]
        wp = int(match.group("wp"))
        target = pre if match.group("kind") == "PRE" else post
        for line in block.splitlines():
            hit = CONDITION_TAG_RE.match(line)
            if hit:
                target[wp].add(_norm_predicate(hit.group("pred")))
            elif line.strip().startswith("[vlm]") or line.strip().startswith("VLM"):
                break                       # past the listing, into the check itself

    def holds(bucket, wp):
        return any(p.startswith(GRIP_PREDICATE) for p in bucket.get(wp, ()))

    stages = set(pre) | set(post)
    grasp = next((wp for wp in sorted(stages)
                  if holds(post, wp) and not holds(pre, wp)), None)
    transport = {wp for wp in stages if holds(pre, wp)}
    return grasp, transport


def condition_failure_type(violated: list[str], stage: int | None,
                           tags: dict[str, str], grasp_stage: int | None,
                           transport: set[int]) -> str:
    """The type A1 reports for one FAIL check, from the predicates that failed.

    A violated selected_object(X) or on(X, Y) is decisive and maps to
    WrongObjectSelection under the dataset labeling policy.
    A lone object_in_gripper violation is a grasp problem or a
    slip depending on which primitive it lands in: failing to be holding at the
    grasp stage means the object was never picked up, while failing during
    transport means it was picked up and lost.  A lone robot-alignment condition
    is Wrongtransition: the gripper is not at the pose the stage needs, which is
    the transition failure itself.  Anything else -- several conditions failing
    together, or a gripper open/closed condition -- is an execution-order
    problem, so it is labelled a sequence violation.
    """
    if not violated:
        return "unknown"
    # A violated selected_object(X) says the robot went for the wrong object, and
    # it says so outright -- so it outranks the several-conditions-failed rule
    # below.  What fails alongside it is downstream of that one choice: the arm
    # aligns to the wrong object, then fails to hold the right one.  Same
    # root-cause precedence the detectors get from DETECTOR_PRECEDENCE.
    if any(p.startswith((SELECTION_PREDICATE, "on(")) for p in violated):
        return "WrongObjectSelection"
    only_grip = (len(violated) == 1 and violated[0].startswith(GRIP_PREDICATE))
    if only_grip:
        if stage is not None and stage in transport and stage != grasp_stage:
            return "Slip"
        return "Grasp"
    if len(violated) > 1:
        return "Wrong Sequence"
    predicate = violated[0]
    tag = tags.get(predicate)
    if tag == "ExecutionSequenceMismatch":
        return "Wrong Sequence"
    # A lone end_effector_aligned_with violation says the gripper is not where the
    # stage needs it to be, which is the transition failure itself rather than a
    # symptom of a mis-ordered execution -- the BT tags it WrongPosition for the
    # same reason.  Spelled out here so it holds even when run.log printed no tag.
    if predicate.startswith(ALIGNMENT_PREDICATE):
        return "Wrongtransition"
    return CONDITION_TAG_TO_TYPE.get(tag, "unknown")


def failed_check_predicates(run_text: str) -> list[list[str]]:
    """Every FAIL check's violated predicates, in the order run.log printed them."""
    checks = list(CHECK_VERDICT_RE.finditer(run_text or ""))
    out = []
    for index, match in enumerate(checks):
        if match.group("verdict").strip().upper() not in ("FAIL", "FAILED"):
            continue
        end = checks[index + 1].start() if index + 1 < len(checks) else len(run_text)
        block = run_text[match.end():end]
        out.append([_norm_predicate(h.group("pred"))
                    for h in NOT_SATISFIED_RE.finditer(block)])
    return out


def load_episode(task: str, case_dir: Path) -> dict | None:
    if (task, case_dir.name) in EXCLUDED_EPISODES:
        return None
    run_log = case_dir / "run.log"
    if not run_log.exists():
        return None
    case = case_dir.name
    failure, inj_wp = split_case(case)
    responsible = RESPONSIBLE.get(failure, "")

    run_text = run_log.read_text(errors="ignore")
    stdout = case_dir / "batch_stdout.log"
    if stdout.exists():
        run_text += "\n" + stdout.read_text(errors="ignore")

    n_wp = max((int(b) + 1 for _, b in WAYPOINT_OF_RE.findall(run_text)), default=0)
    bounds = waypoint_bounds(case_dir, n_wp)
    if not n_wp:
        n_wp = (max(bounds) + 1) if bounds else 0
    if not n_wp:
        return None

    events = parse_vlm_events(run_text)
    fires = parse_detector_fires(run_text)

    inj_frame = next((_int(r.get("injection_frame")) for r in events
                      if _int(r.get("injection_frame")) is not None), None)
    if inj_frame is None and inj_wp is not None and inj_wp in bounds:
        inj_frame = bounds[inj_wp][0]
    if inj_frame is None:
        inj_frame = 0

    # ---- A1: condition-VLM alarms ---------------------------------------- #
    # Pair checkpoint verdicts with the violated predicates in the normal
    # run log, preserving their chronological order.  A1 therefore only ever names a failure its own pre/post conditions
    # can express -- no detector, no episode-level diagnosis.
    tags = condition_tags(run_text)
    grasp_stage, transport = grasp_and_transport_stages(run_text)
    fail_predicates = failed_check_predicates(run_text)
    a1: dict[int, dict] = {}
    seen_fail = 0
    no_grasp_predicates: set[str] = set()
    for row in events:
        if row.get("channel") != "condition":
            continue
        if not condition_alarm(row):
            # A passing holding checkpoint is evidence that the grasp recovered.
            wp = _int(row.get("waypoint"))
            kind = (row.get("event_kind") or "").strip().lower()
            if (str(row.get("verdict") or "").strip().upper() == "PASS"
                    and ((kind in ("pre", "prepost") and wp in transport)
                         or (kind == "post" and wp == grasp_stage))):
                no_grasp_predicates.clear()
            continue
        violated = (fail_predicates[seen_fail]
                    if seen_fail < len(fail_predicates) else [])
        seen_fail += 1
        stage = condition_stage(row)
        if stage is None or not 0 <= stage < n_wp:
            continue
        ftype = condition_failure_type(violated, stage, tags, grasp_stage, transport)
        holding_failures = {p for p in violated if p.startswith(GRIP_PREDICATE)}
        if stage == grasp_stage:
            no_grasp_predicates.update(holding_failures)
        if ftype == "Slip" and holding_failures & no_grasp_predicates:
            # The same object was never acquired: a later failed holding check
            # is still No grasp (CSV: Grasp), not Grasp loss (CSV: Slip).
            ftype = "Grasp"
        # A pre-check's STATE predicates report on the waypoint that just ended --
        # 'pre waypoint 4: object_in_gripper' means the object was lost during
        # waypoint 3 -- which is what condition_stage() encodes.  selected_object
        # is not a state predicate: it names the TARGET of the waypoint the check
        # gates, so 'pre waypoint 1: selected_object(chicken)' says waypoint 1 is
        # aimed at the wrong object, and that is evidence about waypoint 1.  The
        # predicate that decides the type therefore also decides the attribution;
        # every other predicate keeps the just-ended-waypoint rule.
        kind = (row.get("event_kind") or "").strip().lower()
        if (any(p.startswith(SELECTION_PREDICATE) for p in violated)
                and kind in ("pre", "prepost")
                and stage + 1 < n_wp):
            stage += 1
        frame = _int(row.get("detection_frame"))
        if frame is None:
            frame = bounds.get(stage, (None, None))[1]
        add_alarm(a1, stage, frame, ftype)

    # ---- A2 / A3: detector alarms ---------------------------------------- #
    confirmed_yes: dict[str, dict[int, int]] = defaultdict(dict)
    for row in events:
        if row.get("channel") != "detector":
            continue
        name = (row.get("event_kind") or "").strip().lower()
        wp = _int(row.get("waypoint"))
        if name in DETECTORS and wp is not None \
                and str(row.get("verdict") or "").strip().upper() == "YES":
            confirmed_yes[name].setdefault(wp, _int(row.get("detection_frame")))

    def fire_frame(name: str, wp: int, step: int | None) -> int | None:
        if step is None:
            return bounds.get(wp, (None, None))[1]
        if name in LOCAL_STEP_DETECTORS:
            start = bounds.get(wp, (None, None))[0]
            return None if start is None else start + step
        return step

    a2: dict[int, dict] = {}
    a3: dict[int, dict] = {}
    for name, per_wp in fires.items():
        # Dataset policy: wrong-object episodes can deviate from the intended
        # pose simply by targeting another object. Do not label those detector
        # fires as independent translation/orientation failures, or use their
        # frames for detection latency. Keep condition-VLM predictions intact.
        if failure == "wrong_object" and name in LOCAL_STEP_DETECTORS:
            continue
        ftype = DETECTOR_TO_TYPE.get(name, name)
        for wp, step in per_wp.items():
            if not 0 <= wp < n_wp:
                continue
            frame = fire_frame(name, wp, step)
            rank = DETECTOR_PRECEDENCE.get(name, 99)
            add_alarm(a2, wp, frame, ftype, rank)
            if name in CONFIRMED_DETECTORS:
                if wp in confirmed_yes.get(name, {}):
                    yes = confirmed_yes[name][wp]
                    add_alarm(a3, wp, yes if yes is not None else frame, ftype, rank)
            else:
                add_alarm(a3, wp, frame, ftype, rank)

    # A4 fuses the two channels with A3 taking priority: where a detector fired,
    # its type IS A4's verdict and the condition VLM's reading of that same
    # checkpoint is dropped.  The sensor measures the fault directly, while a
    # condition violated at that moment is usually its downstream symptom -- the
    # object leaves the gripper BECAUSE the arm hit something -- so keeping both
    # names would credit A4 for a type it only reached through the effect.
    #
    # That reasoning only holds for types a detector could have produced.  When
    # A1 names something OUTSIDE the detector vocabulary -- wrong object, grasp,
    # sequence -- the detector fire is not a competing diagnosis of the same
    # event but the deviation that failure caused, so A1's name is the only
    # account of the root cause and survives alongside it (behind it, since it
    # keeps A1's rank).  Priority resolves disagreements; it does not silence the
    # one channel that can express the failure.
    #
    # The alarm FRAME still comes from whichever channel fired first: a reactive
    # system stops at the first alarm no matter which channel raised it, so
    # priority decides the label, not the latency.
    a4: dict[int, dict] = {}
    for wp, alarm in a3.items():
        for entry in alarm["types"]:
            add_alarm(a4, wp, alarm["frame"], entry["type"], entry["rank"])
    for wp, alarm in a1.items():
        if wp in a4:
            frame = alarm["frame"]
            if frame is not None and (a4[wp]["frame"] is None
                                      or frame < a4[wp]["frame"]):
                a4[wp]["frame"] = frame
            for entry in alarm["types"]:
                if entry["type"] not in DETECTOR_VOCABULARY \
                        and entry["type"] != "unknown":
                    add_alarm(a4, wp, alarm["frame"], entry["type"],
                              entry["rank"])
            continue
        for entry in alarm["types"]:
            add_alarm(a4, wp, alarm["frame"], entry["type"], entry["rank"])

    return {
        "task": task,
        "case": case,
        "failure": failure,
        "inj_wp": inj_wp,
        "inj_frame": inj_frame,
        "n_wp": n_wp,
        "na": episode_na_reason(run_text, case_dir, failure, inj_wp,
                                responsible, fires),
        "alarms": {"A1": a1, "A2": a2, "A3": a3, "A4": a4},
    }


def discover_tasks(bt_root: Path) -> list[str]:
    """Every task under `bt_root`, i.e. every subdirectory holding an episode.

    The run root also carries loose bookkeeping files (plan.csv, summary.csv,
    metrics files) and can hold half-started task folders, so a directory
    only counts as a task once at least one of its case folders has a run.log.
    """
    if not bt_root.is_dir():
        return []
    return sorted(p.name for p in bt_root.iterdir()
                  if p.is_dir() and any(c.is_dir() and (c / "run.log").exists()
                                        for c in p.iterdir()))


def load_all(bt_root: Path, tasks):
    episodes = []
    for task in tasks:
        task_dir = bt_root / task
        if not task_dir.is_dir():
            print(f"!! no run directory for task {task} under {bt_root}",
                  file=sys.stderr)
            continue
        for case_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            episode = load_episode(task, case_dir)
            if episode:
                episodes.append(episode)
    return episodes


def gt_label(episode: dict, wp: int) -> str:
    """'success', or 'failure,<Type>' once the injected failure is present.

    Mirrors assess_reactive_gpt.ground_truth(): stage k of case <failtype>_wpN is
    ground-truth failure iff k >= N, and a case with no _wpN (wrong_sequence_v2)
    is corrupted from the first waypoint.
    """
    if episode["failure"] == "clean":
        return "success"
    inj = episode["inj_wp"]
    if wp < (0 if inj is None else inj):
        return "success"
    return f"failure,{FAILTYPE_TO_TYPE.get(episode['failure'], '?')}"


def remap_types(label: str) -> str:
    """Apply TYPE_REMAP to one 'failure,<Type>[|<Type>...]' cell.

    'success', 'uncertain' and the empty cell pass through untouched.
    """
    if not label.startswith("failure,"):
        return label
    kept: list[str] = []
    for name in label[8:].split("|"):
        target = TYPE_REMAP.get(name, name)
        if target and target not in kept:
            kept.append(target)
    return f"failure,{'|'.join(kept)}" if kept else "success"


def method_label(episode: dict, method: str, wp: int) -> str:
    """One method's verdict at one checkpoint, in the `gt` vocabulary."""
    alarms = episode["alarms"][method]
    if alarms is None:
        return ""                                   # no result for this episode
    alarm = alarms.get(wp)
    return f"failure,{alarm_type(alarm)}" if alarm else "success"


EXCLUDED_EPISODES = frozenset({
    ("push_button", "wrong_object_wp1"),
    ("put_item_in_drawer", "wrong_object_wp1"),
    ("slide_block_to_target", "grasp_wp0"),
})


EXCLUDED_CHECKPOINTS = {
    ("pick_up_cup", "wrong_object_wp1"): frozenset({1, 2}),
}


def checkpoint_excluded(episode: dict, wp: int) -> bool:
    key = (episode["task"], episode["case"])
    return key in EXCLUDED_EPISODES or wp in EXCLUDED_CHECKPOINTS.get(key, ())


def write_samples(episodes, out: Path) -> int:
    rows = 0
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "failure_condition", "waypoint", "gt", *METHODS])
        for episode in episodes:
            for wp in range(episode["n_wp"]):
                if checkpoint_excluded(episode, wp):
                    continue
                writer.writerow(
                    [episode["task"], episode["case"], wp,
                     remap_types(gt_label(episode, wp))]
                    + [remap_types(method_label(episode, m, wp))
                       for m in METHODS])
                rows += 1
    return rows


def write_frames(episodes, out: Path) -> None:
    """Injection frame + each method's first crediting alarm frame, per episode.

    'Crediting' means at or after the injection waypoint -- the same alarm the
    scorer counts as the detection, so the difference is the latency.
    """
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "failure_condition", "injection_frame"]
                        + [f"{m}_alarm_frame" for m in METHODS])
        for episode in episodes:
            if ((episode["task"], episode["case"]) in EXCLUDED_EPISODES
                    or episode["failure"] == "clean"):
                continue
            floor = 0 if episode["inj_wp"] is None else episode["inj_wp"]
            cells = []
            for method in METHODS:
                alarms = episode["alarms"][method]
                frames = [] if not alarms else [
                    a["frame"] for wp, a in alarms.items()
                    if wp >= floor and not checkpoint_excluded(episode, wp)
                    and a["frame"] is not None
                    # an alarm whose every type is dropped by TYPE_REMAP is
                    # scored as 'success', so it sets no latency frame either
                    and remap_types(f"failure,{alarm_type(a)}") != "success"]
                cells.append(min(frames) if frames else "")
            writer.writerow([episode["task"], episode["case"],
                             episode["inj_frame"], *cells])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bt-root", type=Path, default=DEFAULT_BT_ROOT,
                        help="root of the A1..A4 BT runs")

    parser.add_argument("--tasks", nargs="+", default=None,
                        help="tasks to include; default: every task directory "
                             "found under --bt-root")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)


    args = parser.parse_args()

    tasks = args.tasks
    if tasks is None:
        tasks = discover_tasks(args.bt_root)
        if not tasks:
            sys.exit(f"no task directories under {args.bt_root}")
        print(f"   {len(tasks)} task(s) under {args.bt_root}: {', '.join(tasks)}")

    episodes = load_all(args.bt_root, tasks)
    if not episodes:
        sys.exit(f"no episodes found under {args.bt_root}")

    dropped = [e for e in episodes if e["na"]]
    episodes = [e for e in episodes if not e["na"]]
    for episode in dropped:
        print(f"   dropped {episode['task']}/{episode['case']}: {episode['na']}")

    missing = defaultdict(list)
    for episode in episodes:
        for method in METHODS:
            if episode["alarms"][method] is None:
                missing[method].append(f"{episode['task']}/{episode['case']}")
    for method, cases in missing.items():
        print(f"   {method}: no result for {len(cases)} episode(s) "
              f"-> empty cells ({', '.join(cases[:4])}"
              f"{', ...' if len(cases) > 4 else ''})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = write_samples(episodes, args.out)
    frames = args.out.with_name(args.out.stem + "_frames.csv")
    write_frames(episodes, frames)

    clean = sum(1 for e in episodes if e["failure"] == "clean")
    print(f"\n{len(episodes)} episodes ({clean} clean), {rows} checkpoints")
    print(f"wrote {args.out}")
    print(f"wrote {frames}  (latency sidecar)")


if __name__ == "__main__":
    main()
