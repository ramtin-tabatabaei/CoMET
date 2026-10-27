'Score A1–A4 checkpoint samples: detection metrics, failure types, and latency.'


from aha_publish import paths


import argparse
import csv


import re


import sys


from collections import Counter, defaultdict


from pathlib import Path


DEFAULT_SAMPLES = (paths.SCORES_DIR / 'detection_samples.csv')


DEFAULT_OUT_DIR = (paths.SCORES_DIR)


METHODS = ("A1", "A2", "A3", "A4")


METHOD_LABELS = {
    "A1": "A1: Condition VLM",
    "A2": "A2: Detectors",
    "A3": "A3: Det.+Conf.",
    "A4": "A4: Full",
}


FAILURE_ROWS = [
    ("Slip", "Grasp loss"),
    ("collision", "Collision"),
    ("Wrongtransition", "Transition dev."),
    ("WrongOrientation", "Orientation dev."),
    ("WrongObjectSelection", "Wrong object"),
    ("Grasp", "No grasp"),
    ("Wrong Sequence", "Sequence violation"),
]


BLIND = {
    "A1": {"collision"},
    "A2": {"Grasp", "WrongObjectSelection", "Wrong Sequence"},
    "A3": {"Grasp", "WrongObjectSelection", "Wrong Sequence"},
}


def parse_label(value: str) -> tuple[str, str | None] | None:
    """'failure,collision' -> ('failure', 'collision'); 'success' -> ('success', None).

    None (not a tuple) means the cell is empty: no result for this checkpoint.
    Unknown predictions count as success; in mixed labels only known types remain.
    """
    value = (value or "").strip()
    if not value:
        return None
    verdict, _, name = value.partition(",")
    verdict = verdict.strip().lower()
    if verdict == "unknown":
        return "success", None
    if verdict == "failure":
        names = [part.strip() for part in name.split("|")
                 if part.strip().lower() != "unknown"]
        if not names:
            return "success", None
        return "failure", ("|".join(names).strip() or "?")
    return verdict, None


def _int(value):
    try:
        if value in ("", None):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_samples(path: Path) -> list[dict]:
    """One dict per episode: key, failure type, and the per-waypoint rows."""
    if not path.exists():
        sys.exit(f"no sample table at {path}; run make_detection_samples.py first")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        sys.exit(f"{path} is empty")
    missing = [c for c in ("task", "failure_condition", "waypoint", "gt", *METHODS)
               if c not in rows[0]]
    if missing:
        sys.exit(f"{path} is missing column(s): {', '.join(missing)}")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["failure_condition"])].append(row)

    episodes = []
    for (task, case), case_rows in grouped.items():
        case_rows.sort(key=lambda r: _int(r["waypoint"]) or 0)
        gt = [parse_label(r["gt"]) or ("success", None) for r in case_rows]
        # The episode's failure type is whatever its positive checkpoints name;
        # an episode with none is a clean run.
        failure = next((t for v, t in gt if v == "failure"), None)
        preds = {}
        for method in METHODS:
            parsed = [parse_label(r[method]) for r in case_rows]
            # A method with no result for the episode leaves every cell empty.
            preds[method] = None if all(p is None for p in parsed) else [
                p or ("success", None) for p in parsed]
        episodes.append({
            "task": task,
            "case": case,
            "failure": failure,
            "waypoints": [_int(r["waypoint"]) for r in case_rows],
            "gt": gt,
            "preds": preds,
        })
    return episodes


NO_INJECTION_WAYPOINT = re.compile(r"_wp\d+$")


def apply_window(episodes: list[dict], window: str) -> list[dict]:
    """Truncate failure episodes at their injection waypoint when asked."""
    if window != "to-injection":
        return episodes
    out = []
    for episode in episodes:
        gts = [v == "failure" for v, _ in episode["gt"]]
        if not any(gts):
            out.append(episode)                     # clean run: nothing to cut
            continue
        if not NO_INJECTION_WAYPOINT.search(episode["case"]):
            out.append(episode)                     # corrupted throughout
            continue
        keep = gts.index(True) + 1                  # 0..inj inclusive
        trimmed = dict(episode)
        trimmed["gt"] = episode["gt"][:keep]
        trimmed["waypoints"] = episode["waypoints"][:keep]
        trimmed["preds"] = {m: (None if p is None else p[:keep])
                            for m, p in episode["preds"].items()}
        out.append(trimmed)
    return out


def load_frames(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            out[(row["task"], row["failure_condition"])] = row
    return out


def confusion(pairs) -> dict:
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for gt, pred in pairs:
        counts["TP" if gt and pred else
               "FN" if gt else
               "FP" if pred else "TN"] += 1
    return counts


def metrics(counts: dict) -> dict:
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]

    def ratio(num, den):
        return 100.0 * num / den if den else None

    prec, rec = ratio(tp, tp + fp), ratio(tp, tp + fn)
    if prec is None or rec is None:
        f1 = None
    elif prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return {"acc": ratio(tp + tn, tp + fp + fn + tn), "prec": prec, "rec": rec,
            "f1": f1, "fpr": ratio(fp, fp + tn), **counts}


def is_alarm(label: tuple[str, str | None], uncertain_is_failure: bool) -> bool:
    verdict = label[0]
    return verdict == "failure" or (uncertain_is_failure and verdict == "uncertain")


def pairs_for(episodes, method: str, unit: str, uncertain_is_failure=False):
    """(ground truth, prediction) booleans, one pair per sample."""
    for episode in episodes:
        preds = episode["preds"][method]
        if preds is None:
            continue
        gts = [v == "failure" for v, _ in episode["gt"]]
        alarms = [is_alarm(p, uncertain_is_failure) for p in preds]
        if unit == "episode":
            gt = episode["failure"] is not None
            # Credit only an alarm at or after the injection waypoint.
            pred = any(a for g, a in zip(gts, alarms) if g) if gt else any(alarms)
            yield gt, pred
        else:
            yield from zip(gts, alarms)


def type_accuracy(episodes, method: str, uncertain_is_failure=False):
    """Share of true detections whose predicted type matches the truth."""
    hits = total = 0
    for episode in episodes:
        preds = episode["preds"][method]
        if preds is None:
            continue
        for (gt_verdict, gt_type), pred in zip(episode["gt"], preds):
            if gt_verdict != "failure" or not is_alarm(pred, uncertain_is_failure):
                continue
            total += 1
            # A fused label ('Wrongtransition|Grasp') counts when the truth is
            # among the types it names.
            hits += int(gt_type in str(pred[1]).split("|"))
    return (100.0 * hits / total) if total else None


def misclassified(episodes, method: str, uncertain_is_failure=False):
    """Every checkpoint this method got wrong, as (kind, task, case, wp, gt, pred).

    kind is 'missed failure' (a real failure called success/uncertain),
    'false alarm' (an alarm on a clean checkpoint), or 'wrong type' (the failure
    was detected but none of the names it gave match the truth).
    """
    out = []
    for episode in episodes:
        preds = episode["preds"][method]
        if preds is None:
            continue
        for wp, ((gt_verdict, gt_type), pred) in zip(episode["waypoints"],
                                                     zip(episode["gt"], preds)):
            alarm = is_alarm(pred, uncertain_is_failure)
            raw = pred[0] if pred[1] is None else f"{pred[0]},{pred[1]}"
            if gt_verdict == "failure" and not alarm:
                kind = "missed failure"
            elif gt_verdict != "failure" and alarm:
                kind = "false alarm"
            elif gt_verdict == "failure" and gt_type not in str(pred[1]).split("|"):
                kind = "wrong type"
            else:
                continue
            out.append((kind, episode["task"], episode["case"], wp,
                        gt_type or "success", raw))
    order = {"missed failure": 0, "false alarm": 1, "wrong type": 2}
    return sorted(out, key=lambda r: (order[r[0]], r[1], r[2], r[3]))


MACRO_KEYS = ("acc", "prec", "rec", "f1", "fpr", "type", "lat")


def macro_metrics(per_failure: dict, method: str, mask_blind: bool) -> dict:
    """Unweighted mean of one method's per-failure-type scores.

    Table 1 pools checkpoints, so the frequent failure types decide it -- 29
    transition episodes against 2 wrong-object ones.  The macro average weights
    every failure type equally instead, which is the number to read when the
    question is how broadly a method works rather than how it does on the
    average checkpoint.

    A type contributes only where the method has a defined score for it, and
    with mask_blind the types it is structurally blind to drop out as well.  The
    two conventions average over different type sets, so `n` reports how many
    types went into each method's mean -- without it the columns are not
    comparable.
    """
    used = []
    for failure, _ in FAILURE_ROWS:
        entry = per_failure.get(failure)
        if not entry or method not in entry:
            continue                      # no episode of this type in the table

        used.append(entry[method])
    out = {"n": len(used)}
    for key in MACRO_KEYS:
        values = [m[key] for m in used if m.get(key) is not None]
        out[key] = (sum(values) / len(values)) if values else None
    return out


def latency(episodes, frames: dict, method: str):
    """Mean control steps from injection to the first crediting alarm."""
    gaps = []
    for episode in episodes:
        if episode["failure"] is None:
            continue
        row = frames.get((episode["task"], episode["case"]))
        if not row:
            continue
        alarm = _int(row.get(f"{method}_alarm_frame"))
        inj = _int(row.get("injection_frame"))
        if alarm is None or inj is None:
            continue
        gaps.append(max(0, alarm - inj))
    return (sum(gaps) / len(gaps)) if gaps else None


def fmt(value, blind=False) -> str:
    return "--" if blind or value is None else f"{value:.1f}"


def print_table(title: str, header: list[str], rows: list[list],
                groups: list[tuple[str, int]] | None = None) -> None:
    """`groups` spans a banner row over the columns, as [(label, n_cols), ...]."""
    body = [[str(c) for c in row] for row in rows]
    widths = [max(len(r[i]) for r in [header] + body) for i in range(len(header))]
    print(f"\n{title}")
    if groups:
        banner, first = [], 0
        for label, span in groups:
            width = sum(widths[first:first + span]) + 2 * (span - 1)
            banner.append(label.center(width))
            first += span
        print("  " + "  ".join(banner))
    print("  " + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for row in body:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(row, widths)))


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--frames", type=Path, default=None,
                        help="latency sidecar (default: <samples>_frames.csv)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="where the metrics CSVs go")
    parser.add_argument("--unit", choices=("checkpoint", "episode"),
                        default="checkpoint")
    parser.add_argument("--window", choices=("to-injection", "full"),
                        default="to-injection",
                        help="'to-injection' (default) keeps waypoints 0..inj and "
                             "drops everything after the failure waypoint, so each "
                             "failure episode contributes exactly one positive; "
                             "'full' keeps every checkpoint to the end")
    parser.add_argument("--methods", nargs="+", default=list(METHODS),
                        metavar="M",
                        help=f"methods to report (any of {', '.join(METHODS)}, "
                             f"or 'all'). Default: all of them")
    parser.add_argument("--list-errors", dest="list_errors",
                        action="store_true", default=True,
                        help="list every misclassified checkpoint (default on)")
    parser.add_argument("--no-list-errors", dest="list_errors",
                        action="store_false",
                        help="suppress the misclassified-checkpoint listing")


    parser.add_argument("--uncertain-as", choices=("success", "failure"),
                        default="success",
                        help="whether the baseline's 'uncertain' verdict counts "
                             "as an alarm (default: it does not)")
    parser.add_argument("--no-write", action="store_true",
                        help="print only; do not write the metrics CSVs")

    return parser.parse_args()


def main():
    args = parse_args()
    if any(m.lower() == "all" for m in args.methods):
        args.methods = list(METHODS)
    unknown = [m for m in args.methods if m not in METHODS]
    if unknown:
        sys.exit(f"unknown method(s): {', '.join(unknown)}; "
                 f"choose from {', '.join(METHODS)}")
    selected = [m for m in METHODS if m in args.methods]   # keep paper order

    if selected != list(METHODS):
        print(f"Reporting {', '.join(selected)} only "
              f"(--methods all for {', '.join(METHODS)}). "
              f"The metrics CSVs get just these columns.")
    episodes = load_samples(args.samples)
    frames = load_frames(args.frames or
                         args.samples.with_name(args.samples.stem + "_frames.csv"))

    # Only the reported methods decide the common episode set.
    incomplete = [e for e in episodes
                  if any(e["preds"][m] is None for m in selected)]


    if not episodes:
        sys.exit("every episode was filtered out; try --no-common-episodes")

    episodes = apply_window(episodes, args.window)
    clean = [e for e in episodes if e["failure"] is None]
    n_cp = sum(len(e["gt"]) for e in episodes)
    note = (f"unit={args.unit}; window={args.window}; {len(episodes)} episodes "
            f"({len(clean)} clean), {n_cp} checkpoints, "
            f"from {args.samples.name}")

    # With gaps allowed, each column may rest on a different sample set -- say so.
    coverage = []
    for method in selected:
        covered = [e for e in episodes if e["preds"][method] is not None]
        cps = sum(len(e["gt"]) for e in covered)
        gaps = sorted({e["task"] for e in episodes
                       if e["preds"][method] is None})
        coverage.append([METHOD_LABELS[method], len(covered), cps,
                         ", ".join(gaps) if gaps else "-"])
    if any(row[1] != len(episodes) for row in coverage):
        print_table(
            "Per-method coverage (columns below rest on different sample sets)",
            ["Method", "episodes", "checkpoints", "tasks with gaps"], coverage)
    if not frames:
        print("\n  (no latency sidecar found -- Lat. rows will print '--')")

    # ---- per-failure pools ----------------------------------------------- #
    # Computed before table 1 is printed, because table 1's macro half is the
    # mean of these.
    unc = args.uncertain_as == "failure"
    overall = {m: metrics(confusion(pairs_for(episodes, m, args.unit, unc)))
               for m in selected}
    for method in selected:
        overall[method]["type"] = type_accuracy(episodes, method, unc)

    per_failure: dict[str, dict | None] = {}
    for failure, _ in FAILURE_ROWS:
        subset = [e for e in episodes if e["failure"] == failure]
        if not subset:
            per_failure[failure] = None
            continue
        pool = subset + clean
        entry = {}
        for method in selected:
            met = metrics(confusion(pairs_for(pool, method, args.unit, unc)))
            met["lat"] = latency(subset, frames, method)
            met["type"] = type_accuracy(subset, method, unc)
            met["blind"] = failure in BLIND.get(method, set())
            entry[method] = met
        per_failure[failure] = entry

    # ---- table 1: micro and macro side by side ---------------------------- #
    # Micro pools every checkpoint, so the failure types with the most episodes
    # decide it; macro weights each failure type equally.  A method that only
    # works on the common failures scores far better on the left than the right.
    macro = {m: macro_metrics(per_failure, m, False) for m in selected}
    scored = [label for failure, label in FAILURE_ROWS if per_failure.get(failure)]
    panel = ("acc", "prec", "rec", "f1", "fpr", "type")
    heads = ["Acc.", "Prec.", "Rec.", "F1", "FPR", "Type"]
    print_table(
        f"Table 1 -- overall binary detection ({note})",
        ["Method"] + heads + heads + ["TP", "FP", "FN", "TN"],
        [[METHOD_LABELS[m]]
         + [fmt(overall[m][k]) for k in panel]
         + [fmt(macro[m][k]) for k in panel]
         + [overall[m][k] for k in ("TP", "FP", "FN", "TN")] for m in selected],
        groups=[("", 1), ("micro (per checkpoint)", len(heads)),
                (f"macro ({len(scored)} failure types)", len(heads)),
                ("micro counts", 4)])
    print(f"   macro types: {', '.join(scored)}")


    # ---- table 2 ---------------------------------------------------------- #
    for failure, label in FAILURE_ROWS:
        entry = per_failure.get(failure)
        if entry is None:
            print(f"\n  (no {failure} episode in {args.samples.name} "
                  f"-- row left as '--')")
            continue
        subset = [e for e in episodes if e["failure"] == failure]
        print_table(
            f"Table 2 -- {label} ({failure}; {len(subset)} episodes "
            f"+ {len(clean)} clean)",
            ["Method", "Prec.", "Rec.", "FPR", "Lat.", "Type", "TP", "FP",
             "FN", "TN"],
            [[METHOD_LABELS[m]]
             + [fmt(entry[m][k], False)
                for k in ("prec", "rec", "fpr", "lat", "type")]
             + [entry[m][k] for k in ("TP", "FP", "FN", "TN")] for m in selected])


    if args.list_errors:
        for method in selected:
            errors = misclassified(episodes, method, unc)
            tally = Counter(kind for kind, *_ in errors)
            summary = ", ".join(f"{tally[k]} {k}" for k in
                                ("missed failure", "false alarm", "wrong type")
                                if tally[k])
            print_table(
                f"Misclassified checkpoints -- {METHOD_LABELS[method]} "
                f"({len(errors)} of {n_cp}: {summary or 'none'})",
                ["kind", "task", "failure_condition", "wp", "truth", "predicted"],
                [list(row) for row in errors])

    if args.no_write:
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with (args.out_dir / "metrics_overall.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        keys = ("acc", "prec", "rec", "f1", "fpr", "type",
                "TP", "FP", "FN", "TN")
        writer.writerow(["averaging", "method", *keys, "n"])
        for method in selected:
            met = overall[method]
            writer.writerow(["micro", method] + [met[k] for k in keys]
                            + [met["TP"] + met["FP"] + met["FN"] + met["TN"]])
        for method in selected:
            met = macro[method]
            # n is the number of failure types averaged, not a checkpoint count.
            writer.writerow(["macro", method, ""]
                            + [met[k] for k in ("prec", "rec", "f1", "fpr",
                                                "type")]
                            + ["", "", "", ""] + [met["n"]])

    with (args.out_dir / "metrics_per_failure.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["failure", "method", "applicable", "prec", "rec", "fpr",
                         "latency_steps", "type_accuracy",
                         "TP", "FP", "FN", "TN", "n"])
        for failure, _ in FAILURE_ROWS:
            entry = per_failure.get(failure)
            if not entry:
                continue
            for method in selected:
                met = entry[method]
                writer.writerow(
                    [failure, method, not met["blind"]]
                    + [met[k] for k in ("prec", "rec", "fpr", "lat", "type",
                                        "TP", "FP", "FN", "TN")]
                    + [met["TP"] + met["FP"] + met["FN"] + met["TN"]])

    print(f"\nwrote {args.out_dir}/"
          "{metrics_overall.csv,metrics_per_failure.csv}")


if __name__ == "__main__":
    main()
