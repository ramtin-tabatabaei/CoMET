"""Create all four detector plots and trace exports from recorded telemetry."""

from contextlib import contextmanager
import csv
import json
import math
import os
from pathlib import Path

from aha_publish import paths


def _read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def _figure(path, *args, **kwargs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(*args, **kwargs)
    try:
        yield fig, axes
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def _waypoint_boundaries(rows):
    boundaries, previous = [], None
    for row in rows:
        try:
            waypoint = int(float(row["waypoint"]))
            step = int(float(row["step"]))
        except (KeyError, TypeError, ValueError):
            continue
        if waypoint != previous:
            boundaries.append((waypoint, step))
            previous = waypoint
    return boundaries


def _plot_pose(path, rows, task, label, detector, fired_waypoints):
    orientation = detector == "orientation"
    error_key = "angle_to_target" if orientation else "dist_to_target"
    motion_key = "gripper_orientation_delta" if orientation else "gripper_motion"
    unit = "rad" if orientation else "m"
    frames = list(range(len(rows)))
    waypoints = [int(row["waypoint"]) for row in rows]
    values = [float(row[error_key]) for row in rows]
    baseline_path = paths.CALIBRATION_DIR / f"{detector}_arrival_stats" / f"{task}.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    stats = baseline.get("arrival_stats_by_waypoint", {})
    thresholds = [stats.get(str(wp), {}).get("threshold", float("nan")) for wp in waypoints]

    with _figure(path, 2, 1, figsize=(12, 7), sharex=True) as (fig, axes):
        fig.suptitle(f"{task} {label}: {detector} telemetry")
        axes[0].plot(frames, values, label="error to target")
        if stats:
            axes[0].step(frames, thresholds, where="post", linestyle="--",
                         label="calibrated arrival threshold (checked at arrival)")
        arrivals = [i for i, row in enumerate(rows) if int(row.get("path_done") or 0)]
        axes[0].scatter(arrivals, [values[i] for i in arrivals], s=15, label="path complete")
        axes[0].set_ylabel(f"target error ({unit})")
        axes[1].plot(frames, [float(row.get(motion_key) or "nan") for row in rows],
                     label="motion per recorded frame")
        axes[1].set_ylabel(f"motion ({unit}/frame)")
        axes[1].set_xlabel("recorded frame")
        for i, waypoint in enumerate(waypoints):
            if i and waypoint == waypoints[i - 1]:
                continue
            flagged = waypoint in (fired_waypoints or ())
            for ax in axes:
                ax.axvline(i, color="red" if flagged else "gray", alpha=0.35)
                ax.text(i, 0.98, f"wp{waypoint}" + (" (flagged)" if flagged else ""),
                        transform=ax.get_xaxis_transform(), rotation=90, va="top", fontsize=8)
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right", fontsize=8)


def _number(row, key):
    try:
        value = float(row.get(key))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _plot_slip(path, rows, task, label):
    steps = [int(row["step"]) for row in rows]
    grip = [float(row["grip_force"]) for row in rows]
    fired = sorted({int(row["step"]) for row in rows if row["slip_fired"]})
    drop_fraction = float(os.getenv("AHA_SLIP_DROP_FRAC", "0.8"))
    peaks = [_number(row, "held_peak") for row in rows]
    peak_steps = [step for step, peak in zip(steps, peaks) if peak and peak > 0]
    peak_values = [peak for peak in peaks if peak and peak > 0]

    with _figure(path, 1, 1, figsize=(12, 5)) as (fig, ax):
        fig.suptitle(f"{task} {label}: slip telemetry (proportional grip collapse)")
        ax.plot(steps, grip, label="grip_force", color="#2563eb")
        if peak_steps:
            ax.plot(peak_steps, peak_values, label="held peak", color="#10b981",
                    linestyle=":", alpha=0.8)
            ax.plot(peak_steps, [(1 - drop_fraction) * peak for peak in peak_values],
                    label=f"slip level ({int(drop_fraction * 100)}% drop)",
                    color="#a855f7", linestyle="--", alpha=0.8)
        ax.set_ylabel("grip_force")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("frame / detector step")

        for waypoint, step in _waypoint_boundaries(rows):
            ax.axvline(step, color="#7c3aed", linestyle="--", alpha=0.35)
            ax.text(step, 0.98, f"wp{waypoint}", transform=ax.get_xaxis_transform(),
                    rotation=90, va="top", ha="right", fontsize=8, color="#7c3aed")
        for key, text, color, alpha, alignment in (
            ("gripper_close_cmd", "close cmd", "#0891b2", 0.8, "right"),
            ("gripper_open_cmd", "open cmd", "#16a34a", 0.7, "left"),
        ):
            for row in rows:
                if _number(row, key) == 1:
                    step = int(row["step"])
                    ax.axvline(step, color=color, linestyle="-", alpha=alpha)
                    ax.text(step, 0.98, text, transform=ax.get_xaxis_transform(),
                            rotation=90, va="top", ha=alignment, fontsize=8, color=color)
        for row in rows:
            step = int(row["step"])
            if not row["holding_required_phase"]:
                ax.axvspan(step - 0.5, step + 0.5, color="#e5e7eb", alpha=0.18, linewidth=0)
            if row["path_done"]:
                ax.axvline(step, color="#64748b", linestyle=":", alpha=0.18)
        for index, step in enumerate(fired):
            ax.axvline(step, color="#dc2626", linestyle="-", linewidth=1.8,
                       alpha=0.9, label="slip fired" if index == 0 else None)
        ax.legend(loc="upper right", fontsize=8)


def _collision_fired_steps(rows):
    from aha_publish.detectors.collision.detector import DEFAULT_CONSECUTIVE_COLLISION_FRAMES

    required = max(1, int(DEFAULT_CONSECUTIVE_COLLISION_FRAMES))
    return [int(float(row["step"])) for row in rows
            if str(row.get("collision_raw")) == "True"
            and int(float(row["consecutive_collision_frames"])) >= required]


def _plot_collision(path, rows, task, label, waypoint_frames, injected_waypoint):
    from aha_publish.detectors.collision import detector

    stats = detector.load_residual_stats(task)
    thresholds = detector.residual_threshold_vector(stats) if stats is not None else None
    minimum_joints = detector.DEFAULT_RESIDUAL_MIN_JOINTS
    steps = [int(row["step"]) for row in rows]
    fired = _collision_fired_steps(rows)

    with _figure(path, 1, 1, figsize=(11, 6)) as (fig, ax):
        from matplotlib import cm

        fig.suptitle(f"{task} {label}: De Luca generalized-momentum observer "
                     f"(collision method 3, fire when >= {minimum_joints} joints over)")
        colors = cm.tab10([joint / 6 for joint in range(7)])
        for joint in range(7):
            values = [abs(float(row.get(f"r{joint}") or 0.0)) for row in rows]
            ax.plot(steps, values, color=colors[joint], linewidth=1.2, label=f"|r{joint + 1}|")
            if thresholds is not None:
                ax.axhline(thresholds[joint], color=colors[joint], linestyle=":",
                           linewidth=1.0, alpha=0.7)
        ax.set_ylabel("per-joint residual |r_i|  (Nm)")
        ax.set_xlabel("frame / simulator step")
        ax.legend(loc="upper left", fontsize=7, ncol=7, title="dotted = per-joint threshold")
        for waypoint, frame in waypoint_frames:
            injected = injected_waypoint is not None and waypoint == injected_waypoint
            color = "#dc2626" if injected else "#9ca3af"
            ax.axvline(frame, color=color, linestyle="-", alpha=0.8 if injected else 0.4,
                       linewidth=1.3)
            ax.text(frame, ax.get_ylim()[1], f"wp{waypoint}", color=color,
                    fontsize=8, va="bottom", ha="center")
        for step in fired:
            ax.axvline(step, color="#000000", linestyle=":", alpha=0.3)
        if fired:
            first = min(fired)
            previous = [waypoint for waypoint, start in waypoint_frames if start <= first]
            waypoint_text = f" (wp{previous[-1]})" if previous else ""
            ax.axvline(first, color="#ef4444", linestyle="-", linewidth=2.4, alpha=0.95,
                       label=f"first detection @ frame {first}{waypoint_text}")
            ax.legend(loc="upper left", fontsize=7, ncol=7)
        ax.grid(True, alpha=0.25)


def render_plots(case_dir, task, label, drift, info, *, failure="none", waypoint=None):
    """Save available detector plots; return paths or individual plot errors."""
    directory = Path(case_dir)
    result, recordings = {}, {}
    for name in ("orientation", "transition", "slip", "collision"):
        filename = drift.get(name)
        if not filename or not Path(filename).is_file():
            continue
        try:
            rows = _read_rows(filename)
            if not rows:
                continue
            recordings[name] = rows
            if name == "slip":
                for row in rows:
                    for key, default in (("holding_required_phase", 1), ("slip_fired", 0)):
                        row[key] = bool(int(row.get(key, default)))
                    row["path_done"] = int(row.get("path_done", 0))
                    row["step"] = int(row.get("step", 0))
            png = directory / f"{name}.png"
            if name in ("orientation", "transition"):
                flags = info.get(name, {})
                _plot_pose(png, rows, task, label, name,
                           flags.get("fired_waypoints_confirmed", flags.get("fired_waypoints", [])))
            elif name == "slip":
                _write_csv(directory / "slip_trace.csv", rows)
                _plot_slip(png, rows, task, label)
            else:
                _write_csv(directory / "collision_trace.csv", rows)
                boundaries = _waypoint_boundaries(rows) or _waypoint_boundaries(recordings.get("slip", []))
                _plot_collision(png, rows, task, label, boundaries,
                                waypoint if failure == "collision" else None)
            result[name] = str(png)
        except Exception as exc:
            result[name] = f"plot_err:{exc}"
    return result
