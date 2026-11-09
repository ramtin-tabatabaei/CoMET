"""Create all four detector plots and trace exports from recorded telemetry."""

from contextlib import contextmanager
import csv
import json
import math
import os
from pathlib import Path
import threading

from src import paths

_PLOT_LOCK = threading.RLock()
_INK = '#172033'
_MUTED = '#64748b'
_BLUE = '#2563eb'
_TEAL = '#0f766e'
_AMBER = '#b45309'
_RED = '#dc2626'
_STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'text.color': _INK, 'axes.labelcolor': _MUTED,
    'axes.edgecolor': '#cbd5e1', 'axes.linewidth': .8,
    'xtick.color': _MUTED, 'ytick.color': _MUTED,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'legend.fontsize': 9,
    'lines.linewidth': 1.9, 'svg.fonttype': 'none',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
}


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

    # Matplotlib's global style and pyplot state are shared by batch threads.
    with _PLOT_LOCK, matplotlib.rc_context(_STYLE):
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        fig, axes = plt.subplots(*args, **kwargs)
        height = fig.get_figheight()
        fig.subplots_adjust(left=.085, right=.975, bottom=.95 / height, top=1 - 1.55 / height,
                            hspace=.30, wspace=.24)
        for ax in fig.axes:
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#e8edf3', linewidth=.7)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.tick_params(length=3, pad=6)
            ax.margins(x=.01, y=.12)
        try:
            yield fig, axes
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=180, facecolor='white')
            fig.savefig(path.with_suffix('.svg'), facecolor='white')
        finally:
            plt.close(fig)


def _heading(fig, title, task, label, detail):
    case = 'Clean run' if label == 'clean' else label.replace('_', ' ')
    height = fig.get_figheight()
    fig.text(.085, 1 - .25 / height, title, fontsize=19, fontweight='bold', va='top')
    fig.text(.085, 1 - .62 / height, f'{task.replace("_", " ").capitalize()}  /  {case}  /  {detail}',
             color=_MUTED, fontsize=10, va='top')


def _legend(fig, handles, labels, ncol=4):
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.079, 1 - .88 / fig.get_figheight()),
               ncol=ncol, handlelength=2.6, columnspacing=2, borderaxespad=0)


def _waypoint_strip(fig, boundaries, limits, *, flagged=(), injected=None, bounds=(.085, .89)):
    if not boundaries:
        return
    height = fig.get_figheight()
    left, width = bounds
    strip = fig.add_axes([left, 1 - 1.33 / height, width, .16 / height])
    strip.set_xlim(*limits)
    strip.set_ylim(0, 1)
    strip.set_axis_off()
    for index, (waypoint, start) in enumerate(boundaries):
        end = boundaries[index + 1][1] if index + 1 < len(boundaries) else limits[1]
        color = _RED if waypoint in flagged else '#7c3aed' if waypoint == injected else _MUTED
        strip.axvspan(start, end, color=color, alpha=.09 if index % 2 == 0 else .04, linewidth=0)
        strip.text((start + end) / 2, .5, f'WP {waypoint}', ha='center', va='center',
                   color=color, fontsize=8, fontweight='medium', clip_on=True)


def _guides(axes, boundaries, *, injected=None):
    for ax in axes:
        for waypoint, start in boundaries[1:]:
            ax.axvline(start, color='#7c3aed' if waypoint == injected else '#cbd5e1',
                       linewidth=.8, alpha=.65, zorder=0)


def _limits(frames):
    return (frames[0] - .5, frames[-1] + .5)


def _footer(fig, text):
    fig.text(.085, .15 / fig.get_figheight(), text, fontsize=8.5, color=_MUTED, va='bottom')


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

    boundaries = [(wp, index) for index, wp in enumerate(waypoints)
                  if index == 0 or wp != waypoints[index - 1]]
    with _figure(path, 2, 1, figsize=(13, 7.5), sharex=True,
                 gridspec_kw={'height_ratios': [2, 1]}) as (fig, axes):
        _heading(fig, f'{detector.capitalize()} during motion', task, label,
                 f'{len(rows):,} recorded frames')
        axes[0].plot(frames, values, color=_BLUE, label='Target error')
        axes[0].fill_between(frames, values, 0, color=_BLUE, alpha=.055)
        if stats:
            axes[0].step(frames, thresholds, where="post", linestyle="--",
                         color=_AMBER, linewidth=1.5, label='Arrival tolerance')
        arrivals = [i for i, row in enumerate(rows) if int(row.get("path_done") or 0)]
        axes[0].scatter(arrivals, [values[i] for i in arrivals], s=40, color=_BLUE,
                        edgecolors='white', linewidths=1.2, zorder=5, label='Waypoint arrival')
        axes[0].set_ylabel(f'{"Orientation" if orientation else "Position"} error ({unit})', labelpad=12)
        axes[1].plot(frames, [float(row.get(motion_key) or "nan") for row in rows],
                     color=_TEAL, linewidth=1.7)
        axes[1].set_ylabel(f'Motion ({unit}/frame)', labelpad=12)
        axes[1].set_xlabel('Recorded frame', labelpad=10)
        for ax in axes:
            ax.set_ylim(bottom=0)
            ax.set_xlim(*_limits(frames))
        _guides(axes, boundaries)
        _waypoint_strip(fig, boundaries, _limits(frames), flagged=fired_waypoints or ())
        _legend(fig, *axes[0].get_legend_handles_labels(), ncol=3)
        note = 'Tolerance is checked at waypoint arrival; errors can be larger while moving toward the target.'
        if fired_waypoints:
            note += '  Red waypoint labels indicate detections.'
        _footer(fig, note)


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
    peak_values = [peak if peak is not None and peak > 0 else float('nan') for peak in peaks]
    boundaries = _waypoint_boundaries(rows)

    with _figure(path, 1, 1, figsize=(13, 5.5)) as (fig, ax):
        _heading(fig, 'Grasp stability', task, label, f'{len(rows):,} recorded frames')
        ax.plot(steps, grip, label='Grip force', color=_BLUE, zorder=3)
        ax.fill_between(steps, grip, 0, color=_BLUE, alpha=.055)
        if any(peak is not None and peak > 0 for peak in peaks):
            ax.plot(steps, peak_values, label='Held-force peak', color=_TEAL,
                    linestyle=':', linewidth=1.6)
            ax.plot(steps, [(1 - drop_fraction) * peak for peak in peak_values],
                    label=f'Slip level ({drop_fraction:.0%} drop)',
                    color=_AMBER, linestyle='--', linewidth=1.5)
        ax.set_ylabel('Grip force (N)', labelpad=12)
        ax.set_xlabel('Detector step', labelpad=10)
        ax.set_ylim(bottom=0)
        ax.set_xlim(*_limits(steps))

        # Shade contiguous intervals, instead of drawing a rectangle per frame.
        inactive = None
        for index in range(len(rows) + 1):
            not_required = index < len(rows) and not rows[index]['holding_required_phase']
            if not_required and inactive is None:
                inactive = index
            if not not_required and inactive is not None:
                end = steps[index] - .5 if index < len(rows) else steps[-1] + .5
                ax.axvspan(steps[inactive] - .5, end, color='#e2e8f0', alpha=.4,
                           linewidth=0, zorder=0)
                inactive = None
        for key, text in (('gripper_close_cmd', 'Close'), ('gripper_open_cmd', 'Open')):
            for row in rows:
                if _number(row, key) == 1:
                    step = int(row["step"])
                    ax.axvline(step, color=_MUTED, linestyle='--', linewidth=.8, alpha=.65)
                    ax.annotate(text, xy=(step, .04), xytext=(5, 3),
                                xycoords=ax.get_xaxis_transform(), textcoords='offset points',
                                fontsize=8, color=_INK,
                                bbox={'facecolor': 'white', 'edgecolor': '#e2e8f0', 'pad': 3})
        for index, step in enumerate(fired):
            ax.axvline(step, color=_RED, linewidth=1.3, alpha=.8,
                       label='Slip trigger' if index == 0 else None)
        _guides([ax], boundaries)
        _waypoint_strip(fig, boundaries, _limits(steps))
        _legend(fig, *ax.get_legend_handles_labels())
        _footer(fig, 'Gray intervals: grasp is not required.  Close / Open markers show commanded gripper actions.')


def _collision_fired_steps(rows):
    from src.detectors.collision.detector import DEFAULT_CONSECUTIVE_COLLISION_FRAMES

    required = max(1, int(DEFAULT_CONSECUTIVE_COLLISION_FRAMES))
    return [int(float(row["step"])) for row in rows
            if str(row.get("collision_raw")) == "True"
            and int(float(row["consecutive_collision_frames"])) >= required]


def _plot_collision(path, rows, task, label, waypoint_frames, injected_waypoint):
    from src.detectors.collision import detector

    stats = detector.load_residual_stats(task)
    thresholds = detector.residual_threshold_vector(stats) if stats is not None else None
    minimum_joints = detector.DEFAULT_RESIDUAL_MIN_JOINTS
    steps = [int(row["step"]) for row in rows]
    fired = _collision_fired_steps(rows)

    with _figure(path, 4, 2, figsize=(13, 10), sharex=True) as (fig, grid):
        from matplotlib.lines import Line2D

        _heading(fig, 'Collision monitoring', task, label,
                 f'De Luca momentum observer  /  {len(rows):,} recorded frames')
        fig.subplots_adjust(hspace=.52, wspace=.23)
        axes = list(grid.flat)
        recorded_limits = False
        any_limits = False
        for joint in range(7):
            ax = axes[joint]
            values = [abs(value) if (value := _number(row, f'r{joint}')) is not None
                      else float('nan') for row in rows]
            ax.plot(steps, values, color=_BLUE, linewidth=1.6)
            ax.fill_between(steps, values, 0, color=_BLUE, alpha=.055)
            # A replot should use the limits recorded during this run, even if
            # the current calibration settings have since changed.
            limits = [_number(row, f'rthr{joint}') for row in rows]
            if any(value is not None for value in limits):
                recorded_limits = True
                limits = [value if value is not None else float('nan') for value in limits]
            elif thresholds is not None:
                limits = [float(thresholds[joint])] * len(rows)
            else:
                limits = [float('nan')] * len(rows)
            finite_limits = [value for value in limits if math.isfinite(value)]
            if finite_limits:
                any_limits = True
                ax.step(steps, limits, where='post', color=_AMBER,
                        linestyle='--', linewidth=1.3)
                ax.text(.99, 1.035, f'Limit {finite_limits[-1]:.2f} Nm',
                        transform=ax.transAxes, ha='right', fontsize=8, color=_AMBER)
            ax.set_title(f'Joint {joint + 1}', loc='left', fontsize=10, fontweight='bold', pad=7)
            ax.set_ylabel('|Residual| (Nm)', fontsize=9, labelpad=8)
            ax.set_xlim(*_limits(steps))
            ax.set_ylim(bottom=0)
            if fired:
                ax.axvline(min(fired), color=_RED, linewidth=1, alpha=.75)
        # The last panel is a reading guide, leaving seven uncluttered signals.
        axes[7].set_axis_off()
        axes[7].text(0, .93, 'Observer summary', fontsize=11, fontweight='bold', transform=axes[7].transAxes)
        threshold_source = ('Recorded per frame' if recorded_limits else
                            'Calibration fallback' if thresholds is not None else 'Unavailable')
        axes[7].text(0, .70, f'Trigger: at least {minimum_joints} joint(s) over their limit.',
                     fontsize=9, color=_MUTED, transform=axes[7].transAxes, va='top')
        axes[7].text(0, .48, f'{len(fired)} raw trigger frames  /  Thresholds: {threshold_source.lower()}',
                     fontsize=9, color=_MUTED, transform=axes[7].transAxes, va='top')
        axes[7].text(0, .04, 'Each joint has its own y-axis scale.\n'
                     'Raw triggers precede any VLM confirmation.', fontsize=8.5,
                     color=_MUTED, transform=axes[7].transAxes, va='bottom')
        for index in (5, 6):
            axes[index].tick_params(labelbottom=True)
            axes[index].set_xlabel('Simulator step', labelpad=8)
        _guides(axes[:7], waypoint_frames, injected=injected_waypoint)
        for ax in axes[:2]:
            position = ax.get_position()
            _waypoint_strip(fig, waypoint_frames, _limits(steps), injected=injected_waypoint,
                            bounds=(position.x0, position.width))
        handles = [Line2D([], [], color=_BLUE, linewidth=1.8)]
        labels = ['Absolute joint residual']
        if any_limits:
            handles.append(Line2D([], [], color=_AMBER, linestyle='--', linewidth=1.3))
            labels.append('Per-joint threshold')
        if fired:
            handles.append(Line2D([], [], color=_RED, linewidth=1.2))
            labels.append('First raw trigger')
        _legend(fig, handles, labels, ncol=3)
        _footer(fig, 'Waypoint boundaries are shared across panels.  Purple waypoint labels mark the collision injection, when present.')


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
