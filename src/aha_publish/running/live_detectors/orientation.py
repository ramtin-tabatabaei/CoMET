'Live orientation detector: compare the gripper orientation with the clean-calibrated waypoint reference at arrival.'

from aha_publish import paths

import os
from pathlib import Path

import numpy as np

from ._bundle import (
    VLM_AUTO, load_detector_bundle)

NAME = "orientation"
FAILURE = "orientation"
CONDITION = "orientation_maintained() == True"
# No live waypoint pose needed: arrival angle is measured against the composed reference
# (ttm_waypoint_angle), never the live target.
NEEDS_WAYPOINT_POSE = False

# Flat floor for the arrival threshold (rad): the smallest arrival angle worth
# firing on -- above clean servo noise (~0.07 rad) and below a real mis-
# orientation. DECOUPLED from the abstain ceiling (low floor catches real
# failures whose settled deviation can be well under the injected rotation; the
# high ceiling drops unconstrained waypoints).
ARRIVAL_ANGLE_THRESHOLD = float(
    os.getenv("AHA_ORIENTATION_ARRIVAL_THRESHOLD", "0.3"))


_CALIBRATION_ROOT = Path(os.getenv(
    "AHA_CALIBRATION_ROOT",
    str(paths.CALIBRATION_DIR),
)).expanduser()
ARRIVAL_STATS_DIR = Path(os.getenv(
    "AHA_ORIENTATION_STATS_DIR",
    str(_CALIBRATION_ROOT / "orientation_arrival_stats"),
)).expanduser()

# Per-waypoint arrival bands are loaded from the clean calibration.

ARRIVAL_SIGMA_K = float(os.getenv("AHA_ORIENTATION_ARRIVAL_SIGMA_K", "3"))

# When set, every frame's signals are appended to this CSV so the eval harness
# can derive the thresholds offline from clean + failure trajectories.
DRIFT_LOG_PATH = os.getenv("AHA_ORIENTATION_DRIFT_LOG", "").strip()


def load_arrival_baseline(task_name):
    'Load the per-waypoint arrival bands exported by clean calibration.'
    import json
    try:
        path = ARRIVAL_STATS_DIR / f"{task_name}.json"
        if path.exists():
            data = json.loads(path.read_text())

            stats = data.get("arrival_stats_by_waypoint")
            if stats:
                out = {}
                for k, v in stats.items():
                    mean = float(v.get("mean", 0.0))
                    std = float(v.get("std", 0.0))
                    mx = float(v.get("max", mean))
                    median = float(v.get("median", mean))
                    threshold = float(
                        v.get(
                            "threshold",
                            max(ARRIVAL_ANGLE_THRESHOLD,
                                median + ARRIVAL_SIGMA_K * std),
                        )
                    )
                    out[int(k)] = {
                        "mean": mean,
                        "median": median,
                        "std": std,
                        "max": mx,
                        "threshold": threshold,
                    }
                return out
    except Exception as exc:
        print(f"  [orientation] could not load arrival baseline: {exc}")
    return {}


class LiveDetector:
    NEEDS_ORIGINAL_POSE = True

    def __init__(self, task_name, *, env_wrapper=None, vlm_enabled=True, vlm_model=None, n_waypoints=0, failure_waypoint=None):
        self.task_name = task_name
        self.env_wrapper = env_wrapper

        self.vlm_model = vlm_model

        self.failure_waypoint = failure_waypoint

        b = load_detector_bundle(NAME)
        self.det = b.detector


        self.arrival_thr = ARRIVAL_ANGLE_THRESHOLD
        self.arrival_baseline = load_arrival_baseline(task_name)

        print(
            "  [orientation] C1 arrival: fire at waypoint end when the angle "
            f"to the waypoint orientation > "
            f"max({self.arrival_thr:g}, median+{ARRIVAL_SIGMA_K:g}*robust_std) per waypoint"
        )


        if self.arrival_baseline:
            print(
                f"  [orientation] arrival baseline ({len(self.arrival_baseline)} wp): "
                + ", ".join(f"wp{w}>={self._waypoint_threshold(w):.3f}"
                            for w in sorted(self.arrival_baseline))
            )
        print()

        self.logs = []
        self.step_counter = 0
        self.paused_steps = []
        self._cur_wp = None
        self._local_step = 0
        self._consec = 0


        self._prev_err = float('nan')
        self._cumulative_runup = 0.0
        self.waypoint_obs = []  # observations during the active waypoint (for VLM)

        self.fed = 0
        self.checked = 0
        self.fire_count = 0
        self.first_fire = None
        self.fire_mode = None
        self.fired_waypoints = set()
        self._fired_this_execution = set()
        self.disabled_reason = None
        self.diag = {}
        self._wp_arrival_dev = {}  # waypoint -> worst arrival deviation (rad)
        self._wp_drift_peak = {}   # waypoint -> worst accumulated drift (rad)
        self._wp_grad_peak = {}    # waypoint -> worst per-frame step (rad/frame)
        self.last_waypoint = None

        # Per-frame drift-signal CSV (for offline (theta,N) sweeping).
        self._drift_log_rows = []
        self._drift_log_path = DRIFT_LOG_PATH or None


    def _waypoint_threshold(self, waypoint):
        'Per-waypoint clean arrival threshold, bounded by the configured floor.'
        base = self.arrival_baseline.get(waypoint)
        if base is None:
            return self.arrival_thr
        return max(self.arrival_thr, float(base.get("threshold", self.arrival_thr)))


    def _reset_waypoint(self, waypoint):
        self._fired_this_execution.clear()
        self._cur_wp = waypoint
        self.logs = []
        self._local_step = 0
        self._consec = 0


        self._prev_err = float('nan')
        self._cumulative_runup = 0.0
        self.waypoint_obs = []

    def step(self, obs, waypoint=None, path_done=False, waypoint_pose=None,
             original_pose=None, report=True, phase="move"):
        if self.disabled_reason is not None:
            return False
        self.fed += 1
        if report:
            self.checked += 1
        self.last_waypoint = waypoint
        det = self.det
        try:
            if waypoint != self._cur_wp:
                self._reset_waypoint(waypoint)
            self.waypoint_obs.append(obs)
            local_step = self._local_step
            self._local_step += 1
            self.step_counter += 1

            reference_pose = (
                original_pose if original_pose is not None else waypoint_pose)
            row = det.obs_to_row(
                obs, local_step, waypoint=waypoint, waypoint_pose=waypoint_pose,
                ttm_waypoint_pose=reference_pose)
            self.logs.append(row)
            if len(self.logs) > 300:
                del self.logs[0]
            det.update_deltas(self.logs)

            if phase == "gripper":

                return False

            # Per-frame orientation step (rate of change) for the gradient check.
            god = row.get("gripper_orientation_delta", float('nan'))
            try:
                god = float(god)
            except (TypeError, ValueError):
                god = float('nan')
            if np.isfinite(god):
                self._wp_grad_peak[waypoint] = max(
                    self._wp_grad_peak.get(waypoint, 0.0), god)

            err = row.get("ttm_waypoint_angle", float('nan'))
            try:
                err = float(err)
            except (TypeError, ValueError):
                err = float('nan')
            if not np.isfinite(err):
                err = row.get("live_waypoint_angle", float('nan'))
                try:
                    err = float(err)
                except (TypeError, ValueError):
                    err = float('nan')
            # Live run-up = cumulative positive increase of the error during
            # this waypoint segment. It only adds the amount PLOT-1 increased;
            # decreases do not subtract from the cumulative value.
            runup_positive = False
            if np.isfinite(err):
                if np.isfinite(self._prev_err):
                    increase = max(0.0, err - self._prev_err)
                    if increase > 0.0:
                        self._cumulative_runup += increase
                        runup_positive = True
                self._prev_err = err
            runup = self._cumulative_runup if np.isfinite(err) else float('nan')
            if self._drift_log_path is not None:
                self._drift_log_rows.append(
                    (self.task_name, waypoint, local_step, row["step"],
                     err, runup, int(runup_positive),
                     god, int(bool(path_done))))

            if np.isfinite(runup):
                self._wp_drift_peak[waypoint] = max(
                    self._wp_drift_peak.get(waypoint, 0.0), runup)


            if not path_done:

                return False

            # C1 arrival uses the gripper-vs-correct error (err is
            # ttm_waypoint_angle = gripper vs the original/pre-injection
            # orientation), so a mutated-target rotation the arm tracked cleanly
            # still shows a large arrival deviation and fires.
            dev = err
            if not np.isfinite(dev):

                return False

            self.diag["max_arrival_dev"] = max(
                self.diag.get("max_arrival_dev", 0.0), dev)
            self._wp_arrival_dev[waypoint] = max(
                self._wp_arrival_dev.get(waypoint, 0.0), dev)


            wp_thr = self._waypoint_threshold(waypoint)
            raw = dev >= wp_thr
            reason = (
                f"arrived {dev:.3f} rad off the waypoint's correct "
                f"orientation (>= {wp_thr:.3f})")
            if (raw and report and waypoint not in self._fired_this_execution
):
                return self._do_fire(row, row["step"], waypoint, reason,
                                     "arrival")

            return False
        except Exception as exc:
            self.disabled_reason = f"runtime error: {exc}"
            print(f"  [detector:orientation] disabled mid-run — {exc}")
            return False

    def _do_fire(self, row, step, waypoint, reason, mode):
        'Record and report an arrival error above its calibrated band.'
        self.fire_count += 1
        self.fired_waypoints.add(waypoint)
        self._fired_this_execution.add(waypoint)
        self._flush_drift_log()
        if self.first_fire is None:
            self.first_fire = (step, reason, waypoint)
            self.fire_mode = mode
        self._on_detection(step, reason, row, waypoint, mode)
        return True

    def _on_detection(self, step, reason, row, waypoint, mode="?"):
        print(
            f"\n  [DETECTOR] ORIENTATION failure at waypoint {waypoint} "
            f"(step {step}) mode={mode}  reason={reason}\n",
            flush=True,
        )
        self.paused_steps.append(step)


        if not VLM_AUTO:
            try:
                input("  [orientation] press Enter to continue the run...")
            except EOFError:
                pass


    def diagnostics_line(self):
        per_wp = "; ".join(
            f"wp{wp}={val:.3f}" for wp, val in sorted(self._wp_arrival_dev.items())
        )
        drift_wp = "; ".join(
            f"wp{wp}={val:.3f}" for wp, val in sorted(self._wp_drift_peak.items())
        )
        grad_wp = "; ".join(
            f"wp{wp}={val:.3f}" for wp, val in sorted(self._wp_grad_peak.items())
        )
        fired_by = f"  fired_by={self.fire_mode}" if self.fire_mode else ""
        return (f"      diag: fed {self.fed} obs; max arrival deviation="
                f"{self.diag.get('max_arrival_dev', 0):.3f} rad "
                f"(fires >= {self.arrival_thr:.3f}){fired_by}"
                f"{('  per-waypoint: ' + per_wp) if per_wp else ''}"
                f"{('  drift-peak per-waypoint: ' + drift_wp) if drift_wp else ''}"
                f"{('  gradient-peak per-waypoint: ' + grad_wp) if grad_wp else ''}")

    def _flush_drift_log(self):
        if not self._drift_log_path or not self._drift_log_rows:
            return
        import csv
        path = self._drift_log_path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            exists = os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if not exists:
                    w.writerow(["task", "waypoint", "local_step", "step",
                                "angle_to_target", "runup",
                                "runup_positive", "gripper_orientation_delta",
                                "path_done"])
                w.writerows(self._drift_log_rows)
            self._drift_log_rows = []
        except Exception as exc:
            print(f"  [orientation] drift-log flush failed: {exc}")

    def close(self):
        self._flush_drift_log()
