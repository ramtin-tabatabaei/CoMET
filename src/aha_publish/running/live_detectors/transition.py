'Live transition detector: compare the gripper position with the clean-calibrated waypoint reference at arrival.'

from aha_publish import paths

import os
from pathlib import Path

import numpy as np

from ._bundle import (
    VLM_AUTO, load_detector_bundle)

NAME = "transition"
FAILURE = "transition"
CONDITION = "reaches_waypoint() == True"
# No live waypoint pose needed: arrival distance is measured against the composed reference
# (ttm_waypoint_distance), never the live target.
NEEDS_WAYPOINT_POSE = False

# Flat floor for the arrival threshold (m): the smallest arrival deviation worth
# firing on. It must sit ABOVE clean servo noise (~0.002 m) and BELOW the real
# failure arrival deviation. This is deliberately DECOUPLED from the abstain
# ceiling: the floor defines the smallest arrival error we care to flag, while
# the high ceiling drops dynamic waypoints.
ARRIVAL_DISTANCE_THRESHOLD = float(
    os.getenv("AHA_TRANSITION_ARRIVAL_THRESHOLD", "0.02"))
# Use the clean per-waypoint thresholds by default, bounded by the floor above.
ARRIVAL_USE_BASELINE = os.getenv(
    "AHA_TRANSITION_ARRIVAL_USE_BASELINE", "1").strip().lower() in (
        "1", "true", "yes", "on")


_CALIBRATION_ROOT = Path(os.getenv(
    "AHA_CALIBRATION_ROOT",
    str(paths.CALIBRATION_DIR),
)).expanduser()
ARRIVAL_STATS_DIR = Path(os.getenv(
    "AHA_TRANSITION_STATS_DIR",
    str(_CALIBRATION_ROOT / "transition_arrival_stats"),
)).expanduser()

ARRIVAL_SIGMA_K = float(os.getenv("AHA_TRANSITION_ARRIVAL_SIGMA_K", "3"))

# When set, every frame's signals are appended to this CSV so the eval harness
# can derive the per-task thresholds offline from clean trajectories.
DRIFT_LOG_PATH = os.getenv("AHA_TRANSITION_DRIFT_LOG", "").strip()


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
                            max(ARRIVAL_DISTANCE_THRESHOLD,
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
        print(f"  [transition] could not load arrival baseline: {exc}")
    return {}


class LiveDetector:
    NEEDS_ORIGINAL_POSE = True

    def __init__(self, task_name, *, env_wrapper=None, vlm_enabled=True, vlm_model=None, n_waypoints=0, failure_waypoint=None):
        self.task_name = task_name
        self.env_wrapper = env_wrapper

        self.vlm_model = vlm_model


        b = load_detector_bundle(NAME)
        self.det = b.detector




        self.arrival_thr = ARRIVAL_DISTANCE_THRESHOLD
        self.failure_waypoint = failure_waypoint
        self.arrival_baseline = load_arrival_baseline(task_name)

        print(
            "  [transition] C1 arrival: fire at waypoint end when the "
            f"distance to the waypoint position > "
            f"max({self.arrival_thr:g}, median+{ARRIVAL_SIGMA_K:g}*robust_std) per waypoint"
        )


        if self.arrival_baseline:
            print(
                f"  [transition] arrival baseline ({len(self.arrival_baseline)} wp): "
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
        self._last_motion_err = float('nan')
        self._cumulative_runup = 0.0
        self.recent_obs = []

        self.fed = 0
        self.checked = 0
        self.fire_count = 0
        self.first_fire = None
        self.fire_mode = None
        self.fired_waypoints = set()
        self._fired_this_execution = set()
        self.disabled_reason = None
        self.diag = {}
        self._wp_arrival_dist = {}  # waypoint -> worst arrival distance (m)
        self._wp_rise_peak = {}     # waypoint -> worst accumulated run-up (m)
        self.last_waypoint = None

        # Per-frame signal CSV (for offline per-task threshold calibration).
        self._drift_log_rows = []
        self._drift_log_path = DRIFT_LOG_PATH or None


    def _waypoint_threshold(self, waypoint):
        'Per-waypoint clean arrival threshold, bounded by the configured floor.'
        if not ARRIVAL_USE_BASELINE:
            return self.arrival_thr
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
        self._last_motion_err = float('nan')
        self._cumulative_runup = 0.0

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
            self.recent_obs.append(obs)
            if len(self.recent_obs) > 9:
                self.recent_obs.pop(0)
            local_step = self._local_step
            self._local_step += 1
            self.step_counter += 1

            row = det.obs_to_row(
                obs, local_step, waypoint=waypoint, waypoint_pose=waypoint_pose,
                ttm_waypoint_pose=original_pose)
            self.logs.append(row)
            if len(self.logs) > 300:
                del self.logs[0]
            det.update_deltas(self.logs)

            if phase == "gripper":

                return False

            # Per-frame signals: end-effector step (rate), distance to the
            # original/TTM waypoint pose, and its per-frame change. This is the
            # value judged at waypoint arrival, so a shifted waypoint target is
            # detected only when the robot reaches the shifted pose, not at the
            # instant the waypoint command changes.
            gm = row.get("gripper_motion", float('nan'))
            try:
                gm = float(gm)
            except (TypeError, ValueError):
                gm = float('nan')
            err = row.get("waypoint_distance", float('nan'))
            try:
                err = float(err)
            except (TypeError, ValueError):
                err = float('nan')

            # Live run-up = cumulative positive increase of the distance during
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
                     err, runup, int(runup_positive), gm, int(bool(path_done))))

            if np.isfinite(runup):
                self._wp_rise_peak[waypoint] = max(
                    self._wp_rise_peak.get(waypoint, 0.0), runup)

            if not path_done and np.isfinite(err):
                self._last_motion_err = err


            if not path_done:

                return False

            arrival_err = (
                self._last_motion_err
                if np.isfinite(self._last_motion_err)
                else err
            )
            if not np.isfinite(arrival_err):

                return False

            self.diag["max_distance"] = max(
                self.diag.get("max_distance", 0.0), arrival_err)
            self._wp_arrival_dist[waypoint] = max(
                self._wp_arrival_dist.get(waypoint, 0.0), arrival_err)


            check_row = dict(row)
            check_row["waypoint_distance"] = arrival_err
            check_row["waypoint_distance_source"] = row.get(
                "waypoint_distance_source", "ttm_recalculated")


            wp_thr = self._waypoint_threshold(waypoint)
            raw = arrival_err >= wp_thr
            reason = (
                f"previous in-motion distance to waypoint "
                f"({check_row['waypoint_distance_source']}) was "
                f"{arrival_err:.3f} m before arrival (>= {wp_thr:.3f})")
            if (raw and report and waypoint not in self._fired_this_execution
):
                return self._do_fire(check_row, row["step"], waypoint, reason,
                                     "arrival")

            return False
        except Exception as exc:
            self.disabled_reason = f"runtime error: {exc}"
            print(f"  [detector:transition] disabled mid-run — {exc}")
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
            f"\n  [DETECTOR] TRANSITION failure at waypoint {waypoint} "
            f"(step {step}) mode={mode}  reason={reason}\n",
            flush=True,
        )
        self.paused_steps.append(step)


        if not VLM_AUTO:
            try:
                input("  [transition] press Enter to continue the run...")
            except EOFError:
                pass


    def diagnostics_line(self):
        per_wp = "; ".join(
            f"wp{wp}={val:.3f}(>={self._waypoint_threshold(wp):.3f})"
            for wp, val in sorted(self._wp_arrival_dist.items())
        )
        rise_wp = "; ".join(
            f"wp{wp}={val:.3f}" for wp, val in sorted(self._wp_rise_peak.items())
        )
        fired_by = f"  fired_by={self.fire_mode}" if self.fire_mode else ""
        return (f"      diag: fed {self.fed} obs; max waypoint-distance="
                f"{self.diag.get('max_distance', 0):.3f} m "
                f"(fires >= {self.arrival_thr:.3f}){fired_by}"
                f"{('  arrival distance per-waypoint: ' + per_wp) if per_wp else ''}"
                f"{('  rise-peak per-waypoint: ' + rise_wp) if rise_wp else ''}")

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
                                "dist_to_target", "runup", "runup_positive",
                                "gripper_motion", "path_done"])
                w.writerows(self._drift_log_rows)
            self._drift_log_rows = []
        except Exception as exc:
            print(f"  [transition] drift-log flush failed: {exc}")


    def close(self):
        self._flush_drift_log()
