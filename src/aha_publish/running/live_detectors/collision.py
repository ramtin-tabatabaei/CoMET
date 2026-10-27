'Detect calibrated De Luca momentum residuals and confirm collision candidates with the VLM.'


import os

from ._bundle import load_detector_bundle

NAME = "collision"
FAILURE = "collision"
CONDITION = "no_collision() == True"
# Collision uses joint-torque residuals and needs no waypoint pose.
NEEDS_WAYPOINT_POSE = False

# Optional per-step torque-telemetry log (same columns as
# plot_maker). Set AHA_COLLISION_DRIFT_LOG to a path
# and the detector streams every fed frame there so a caller can render the
# residual trace for the actual run. Unset -> no logging.
COLLISION_DRIFT_LOG_PATH = os.getenv("AHA_COLLISION_DRIFT_LOG", "").strip()

# Column order for the recorded residual telemetry.
_DRIFT_FIELDS = ('step', 'torque_norm', 'torque_delta', 'waypoint', 'collision_raw', 'consecutive_collision_frames', 'collision_reason', 'is_holding', 'r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'residual_score', 'residual_norm', 'rthr0', 'rthr1', 'rthr2', 'rthr3', 'rthr4', 'rthr5', 'rthr6', 'residual_threshold', 'residual_crossed', 'residual_joints_over')


class LiveDetector:
    # This class-level copy is what LiveDetectorMonitor's skip gate actually
    # reads (__init__.py:325); the module-level constant above is not consulted.
    NEEDS_WAYPOINT_POSE = False

    def __init__(self, task_name, *, env_wrapper=None, vlm_enabled=True, vlm_model=None, n_waypoints=0, failure_waypoint=None):
        self.task_name = task_name
        self.env_wrapper = env_wrapper
        self.vlm_enabled = vlm_enabled
        self.vlm_model = vlm_model

        # waypoint index -> object the BT stage manipulates at that waypoint.
        # Filled in by LiveDetectorMonitor from the BT stages; the VLM verifier
        # uses it to tell intended contact (with this object) from unintended.
        self.bt_target_objects = {}
        # waypoint index -> that BT stage's contract: {primitive,
        # preconditions, postconditions}. Also filled in by
        # LiveDetectorMonitor; it replaces the description JSON's prose action
        # in the VLM prompt.
        self.bt_stage_contexts = {}

        b = load_detector_bundle(NAME)
        self.det = b.detector

        self.vlm_mod = b.vlm
        self.WARMUP = int(self.det.WARMUP_STEPS)

        # --- detector settings (copied from interactive.run setup) ---
        settings = self.det.default_detector_settings()
        self.settings = settings
        # Method-3 momentum-observer per-joint residual thresholds from a clean
        # run (None -> warmup residual-envelope fallback in freeze_thresholds).
        try:
            self.residual_stats = self.det.load_residual_stats(task_name)
        except Exception:
            self.residual_stats = None
        # Use the detector's configured persistence (default: one frame).
        _consec = int(self.settings.get(
            "consecutive_collision_frames",
            self.det.DEFAULT_CONSECUTIVE_COLLISION_FRAMES))
        self.settings["consecutive_collision_frames"] = max(1, _consec)

        # Per-step telemetry for the default saved residual trace.
        self._drift_log_path = COLLISION_DRIFT_LOG_PATH or None
        self._drift_log_rows = []

        # --- per-run state (copied from interactive.run) ---
        self.logs = []
        self.paused_steps = []
        self.step_counter = 0
        self.frozen_thr = None
        self.consecutive = 0
        self.recent_obs = []

        # --- bookkeeping for the bt_gui coordinator ---
        self.fed = 0
        self.checked = 0
        self.fire_count = 0
        self.first_fire = None
        self.fired_waypoints = set()
        self._vlm_rejected = set()  # waypoints whose detection the VLM refuted
        self.disabled_reason = None
        self.diag = {}
        self.last_waypoint = None

    # ---- per-step driver (copied patched_step body) -----------------------
    def step(self, obs, waypoint=None, path_done=False, waypoint_pose=None,
             original_pose=None, report=True):
        if self.disabled_reason is not None:
            return False
        self.fed += 1
        if report:
            self.checked += 1
        self.last_waypoint = waypoint
        det = self.det
        try:
            step = self.step_counter
            self.step_counter += 1

            self.recent_obs.append(obs)
            if len(self.recent_obs) > 9:
                self.recent_obs.pop(0)
            row = det.obs_to_row(obs, step)
            self.logs.append(row)
            if len(self.logs) > 300:
                del self.logs[0]
            det.update_deltas(self.logs)

            self.diag["max_tq_norm"] = max(
                self.diag.get("max_tq_norm", 0.0), row.get("torque_norm", 0.0))
            self.diag["max_tq_delta"] = max(
                self.diag.get("max_tq_delta", 0.0), row.get("torque_delta", 0.0))

            if step < self.WARMUP:
                self.consecutive = 0
                self._append_drift_row(row, "warmup")

                return False

            if self.frozen_thr is None:
                self.frozen_thr = det.freeze_thresholds(
                    self.logs[:self.WARMUP], self.settings["threshold_overrides"],
                    residual_stats=self.residual_stats)

            raw, reason = det.check_collision(row, self.frozen_thr)
            row["collision_reason"] = reason
            row["collision"] = raw
            if raw:
                self.consecutive += 1
            else:
                self.consecutive = 0
            self._append_drift_row(row, reason)


            required = max(1, int(self.settings["consecutive_collision_frames"]))
            confirmed = raw and self.consecutive >= required and report
            # Report once per waypoint (the BT run carries many steps per waypoint).
            if (confirmed and waypoint not in self.fired_waypoints
                    and waypoint not in self._vlm_rejected):
                return self._fire(step, reason, row, waypoint)

            return False
        except Exception as exc:
            self.disabled_reason = f"runtime error: {exc}"
            print(f"  [detector:collision] disabled mid-run — {exc}")
            return False


    # ---- fire: bookkeeping + announce -> (deferred) VLM verify -------------
    def _fire(self, step, reason, row, waypoint):
        'Record a detector candidate, then apply immediate VLM confirmation when enabled.'
        self.fire_count += 1
        self.fired_waypoints.add(waypoint)
        if self.first_fire is None:
            self.first_fire = (step, reason, waypoint)
        self._announce_detection(step, reason, row, waypoint)


        verdict = self._run_vlm_and_pause(row, waypoint)
        return self._apply_vlm_verdict(verdict, waypoint)

    # ---- on-detection: print -> plot -> (optional) VLM -> pause -----------
    def _announce_detection(self, step, reason, row, waypoint):
        print(
            f"\n  [DETECTOR] COLLISION detected at waypoint {waypoint} "
            f"(step {step})  reason={reason}  "
            f"torque_norm={row['torque_norm']:.3f}  "
            f"torque_delta={row['torque_delta']:.3f}\n",
            flush=True,
        )
        self.paused_steps.append(step)


    def _run_vlm_and_pause(self, row, waypoint):
        verdict = None
        if self.vlm_enabled and self.vlm_mod is not None:
            verdict = self._prompt_vlm(row, waypoint)

        return verdict

    def _apply_vlm_verdict(self, verdict, waypoint):
        if verdict is False:
            print("  [detector:collision] detection RETRACTED — "
                  "VLM verifier says no collision.")
            self.fired_waypoints.discard(waypoint)
            self._vlm_rejected.add(waypoint)
            self.fire_count -= 1

            return False

        return True


    # ---- VLM confirmation (copied prompt_vlm_confirmation) ----------------
    def _prompt_vlm(self, row, waypoint):
        vlm = self.vlm_mod
        try:
            sampled_obs, step_offsets = vlm.sample_recent_observations(
                list(self.recent_obs))

        except Exception as exc:
            print(f"  [vlm] image preview failed: {exc}")
            sampled_obs, step_offsets = list(self.recent_obs), None

        print("  [vlm] auto-confirming collision detection with the VLM "
              "verifier...")
        try:
            result = vlm.confirm_collision_with_openai(
                sampled_obs,
                step_offsets=step_offsets,
                row=row,
                telemetry_history=list(self.logs),
                env_wrapper=self.env_wrapper,
                task_name=self.task_name,
                waypoint_index=waypoint,
                target_object=(self.bt_target_objects or {}).get(waypoint),
                stage_context=(self.bt_stage_contexts or {}).get(waypoint),
                model=self.vlm_model,
            )
        except Exception as exc:
            print(f"  [vlm] confirmation failed: {exc}")
            return None
        verdict = result.get("collision_happened")
        vtext = "YES" if verdict is True else "NO" if verdict is False else "UNKNOWN"
        print(f"  [vlm:{result.get('model')}] collision={vtext}  "
              f"{result.get('explanation', '')}  "
              f"cameras={', '.join(result.get('camera_names', []))}")


        return verdict

    def _append_drift_row(self, row, reason):
        """Record one per-step torque and residual telemetry row."""
        if not self._drift_log_path:
            return
        frozen = self.frozen_thr
        holding = bool(row.get("is_holding", False))
        # Per-joint momentum residuals are populated by update_deltas.
        mo = row.get("mo_residual")
        resid_cols = {
            f"r{j}": (float(mo[j]) if mo is not None and j < len(mo) else "")
            for j in range(7)
        }
        resid_cols["residual_score"] = row.get("residual_score", "")
        resid_cols["residual_norm"] = row.get("residual_norm", "")
        # The per-joint threshold the gate actually compared against, so a trace
        # can be re-scored later without having to infer which calibration and
        # which multipliers the run used.
        thr_vec = frozen.get("residual_thr_vec") if frozen else None
        for j in range(7):
            resid_cols[f"rthr{j}"] = (
                float(thr_vec[j]) if thr_vec is not None and j < len(thr_vec) else "")
        resid_cols["residual_threshold"] = (
            frozen.get("residual_thr", "") if frozen else "")
        resid_cols["residual_crossed"] = row.get("residual_crossed", "")
        resid_cols["residual_joints_over"] = row.get("residual_joints_over", "")
        self._drift_log_rows.append({
            **resid_cols,
            "step": row.get("step", self.step_counter),
            "torque_norm": row.get("torque_norm", ""),
            "torque_delta": row.get("torque_delta", ""),
            "waypoint": row.get("waypoint", ""),
            "collision_raw": row.get("collision", False),
            "consecutive_collision_frames": self.consecutive,
            "collision_reason": reason or row.get("collision_reason", ""),
            "is_holding": holding,
        })
        if len(self._drift_log_rows) >= 100:
            self._flush_drift_log()

    def _flush_drift_log(self):
        if not self._drift_log_path or not self._drift_log_rows:
            return
        import csv as _csv
        write_header = (not os.path.exists(self._drift_log_path)
                        or os.path.getsize(self._drift_log_path) == 0)
        try:
            with open(self._drift_log_path, "a", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=list(_DRIFT_FIELDS))
                if write_header:
                    w.writeheader()
                w.writerows(self._drift_log_rows)
            self._drift_log_rows = []
        except Exception as exc:
            print(f"  [detector:collision] drift log write failed ({exc})")
            self._drift_log_path = None

    def diagnostics_line(self):
        return (f"      diag: fed {self.fed} obs; max torque_norm="
                f"{self.diag.get('max_tq_norm', 0):.3f}, max torque_delta="
                f"{self.diag.get('max_tq_delta', 0):.3f}")

    def close(self):
        self._flush_drift_log()
