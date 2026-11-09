'Detect proportional grip-force collapse during an active grasp and confirm it with the VLM.'


import os

from ._bundle import load_detector_bundle

NAME = "slip"
FAILURE = "slip"
CONDITION = "maintains_grasp() == True"
NEEDS_WAYPOINT_POSE = False

# When set, every frame's signals are appended to this CSV so the eval harness
# can derive signals offline from BT-main runs (clean + failure trajectories).
SLIP_DRIFT_LOG_PATH = os.getenv("AHA_SLIP_DRIFT_LOG", "").strip()


class LiveDetector:
    def __init__(self, task_name, *, env_wrapper=None, vlm_enabled=True, vlm_model=None, n_waypoints=0, failure_waypoint=None):
        self.task_name = task_name
        self.env_wrapper = env_wrapper
        self.vlm_enabled = vlm_enabled
        self.vlm_model = vlm_model


        b = load_detector_bundle(NAME)
        self.det = b.detector

        self.vlm_mod = b.vlm
        self.WARMUP = int(self.det.WARMUP_STEPS)

        settings = self.det.default_detector_settings()
        try:
            apply_stats = getattr(
                self.det, "apply_task_grip_force_stats_thresholds", None)
            if apply_stats is not None:
                settings = apply_stats(settings, task_name)
        except Exception as exc:
            print(f"  [detector:slip] task grip-force stats unavailable ({exc}); "
                  f"using default thresholds.")
        self.settings = settings
        try:
            self.phase_info = self.det.load_holding_requirement_from_waypoints(
                task_name)
        except Exception:
            self.phase_info = {"enabled": False}

        self.logs = []
        self.paused_steps = []
        self.step_counter = 0
        self.frozen_thr = None
        self.recent_obs = []
        self._candidate_active = False
        self._consec = 0
        self._latched = False


        self._vlm_rejected = set()   # waypoints whose detection the VLM refuted
        self.fed = 0
        self.checked = 0
        self.fire_count = 0
        self.first_fire = None
        self.fired_waypoints = set()
        self.disabled_reason = None
        self.diag = {}
        self.last_waypoint = None


        self._drift_log_path = SLIP_DRIFT_LOG_PATH or None
        self._drift_log_rows = []
        self._last_obs_for_drift = None
        # Set by the runner the instant it issues an open_gripper command; the
        # next logged telemetry frame records it (gripper_open_cmd=1). This is
        # the exact command frame -- earlier and more accurate than the jaw
        # finishing its open motion.
        self._open_cmd_pending = False
        self._close_cmd_pending = False
        # Method-1 proportional state: peak grip during the current grasp, and
        # whether we are inside a grasp window (between a close and the next open
        # command). A grip drop outside this window is a commanded release, not a
        # slip -- so slip detection is gated on _grasp_active.
        self._held_peak = 0.0
        self._grasp_active = False

    def mark_open_command(self):
        """Called by the runner when it issues an open_gripper command: the grasp
        is intentionally released, so the grasp window closes (no slip after)."""
        self._open_cmd_pending = True
        self._grasp_active = False

    def mark_close_command(self):
        """Called by the runner when it issues a close_gripper command; opens a
        fresh grasp window and resets the held-peak baseline."""
        self._close_cmd_pending = True
        self._held_peak = 0.0
        self._grasp_active = True
    def step(self, obs, waypoint=None, path_done=False, waypoint_pose=None,
             original_pose=None, report=True):
        logs_before = len(self.logs)
        self._last_obs_for_drift = obs
        result = self._step_impl(obs, waypoint, path_done, waypoint_pose,
                                 original_pose, report)
        if self._drift_log_path and len(self.logs) > logs_before:
            self._append_drift_row(waypoint, path_done, result)
        return result

    def _step_impl(self, obs, waypoint=None, path_done=False, waypoint_pose=None,
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
            if len(self.recent_obs) > 30:
                self.recent_obs.pop(0)
            row = det.obs_to_row(obs, step)
            row["waypoint"] = waypoint
            phase_gate = (
                self.settings.get("use_waypoint_phase_gate", True)
                and bool(self.phase_info.get("enabled", False)))
            row["holding_required_phase"] = (
                det.waypoint_requires_holding(waypoint, self.phase_info)
                if phase_gate else True)

            self.logs.append(row)
            # No cap: keep the full per-step history for the whole run. A rolling
            # cap here also silently froze the drift/trace CSV once reached, because
            # step()'s `len(self.logs) > logs_before` write-gate can never be true
            # when append+trim keeps the length constant. Unbounded keeps every
            # waypoint's rows (the recompute below is idempotent and cheap at run
            # lengths of a few hundred steps).
            det.update_deltas(self.logs)

            if step < self.WARMUP:
                self._candidate_active = False
                self._consec = 0
                self._latched = False

                return False

            if self.frozen_thr is None:
                self.frozen_thr = det.freeze_thresholds(
                    self.settings["threshold_overrides"])
            det.update_holding_states(self.logs, self.frozen_thr)


            self.diag["max_grip"] = max(
                self.diag.get("max_grip", 0.0), row.get("grip_force", 0.0))
            self.diag["max_drop"] = max(
                self.diag.get("max_drop", 0.0), row.get("grip_force_drop", 0.0))
            if row.get("is_holding"):
                self.diag["holding_steps"] = self.diag.get("holding_steps", 0) + 1

            if not row["holding_required_phase"]:
                # A release stage still requires holding until the open command.
                if not self._grasp_active:
                    self._candidate_active = False
                    self._consec = 0
                    self._latched = False


                    return False

            # Method 1 (proportional): track the peak grip while a real grasp is
            # held -- "real" = grip cleared the baseline hold threshold at some
            # point -- and flag a slip when grip collapses by AHA_SLIP_DROP_FRAC
            # of that peak. Self-scaling per object; a load change that never
            # collapses (open_window) does not fire.
            _grip_now = float(row.get("grip_force", 0.0))
            _hold_thr = float((self.frozen_thr or {}).get("grip_force_threshold", 1.0))
            # Only track/detect inside the grasp window (close -> next open). A
            # drop after a commanded open (close_box) is a release, not a slip.
            # Only the grasp window gates detection. holding_required_phase used
            # to reset the peak too, which switched slip detection off entirely
            # at release waypoints (open_jar wp5) -- a slip injected there was
            # structurally undetectable even when it happened well before the
            # commanded open. _grasp_active alone is the correct guard: it closes
            # on mark_open_command(), so a drop after the commanded open is still
            # treated as a release, while a drop before it can fire.
            if not self._grasp_active:
                self._held_peak = 0.0
            elif _grip_now > _hold_thr:
                self._held_peak = max(self._held_peak, _grip_now)
            _grasp_est = (self._grasp_active
                          and self._held_peak > _hold_thr)   # a real grasp happened
            _drop_frac = float(os.getenv("AHA_SLIP_DROP_FRAC", "0.8"))
            _prop_collapse = bool(
                _grasp_est and self._held_peak > 0.0
                and _grip_now <= (1.0 - _drop_frac) * self._held_peak)
            row["held_peak"] = self._held_peak
            row["proportional_collapse"] = _prop_collapse
            row["prop_drop"] = (
                1.0 - _grip_now / self._held_peak if self._held_peak > 0.0 else 0.0)

            # Annotate force thresholds and score for diagnostics; the gate
            # counts consecutive proportional-collapse frames below.
            _, reason = det.check_slip(
                row, self.frozen_thr,
                required_prior_holding_steps=self.settings[
                    "required_prior_holding_steps"],
                hold_lost_grace_steps=self.settings["hold_lost_grace_steps"])
            if not report:
                self._candidate_active = False
                self._consec = 0
                self._latched = False
                return False
            _active = row["proportional_collapse"]
            # Count consecutive frames of the slip signal itself. Previously the
            # counter was reset to 1 on every frame where check_slip() returned
            # True, so under method 1 (where slip_started IS _active) it could
            # only advance once the hold was declared lost by the grip-force
            # threshold -- i.e. it counted "frames below the acquisition
            # threshold", not frames of proportional collapse. A slip that
            # leaves residual load on the gripper (a handle that stays in
            # contact) then pinned the counter at 1 forever and never fired.
            if _active:
                self._candidate_active = True
                self._consec += 1
            elif self._candidate_active:
                self._candidate_active = False
                self._consec = 0
                self._latched = False
            else:
                self._consec = 0
                if row.get("prior_holding_streak", 0) > 0:
                    self._latched = False

            required = max(1, int(self.settings["consecutive_slip_frames"]))
            raw_slip = self._candidate_active and self._consec >= required
            confirmed_now = raw_slip and not self._latched
            if raw_slip:
                self._latched = True

            fired = (confirmed_now and report
                     and waypoint not in self.fired_waypoints
                     and waypoint not in self._vlm_rejected)
            if fired:
                return self._fire(step, reason, row, waypoint)

            return False
        except Exception as exc:
            self.disabled_reason = f"runtime error: {exc}"
            print(f"  [detector:slip] disabled mid-run — {exc}")
            return False

    def _append_drift_row(self, waypoint, path_done, fired):
        row = self.logs[-1]
        thr = self.frozen_thr or {}
        try:
            gripper_open = float(getattr(self._last_obs_for_drift, "gripper_open",
                                         float('nan')))
        except (TypeError, ValueError):
            gripper_open = float('nan')
        # Continuous jaw-open fraction (0=closed, 1=open). obs.gripper_open is
        # binarized at >0.95, so it only flips when the jaws are nearly fully
        # open -- well after they start opening (and after the grip drops). The
        # raw get_open_amount() captures the start of the opening motion.
        gripper_open_amount = float('nan')
        try:
            gripper_open_amount = float(
                self.env_wrapper.robot.gripper.get_open_amount()[0])
        except Exception:
            pass
        open_cmd = 1 if self._open_cmd_pending else 0
        self._open_cmd_pending = False
        close_cmd = 1 if self._close_cmd_pending else 0
        self._close_cmd_pending = False
        self._drift_log_rows.append((
            self.task_name,
            "" if waypoint is None else int(waypoint),
            int(row.get("step", 0)),
            self.step_counter - 1,
            float(row.get("grip_force", float('nan'))),
            float(row.get("left_grip_force", float('nan'))),
            float(row.get("right_grip_force", float('nan'))),
            float(row.get("grip_force_drop", float('nan'))),
            float(row.get("grip_force_delta", float('nan'))),
            gripper_open,
            gripper_open_amount,
            open_cmd,
            close_cmd,
            float(row.get("held_peak", float('nan'))),
            float(row.get("prop_drop", float('nan'))),
            int(bool(row.get("is_holding", False))),
            int(bool(row.get("holding_required_phase", True))),
            int(row.get("prior_holding_streak", 0)),
            ("" if row.get("steps_since_holding") is None
             else int(row["steps_since_holding"])),
            int(bool(row.get("force_released", False))),
            int(bool(row.get("force_drop_crossed", False))),
            int(self._candidate_active),
            self._consec,
            float(row.get("slip_score", 0.0)),
            float(thr.get("grip_force_threshold",
                          self.settings["threshold_overrides"].get(
                              "grip_force_threshold", 1.0))),
            float(thr.get("grip_force_drop_threshold",
                          self.settings["threshold_overrides"].get(
                              "grip_force_drop_threshold", 1.0))),
            int(bool(path_done)),
            int(bool(fired)),
        ))
        if len(self._drift_log_rows) >= 100:
            self._flush_drift_log()

    def _flush_drift_log(self):
        if not self._drift_log_path or not self._drift_log_rows:
            return
        import csv as _csv
        path = self._drift_log_path
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            exists = os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = _csv.writer(f)
                if not exists:
                    w.writerow(['task', 'waypoint', 'local_step', 'step', 'grip_force', 'left_grip_force', 'right_grip_force', 'grip_force_drop', 'grip_force_delta', 'gripper_open', 'gripper_open_amount', 'gripper_open_cmd', 'gripper_close_cmd', 'held_peak', 'prop_drop', 'is_holding', 'holding_required_phase', 'prior_holding_streak', 'steps_since_holding', 'force_released', 'force_drop_crossed', 'slip_candidate_active', 'low_force_streak', 'slip_score', 'grip_force_threshold', 'grip_force_drop_threshold', 'path_done', 'slip_fired'])
                w.writerows(self._drift_log_rows)
            self._drift_log_rows = []
        except Exception as exc:
            print(f"  [slip] drift-log flush failed: {exc}")


    def _fire(self, step, reason, row, waypoint):
        'Record a detector candidate, then apply immediate VLM confirmation when enabled.'
        self.fire_count += 1
        self.fired_waypoints.add(waypoint)
        if self.first_fire is None:
            self.first_fire = (step, reason, waypoint)
        self._announce_detection(step, reason, row, waypoint)


        verdict = self._run_vlm_and_pause(row, waypoint)
        return self._apply_vlm_verdict(verdict, waypoint)

    def _announce_detection(self, step, reason, row, waypoint):
        print(
            f"\n  [DETECTOR] SLIP detected at waypoint {waypoint} (step {step})  "
            f"reason={reason}  grip={row['grip_force']:.3f}  "
            f"drop={row['grip_force_drop']:.3f}  "
            f"prior_hold={row.get('prior_holding_streak')}\n",
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
            print("  [detector:slip] detection RETRACTED — "
                  "VLM verifier says no slip.")
            self.fired_waypoints.discard(waypoint)
            self._vlm_rejected.add(waypoint)
            self.fire_count -= 1

            return False

        return True


    def _prompt_vlm(self, row, waypoint):
        vlm = self.vlm_mod
        obs_source = list(self.recent_obs)
        try:
            sampled_obs, step_offsets = vlm.sample_recent_observations(
                obs_source)

        except Exception as exc:
            print(f"  [vlm] image preview failed: {exc}")
            sampled_obs, step_offsets = list(self.recent_obs), None

        print("  [vlm] auto-confirming slip detection with the VLM "
              "verifier...")
        try:
            result = vlm.confirm_slip_with_openai(
                sampled_obs,
                step_offsets=step_offsets,
                row=row,
                telemetry_history=list(self.logs),
                env_wrapper=self.env_wrapper,
                task_name=self.task_name,
                waypoint_index=waypoint,
                waypoints_description_path=self.settings.get('waypoints_description_path'),
                model=self.vlm_model,
            )
        except Exception as exc:
            print(f"  [vlm] confirmation failed: {exc}")
            return None
        verdict = result.get("slip_happened")
        vtext = "YES" if verdict is True else "NO" if verdict is False else "UNKNOWN"
        print(f"  [vlm:{result.get('model')}] slip={vtext}  "
              f"{result.get('explanation', '')}  "
              f"cameras={', '.join(result.get('camera_names', []))}")


        return verdict


    def diagnostics_line(self):
        thr = self.frozen_thr or {}
        return (f"      diag: fed {self.fed} obs; max grip="
                f"{self.diag.get('max_grip', 0):.3f} (hold>"
                f"{thr.get('grip_force_threshold', 'n/a')}); holding obs "
                f"{self.diag.get('holding_steps', 0)}; max drop="
                f"{self.diag.get('max_drop', 0):.3f}")

    def close(self):
        self._flush_drift_log()
