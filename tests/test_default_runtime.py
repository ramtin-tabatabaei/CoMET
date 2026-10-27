"""Default detector and export paths after removal of disabled features."""
import contextlib
import csv
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from aha_publish import paths
from aha_publish.scoring import make_detection_samples as samples
from aha_publish.running.run_support import parse_vlm_events
from aha_publish.detectors.collision import detector as collision_core
from aha_publish.detectors.slip import detector as slip_core
from aha_publish.running import bt_tree
from aha_publish.running import plot_maker
from aha_publish.running import bt_runner as runner
from aha_publish.running.live_detectors import LiveDetectorMonitor, collision, orientation, slip, transition


class DefaultRuntimeTests(unittest.TestCase):
    def test_move_leaf_executes_motion_for_clean_and_failure_runs(self):
        for failure_active in (False, True):
            with self.subTest(failure_active=failure_active):
                move = mock.Mock()
                ctx = SimpleNamespace(runner=SimpleNamespace(bt_run_move=move),
                                      failure_active=failure_active, failure_waypoint=0,
                                      monitor=SimpleNamespace(current_step=lambda: 0))
                waypoint, stage = object(), {}
                leaf = bt_tree.MovePath('move', ctx, 0, waypoint, stage)
                self.assertEqual(leaf.update(), bt_tree.Status.SUCCESS)
                move.assert_called_once_with(ctx, 0, waypoint, stage)

    def test_collision_uses_residual_band_instead_of_raw_torque(self):
        thresholds = {'residual_thr': 3.0, 'residual_thr_vec': np.full(7, 3.0)}
        row = {'mo_residual': np.zeros(7), 'residual_score': 0.0,
               'torque_norm': 10000.0, 'torque_delta': 10000.0}
        self.assertEqual(collision_core.check_collision(row, thresholds), (False, ''))
        row.update(mo_residual=np.array([0, 0, 4, 0, 0, 0, 0]), residual_score=4.0)
        self.assertEqual(collision_core.check_collision(row, thresholds), (True, 'momentum_residual'))

    def test_arrival_checks_use_saved_bands_and_do_not_fire_during_motion(self):
        reference = [0, 0, 0, 0, 0, 0, 1]
        cases = ((orientation, [0, 0, 0, 0, 0, math.sin(.3), math.cos(.3)]),
                 (transition, [.2, 0, 0, 0, 0, 0, 1]))
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(folder)
            (directory / 'task.json').write_text(json.dumps({
                'arrival_stats_by_waypoint': {'0': {'threshold': .1, 'median': 0, 'std': .01, 'max': .05}}
            }))
            for module, pose in cases:
                with self.subTest(detector=module.NAME), \
                     mock.patch.object(module, 'ARRIVAL_STATS_DIR', directory), \
                     mock.patch.object(module, 'VLM_AUTO', True):
                    detector = module.LiveDetector('task', vlm_enabled=False)
                    monitor = LiveDetectorMonitor.__new__(LiveDetectorMonitor)
                    monitor._pose_measurements = {}
                    monitor.detectors = {module.NAME: detector}
                    monitor.begin_waypoint(0)
                    self.assertAlmostEqual(detector._waypoint_threshold(0), max(.1, detector.arrival_thr))
                    observation = SimpleNamespace(gripper_pose=pose)
                    self.assertFalse(detector.step(observation, waypoint=0,
                                                   original_pose=reference, path_done=False))
                    self.assertTrue(detector.step(observation, waypoint=0,
                                                  original_pose=reference, path_done=True))
                    self.assertIsNone(detector.disabled_reason)
                    self.assertEqual(detector.fired_waypoints, {0})

    def test_all_four_default_telemetry_exports_render(self):
        reference = [0, 0, 0, 0, 0, 0, 1]
        observation = SimpleNamespace(gripper_pose=reference, gripper_open=1.0)
        dynamics = SimpleNamespace(N=7, mass_matrix=lambda q: np.eye(7),
                                   gravity_torque=lambda q: np.zeros(7),
                                   coriolis_transpose_qd=lambda q, v: np.zeros(7))
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
             mock.patch.object(collision_core, '_panda_dyn', dynamics):
            directory = Path(folder)
            drift = {}
            for module, variable in ((orientation, 'DRIFT_LOG_PATH'),
                                     (transition, 'DRIFT_LOG_PATH'),
                                     (slip, 'SLIP_DRIFT_LOG_PATH'),
                                     (collision, 'COLLISION_DRIFT_LOG_PATH')):
                filename = directory / f'{module.NAME}_drift.csv'
                drift[module.NAME] = str(filename)
                with mock.patch.object(module, variable, str(filename)):
                    detector = module.LiveDetector('synthetic_task', vlm_enabled=False)
                    for frame in range(7):
                        detector.step(observation, waypoint=0, original_pose=reference,
                                      path_done=frame == 6, report=False)
                    self.assertIsNone(detector.disabled_reason, module.NAME)
                    detector.close()
                with filename.open(newline='') as handle:
                    rows = list(csv.DictReader(handle))
                self.assertTrue(rows, module.NAME)
                for row in rows:
                    self.assertNotIn(None, row, module.NAME)
                    self.assertNotIn(None, row.values(), module.NAME)
            with mock.patch.object(paths, 'CALIBRATION_DIR', directory):
                plots = runner.render_plots(str(directory), 'synthetic_task', 'clean', drift, {})
            self.assertEqual(set(plots), set(runner.DETECTORS))
            for name, filename in plots.items():
                self.assertFalse(filename.startswith('plot_err:'), (name, filename))
                self.assertEqual(Path(filename).read_bytes()[:8], b'\x89PNG\r\n\x1a\n')

    def test_one_bad_trace_does_not_block_other_detector_plots(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            bad = directory / 'orientation_drift.csv'
            bad.write_text('waypoint,angle_to_target\n0,invalid\n')
            good = directory / 'transition_drift.csv'
            good.write_text('waypoint,dist_to_target,gripper_motion,path_done\n0,0.02,0.01,1\n')
            with mock.patch.object(paths, 'CALIBRATION_DIR', directory):
                result = plot_maker.render_plots(directory, 'task', 'clean', {
                    'orientation': str(bad), 'transition': str(good),
                }, {})
            self.assertTrue(result['orientation'].startswith('plot_err:'))
            self.assertEqual(Path(result['transition']).read_bytes()[:8], b'\x89PNG\r\n\x1a\n')

    def test_arrival_checks_keep_the_correct_reference_when_the_target_changes(self):
        reference = [0, 0, 0, 0, 0, 0, 1]
        cases = ((orientation, [0, 0, 0, 0, 0, math.sin(.3), math.cos(.3)]),
                 (transition, [.2, 0, 0, 0, 0, 0, 1]))
        with contextlib.redirect_stdout(io.StringIO()):
            for module, shifted_pose in cases:
                with self.subTest(detector=module.NAME), \
                     mock.patch.object(module, 'load_arrival_baseline', return_value={}), \
                     mock.patch.object(module, 'VLM_AUTO', True):
                    detector = module.LiveDetector('task', vlm_enabled=False)
                    observation = SimpleNamespace(gripper_pose=shifted_pose)
                    self.assertFalse(detector.step(observation, waypoint=0,
                                                   waypoint_pose=shifted_pose,
                                                   original_pose=reference))
                    self.assertTrue(detector.step(observation, waypoint=0,
                                                  waypoint_pose=shifted_pose,
                                                  original_pose=reference, path_done=True))
                    self.assertIsNone(detector.disabled_reason)

    def test_transition_keeps_the_last_motion_error_for_the_arrival_check(self):
        reference = [0, 0, 0, 0, 0, 0, 1]
        with contextlib.redirect_stdout(io.StringIO()), \
             mock.patch.object(transition, 'load_arrival_baseline', return_value={}), \
             mock.patch.object(transition, 'VLM_AUTO', True):
            detector = transition.LiveDetector('task', vlm_enabled=False)
            observation = SimpleNamespace(gripper_pose=[.2, 0, 0, 0, 0, 0, 1])
            self.assertFalse(detector.step(observation, waypoint=0, original_pose=reference))
            observation = SimpleNamespace(gripper_pose=reference)
            self.assertTrue(detector.step(observation, waypoint=0,
                                          original_pose=reference, path_done=True))
            self.assertAlmostEqual(detector._wp_arrival_dist[0], .2)
            self.assertIsNone(detector.disabled_reason)

    def test_slip_and_collision_keep_immediate_vlm_confirmation_and_retraction(self):
        dynamics = SimpleNamespace(N=7, mass_matrix=lambda q: np.eye(7),
                                   gravity_torque=lambda q: np.zeros(7),
                                   coriolis_transpose_qd=lambda q, v: np.zeros(7))
        with contextlib.redirect_stdout(io.StringIO()), \
             mock.patch.object(collision_core, '_panda_dyn', dynamics), \
             mock.patch.object(collision_core, 'load_residual_stats', return_value=None), \
             mock.patch.object(slip_core, 'apply_task_grip_force_stats_thresholds',
                               side_effect=lambda settings, task: settings), \
             mock.patch.object(slip_core, 'load_holding_requirement_from_waypoints',
                               return_value={'enabled': False}):
            for module in (slip, collision):
                for verdict in (True, False):
                    with self.subTest(detector=module.NAME, verdict=verdict):
                        confirm = mock.Mock(return_value={
                            f'{module.NAME}_happened': verdict, 'model': 'mock',
                            'explanation': 'Local fixture.', 'camera_names': [],
                        })
                        detector = module.LiveDetector('task', vlm_enabled=True)
                        detector.vlm_mod = SimpleNamespace(
                            sample_recent_observations=lambda observations: (observations, None),
                            **{f'confirm_{module.NAME}_with_openai': confirm})
                        if module is slip:
                            detector.mark_close_command()
                        results = []
                        for frame in range(10):
                            force = 5.0 if frame < 6 else .1
                            obs = SimpleNamespace(
                                gripper_touch_forces=[force / 2, 0, 0] * 2,
                                gripper_open=0.0, joint_positions=np.zeros(7),
                                joint_velocities=np.zeros(7),
                                joint_forces=np.full(7, 0.0 if frame < 5 else 4.0))
                            results.append(detector.step(obs, waypoint=0))
                        self.assertIsNone(detector.disabled_reason)
                        confirm.assert_called_once()
                        self.assertEqual(any(results), verdict)
                        self.assertEqual(detector.fired_waypoints, {0} if verdict else set())
                        self.assertEqual(detector.fire_count, int(verdict))
                        row = confirm.call_args.kwargs['row']
                        if module is slip:
                            self.assertTrue(row['proportional_collapse'])
                            self.assertGreater(row['held_peak'], row['grip_force'])
                        else:
                            self.assertGreater(row['residual_score'], 3.0)
                            self.assertEqual(len(row['mo_residual']), 7)

    def test_commanded_release_does_not_trigger_slip(self):
        with contextlib.redirect_stdout(io.StringIO()), \
             mock.patch.object(slip_core, 'apply_task_grip_force_stats_thresholds',
                               side_effect=lambda settings, task: settings), \
             mock.patch.object(slip_core, 'load_holding_requirement_from_waypoints',
                               return_value={'enabled': False}):
            detector = slip.LiveDetector('task', vlm_enabled=False)
            detector.mark_close_command()
            for frame in range(6):
                self.assertFalse(detector.step(
                    SimpleNamespace(gripper_touch_forces=[2.5, 0, 0] * 2), waypoint=0))
            detector.mark_open_command()
            for frame in range(4):
                self.assertFalse(detector.step(
                    SimpleNamespace(gripper_touch_forces=[.05, 0, 0] * 2), waypoint=0))
            self.assertIsNone(detector.disabled_reason)
            self.assertEqual(detector.fire_count, 0)

    def test_crashed_episode_preserves_log_and_does_not_return_success(self):
        process = SimpleNamespace(returncode=1, stdout='waypoint started', stderr='Traceback: crash')
        with tempfile.TemporaryDirectory() as folder, \
             mock.patch.object(runner.subprocess, 'run', return_value=process):
            with self.assertRaisesRegex(RuntimeError, 'exit status 1'):
                runner.run_episode('task', 'none', None, folder)
            self.assertIn('Traceback: crash', (Path(folder) / 'run.log').read_text())

    def test_merged_batch_launches_single_episode_and_collects_results(self):
        output = '\n'.join(f'  {name} ok — no {name}' for name in runner.DETECTORS)
        process = SimpleNamespace(returncode=0, stdout=output, stderr='')
        with tempfile.TemporaryDirectory() as folder, \
             mock.patch.object(runner.subprocess, 'run', return_value=process) as execute, \
             mock.patch.object(runner.EV, 'config_failtypes_by_wp', return_value={}), \
             contextlib.redirect_stdout(io.StringIO()):
            runner.main(['batch', '--out', folder, '--task', 'example_task',
                         '--failures', 'none', '--workers', '1'])
            execute.assert_called_once()
            command = execute.call_args.args[0]
            self.assertEqual(Path(command[1]).resolve(), Path(runner.__file__).resolve())
            self.assertEqual(command[2], 'single')
            arguments = runner.parse_args(command[3:])
            self.assertEqual(arguments.task, 'example_task')
            self.assertEqual(arguments.failure, 'none')
            with (Path(folder) / 'results.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row['detector'] for row in rows}, set(runner.DETECTORS))
            self.assertEqual({row['outcome'] for row in rows}, {'TN'})
            self.assertTrue((Path(folder) / 'example_task/clean/run.log').is_file())
            self.assertTrue((Path(folder) / 'summary.csv').is_file())

    def test_default_scoring_reads_condition_failures_from_run_log(self):
        text = """Waypoint 0 of 1
Waypoint 1 of 1
  - selected_object(cup) == True  [WrongObjectSelection]
[vlm] checking preconditions for waypoint 1...
  VLM PRE CHECK: FAIL
    - not_satisfied: selected_object(cup) == True
"""
        with tempfile.TemporaryDirectory() as folder:
            case = Path(folder) / 'wrong_object_wp1'
            case.mkdir()
            (case / 'run.log').write_text(text)
            episode = samples.load_episode('task', case)
            self.assertFalse((case / 'vlm_events.csv').exists())
            self.assertEqual(samples.method_label(episode, 'A1', 1), 'failure,WrongObjectSelection')
            self.assertEqual(samples.method_label(episode, 'A4', 1), 'failure,WrongObjectSelection')

    def test_default_scoring_reads_detector_confirmation_from_run_log(self):
        text = """Waypoint 0 of 1
Waypoint 1 of 1
[DETECTOR] COLLISION detected at waypoint 1 (step 20)
  [vlm:test-model] collision=YES  Contact visible.  cameras=side_rgb
"""
        with tempfile.TemporaryDirectory() as folder:
            case = Path(folder) / 'clean'
            case.mkdir()
            log = case / 'run.log'
            log.write_text(text)
            episode = samples.load_episode('task', case)
            self.assertEqual(samples.method_label(episode, 'A3', 1), 'failure,collision')
            log.write_text(text.replace('collision=YES', 'collision=NO'))
            episode = samples.load_episode('task', case)
            self.assertEqual(samples.method_label(episode, 'A3', 1), 'success')
            self.assertEqual(samples.method_label(episode, 'A2', 1), 'failure,collision')

    def test_combined_checkpoint_and_simulator_pass_do_not_duplicate_events(self):
        events = parse_vlm_events("""[vlm] checking preconditions for waypoint 2 and postconditions for waypoint 1...
  VLM PRE+POST SIMULATOR CHECK: PASS
  VLM PRE+POST CHECK: PASS
""")
        self.assertEqual(events, [{'channel': 'condition', 'event_kind': 'prepost',
                                   'waypoint': 2, 'verdict': 'PASS'}])

    def test_default_checkpoint_vlm_request_still_sends_images_and_parses_verdict(self):
        verdict = {'checkpoint_passed': True, 'conditions': [
            {'condition': 'inside(ball, hoop)', 'status': 'satisfied', 'evidence': 'Visible inside.'}],
            'explanation': 'The condition holds.'}
        response = SimpleNamespace(output_text=json.dumps(verdict),
                                   usage=SimpleNamespace(input_tokens=10, output_tokens=5))
        create = mock.Mock(return_value=response)
        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        helpers = SimpleNamespace(collect_camera_images=lambda *a, **k: [('side_rgb', 'data:image/png;base64,AA==')])
        with mock.patch.object(runner, 'make_openai_client', return_value=client), \
             mock.patch.object(runner, 'load_vlm_helpers', return_value=helpers), \
             mock.patch.object(runner, 'VLM_USAGE', {'calls': 0, 'input_tokens': 0, 'output_tokens': 0}), \
             contextlib.redirect_stdout(io.StringIO()):
            result = runner.run_vlm_condition_check(None, None, None, 'task', 0, {}, 'post',
                                                    ['inside(ball, hoop)'], 'test-model')
        self.assertTrue(result['checkpoint_passed'])
        content = create.call_args.kwargs['input'][1]['content']
        self.assertTrue(any(part['type'] == 'input_image' for part in content))
        self.assertEqual(result['explanation'], verdict['explanation'])


if __name__ == '__main__':
    unittest.main()
