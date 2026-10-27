"""Calibration validation, collection failures, and baseline consistency."""
from contextlib import redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from aha_publish.calibration import artifacts, calibrate_all_clean as collector
from aha_publish.calibration import compute_thresholds, settings


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.config = json.loads(settings.DEFAULT_SETTINGS.read_text())

    def raw_csv(self, rows):
        filename = self.root / 'example_task/ep0.csv'
        filename.parent.mkdir(parents=True, exist_ok=True)
        with filename.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                'waypoint', 'local_step', 'path_done', 'distance_m', 'angle_rad',
                'torque_norm', 'torque_delta', 'grip_force', 'grip_force_drop',
            ])
            writer.writeheader()
            writer.writerows(rows)
        return filename

    def test_default_file_is_validated_against_the_expected_schema(self):
        del self.config['transition']['arrival_scale']
        filename = self.root / 'thresholds.json'
        filename.write_text(json.dumps(self.config))
        with mock.patch.object(settings, 'DEFAULT_SETTINGS', filename):
            with self.assertRaisesRegex(ValueError, 'missing settings in transition'):
                settings.load_settings(filename)

    def test_raw_data_requires_in_motion_pose_measurements(self):
        row = dict(waypoint=0, local_step=0, path_done=1, distance_m=0.1,
                   angle_rad=0.2, torque_norm=2, torque_delta=0,
                   grip_force=1, grip_force_drop=0)
        self.raw_csv([row])
        with self.assertRaisesRegex(ValueError, 'no in-motion measurements'):
            artifacts.validate_raw(self.root, 'example_task')
        self.raw_csv([{**row, 'path_done': 0}])
        self.assertEqual(len(artifacts.validate_raw(self.root, 'example_task')), 1)

    def test_truncated_raw_rows_are_rejected_before_threshold_computation(self):
        row = dict(waypoint=0, local_step=0, path_done=0, distance_m=0.1,
                   angle_rad=0.2, torque_norm=2, torque_delta=0,
                   grip_force=1, grip_force_drop=0)
        filename = self.raw_csv([row])
        with filename.open('a') as stream:
            stream.write('0,1,0,0.1\n')
        with self.assertRaisesRegex(ValueError, 'malformed clean-data row'):
            artifacts.validate_raw(self.root, 'example_task')

    def collector_main(self, tasks, workers=1, **patches):
        previous_root = collector.CALIBRATION_ROOT
        self.addCleanup(collector.configure_output_root, previous_root)
        arguments = ['collect', '--episodes', '1', '--out-root', str(self.root),
                     '--workers', str(workers)]
        for task in tasks:
            arguments += ['--task', task]
        with mock.patch('sys.argv', arguments), \
                mock.patch.object(collector, 'configure_env'), \
                redirect_stdout(StringIO()):
            with mock.patch.multiple(collector, **patches):
                return collector.main()

    def test_existing_episode_count_does_not_skip_incomplete_collection(self):
        run = mock.Mock(return_value=dict(episodes_completed=1))
        result = self.collector_main(['example_task'],
            clean_episode_count=mock.Mock(return_value=1),
            task_done=mock.Mock(return_value=False), run_task=run)
        self.assertEqual(result, 0)
        run.assert_called_once()

    def test_complete_collection_is_reused_without_rerunning_the_simulator(self):
        run = mock.Mock()
        result = self.collector_main(['example_task'],
            clean_episode_count=mock.Mock(return_value=1),
            task_done=mock.Mock(return_value=True), run_task=run)
        self.assertEqual(result, 0)
        run.assert_not_called()

    def test_failed_collection_returns_failure_and_records_the_task(self):
        result = self.collector_main(['example_task'],
            clean_episode_count=mock.Mock(return_value=0),
            run_task=mock.Mock(side_effect=RuntimeError('No successful demo')))
        self.assertEqual(result, 1)
        failures = json.loads((self.root / 'all_clean_calibration_failed_tasks.json').read_text())
        self.assertEqual(failures, [{'task': 'example_task', 'error': 'No successful demo'}])

    def test_failed_parallel_collection_returns_failure(self):
        result = self.collector_main(['first_task', 'second_task'], workers=2,
            clean_episode_count=mock.Mock(return_value=0),
            run_one_subprocess=mock.Mock(return_value=(1, 0.1)))
        self.assertEqual(result, 1)
        failures = json.loads((self.root / 'all_clean_calibration_failed_tasks.json').read_text())
        self.assertEqual({row['task'] for row in failures}, {'first_task', 'second_task'})

    def configure_collector(self):
        previous_root = collector.CALIBRATION_ROOT
        self.addCleanup(collector.configure_output_root, previous_root)
        collector.configure_output_root(self.root)

    def test_recollection_invalidates_the_old_manifest_before_replacing_data(self):
        self.configure_collector()
        manifest = self.root / 'manifests/example_task.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text('{"old": true}')
        backend = SimpleNamespace(FailGenEnvWrapper=mock.Mock(side_effect=RuntimeError('Simulator unavailable')))
        with mock.patch.dict('sys.modules', {'failgen.env_wrapper': backend}), \
                mock.patch.object(collector, 'configure_env'):
            with self.assertRaisesRegex(RuntimeError, 'Simulator unavailable'):
                collector.run_task('example_task', 1, 1)
        self.assertFalse(manifest.exists())

    def test_partial_collection_does_not_export_a_complete_baseline(self):
        self.configure_collector()
        environment = SimpleNamespace(
            on_env_step=mock.Mock(), on_env_waypoint=mock.Mock(),
            on_env_waypoint_end=mock.Mock(), reset=mock.Mock(), shutdown=mock.Mock(),
            get_success=mock.Mock(side_effect=[object(), None, None, None, None, None]),
        )
        backend = SimpleNamespace(FailGenEnvWrapper=mock.Mock(return_value=environment))
        with mock.patch.dict('sys.modules', {'failgen.env_wrapper': backend}), \
                mock.patch.object(collector, 'configure_env'), \
                mock.patch.object(collector, 'write_raw_csv'), \
                mock.patch.object(collector.residuals, 'write_episode'), \
                mock.patch.object(collector, 'write_json') as baseline, \
                redirect_stdout(StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'only 1/2 clean episodes'):
                collector.run_task('example_task', 2, 1)
        baseline.assert_not_called()
        environment.shutdown.assert_called_once()

    def test_orientation_uses_the_worst_clean_arrival_while_transition_uses_median(self):
        args = SimpleNamespace(**{
            prefix + '_' + key: value
            for group, prefix in [('transition', 'trans'), ('orientation', 'ori')]
            for key, value in self.config[group].items()
        })
        trans_params, ori_params = compute_thresholds.build_params(args)
        episodes = [[{'waypoint': '0', 'local_step': '0', 'path_done': '0',
                      'distance_m': str(value), 'angle_rad': str(value)}]
                    for value in (0.2, 0.3, 1.0)]
        trans = compute_thresholds.waypoint_thresholds(episodes, 'distance_m', trans_params)
        ori = compute_thresholds.waypoint_thresholds(episodes, 'angle_rad', ori_params)
        self.assertEqual(trans['0']['threshold'], 0.39)
        self.assertEqual(ori['0']['threshold'], 1.3)

    def test_completion_requires_raw_data_and_matching_residuals(self):
        self.configure_collector()
        row = dict(waypoint=0, local_step=0, path_done=0, distance_m=0.1,
                   angle_rad=0.2, torque_norm=2, torque_delta=0,
                   grip_force=1, grip_force_drop=0)
        filename = self.raw_csv([row])
        target = collector.RAW_DIR / 'example_task/ep0.csv'
        target.parent.mkdir(parents=True)
        target.write_bytes(filename.read_bytes())
        self.assertFalse(collector.task_done('example_task', 1))
        parameters = collector.residuals.observer_parameters(collector.CD)
        target.with_suffix('.residual.json').write_text(json.dumps(dict(
            parameters, task='example_task', episode=0, per_joint_max=[1] * 7)))
        self.assertTrue(collector.task_done('example_task', 1))
        self.assertFalse(collector.task_done('example_task', 2))
        parameters['gain'] += 1
        target.with_suffix('.residual.json').write_text(json.dumps(dict(
            parameters, task='example_task', episode=0, per_joint_max=[1] * 7)))
        self.assertFalse(collector.task_done('example_task', 1))

    def test_collision_method_selector_is_rejected_by_the_config_schema(self):
        for method in (1, 2, 3):
            self.config['collision']['method'] = method
            filename = self.root / 'thresholds.json'
            filename.write_text(json.dumps(self.config))
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, 'Unexpected or missing settings in collision'):
                settings.load_settings(filename)

    def test_runtime_rejects_manifests_from_removed_collision_methods(self):
        manifest = self.root / 'manifests/example_task.json'
        manifest.parent.mkdir(parents=True)
        for method in ('1', '2'):
            manifest.write_text(json.dumps({'runtime_environment': {'AHA_COLLISION_METHOD': method}}))
            with self.subTest(method=method), \
                    mock.patch.object(settings.paths, 'CALIBRATION_DIR', self.root), \
                    self.assertRaisesRegex(ValueError, 'recalibrate with method 3'):
                settings.runtime_environment({}, ['example_task'])

    def test_current_and_legacy_de_luca_manifests_need_no_method_selector(self):
        manifest = self.root / 'manifests/example_task.json'
        manifest.parent.mkdir(parents=True)
        expected = settings.detector_environment(self.config)
        for legacy in (False, True):
            values = dict(expected)
            if legacy:
                values['AHA_COLLISION_METHOD'] = '3'
            manifest.write_text(json.dumps({'runtime_environment': values, 'artifacts': []}))
            with self.subTest(legacy=legacy), \
                    mock.patch.object(settings.paths, 'CALIBRATION_DIR', self.root):
                self.assertEqual(settings.runtime_environment({}, ['example_task']), expected)

    def test_raw_writer_preserves_missing_measurements_and_frame_alignment(self):
        filename = self.root / 'example_task/ep0.csv'
        pose = [dict(global_step=1, waypoint=4, local_step=0, path_done=0,
                     distance_m=0.2, angle_rad=0.3),
                dict(global_step=2, waypoint=4, local_step=1, path_done=1,
                     distance_m=0.1, angle_rad=0.2)]
        grip = [dict(step=0, grip_force=0, grip_force_drop=0),
                dict(step=1, grip_force=5, grip_force_drop=1)]
        collector.write_raw_csv(filename, 'example_task', 0, pose, grip)
        with filename.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row['global_step'] for row in rows], ['0', '1', '2'])
        self.assertEqual(rows[0]['distance_m'], '')
        self.assertEqual((rows[1]['waypoint'], rows[1]['grip_force']), ('4', '5'))
        self.assertEqual(rows[2]['grip_force'], '')
        arrivals = compute_thresholds.per_waypoint_arrivals([rows], 'distance_m')
        self.assertEqual(arrivals, {4: [0.2]})


if __name__ == '__main__':
    unittest.main()
