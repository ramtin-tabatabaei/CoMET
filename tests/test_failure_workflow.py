"""Failure selection and scoring for the supported detector bank."""
import json
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from aha_publish.behavior_trees import failure_definitions
from aha_publish import detectors
from aha_publish.simulation.failures import get_available_failures
from aha_publish.running import live_detectors
from aha_publish.running import bt_runner as runner
from aha_publish.scoring import make_detection_samples as samples


class FailureWorkflowTests(unittest.TestCase):
    def test_grasp_loss_is_assigned_only_to_configured_primitives(self):
        primitives = ['move', 'grasp', 'transport', 'place', 'release', 'push', None]
        stages = [{'primitive': primitive} for primitive in primitives]
        failure_definitions.attach_hold_conditions(stages)
        for stage in stages:
            with self.subTest(primitive=stage['primitive']):
                names = live_detectors.enabled_detectors_for_stage(stage)
                expected = {'collision', 'orientation', 'transition'}
                if stage['primitive'] in {'transport', 'place', 'release'}:
                    expected.add('slip')
                self.assertEqual(names, expected)

    def test_config_edits_update_generated_reopened_and_runtime_stages(self):
        stages = [{'stage': i, 'primitive': primitive}
                  for i, primitive in enumerate(['move', 'grasp', 'transport', 'release'])]
        failure_definitions.attach_hold_conditions(stages)
        with tempfile.TemporaryDirectory() as folder:
            policy_path = Path(folder) / 'policy.json'
            tree_path = Path(folder) / 'tree.json'
            tree_path.write_text(json.dumps({'review': {'stages': stages}}))
            saved_bytes = tree_path.read_bytes()
            with mock.patch.object(detectors, 'PRIMITIVE_POLICY_PATH', policy_path):
                for selected in (['grasp'], [], ['release', 'move']):
                    with self.subTest(selected=selected):
                        policy_path.write_text(json.dumps({'slip': selected}))
                        generated = failure_definitions.attach_hold_conditions(deepcopy(stages))
                        reopened = [dict(stage, hold_conditions=
                                         failure_definitions.hold_conditions_for_saved_stage(stage))
                                    for stage in stages]
                        _, _, runtime = runner.load_bt_stages(tree_path)
                        for tree in (generated, reopened, runtime):
                            for stage in tree:
                                self.assertEqual('slip' in live_detectors.enabled_detectors_for_stage(stage),
                                                 stage['primitive'] in selected)
                        self.assertEqual(tree_path.read_bytes(), saved_bytes)

    def test_unconfigured_detectors_keep_saved_removals_and_deselections(self):
        stage = {'primitive': 'transport', 'hold_conditions': [
            {'detector': 'collision', 'selected': False}, {'detector': 'transition'}]}
        blocks = failure_definitions.hold_conditions_for_saved_stage(stage)
        self.assertEqual({block['detector'] for block in blocks}, {'slip', 'transition'})

    def test_invalid_primitive_config_fails_before_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            policy_path = Path(folder) / 'policy.json'
            with mock.patch.object(detectors, 'PRIMITIVE_POLICY_PATH', policy_path):
                for invalid in ({'slip': ['tranport']}, {'slip': 'transport'},
                                {'unknown': []}, {'slip': [None]}, []):
                    with self.subTest(invalid=invalid):
                        policy_path.write_text(json.dumps(invalid))
                        with self.assertRaises(ValueError):
                            detectors.load_primitive_policy()

    def test_explicit_empty_detector_map_does_not_enable_all_detectors(self):
        with mock.patch.object(live_detectors, '_load_live_detector_class') as load:
            monitor = live_detectors.LiveDetectorMonitor('task', enabled_map={0: set()})
        load.assert_not_called()
        self.assertEqual(monitor.active_at(0), set())

    def test_grip_baseline_survives_excluded_grasp_without_consuming_a_detection(self):
        stages = [{'stage': 0, 'primitive': 'grasp'}, {'stage': 1, 'primitive': 'transport'},
                  {'stage': 2, 'primitive': 'release'}]
        failure_definitions.attach_hold_conditions(stages)
        enabled = {i: live_detectors.enabled_detectors_for_stage(stage)
                   for i, stage in enumerate(stages)}
        with redirect_stdout(StringIO()), \
             mock.patch.object(live_detectors, 'DEFAULT_LIVE_DETECTORS', ('slip',)):
            monitor = live_detectors.LiveDetectorMonitor(
                'synthetic_task', enabled_map=enabled, vlm_enabled=False)
            slip = monitor.detectors['slip']
            slip.phase_info = {'enabled': False}
            slip.settings['threshold_overrides']['grip_force_threshold'] = .1
            slip.settings['consecutive_slip_frames'] = 2
            high = SimpleNamespace(gripper_touch_forces=[5, 0, 0] * 2)
            low = SimpleNamespace(gripper_touch_forces=[.01, 0, 0] * 2)
            with mock.patch.object(slip, '_run_vlm_and_pause', return_value=True) as confirm:
                monitor.note_gripper_close_command()
                for _ in range(slip.WARMUP + 2):
                    monitor.on_step(high, 0, False)
                for _ in range(3):
                    monitor.on_step(low, 0, False)
                self.assertEqual(slip.fire_count, 0)
                confirm.assert_not_called()
                monitor.on_step(low, 1, False)
                self.assertEqual(slip.fire_count, 0)
                monitor.on_step(low, 1, False)
                self.assertEqual(slip.fired_waypoints, {1})
                confirm.assert_called_once()
                monitor.note_gripper_open_command()
                for _ in range(3):
                    monitor.on_step(low, 2, False, phase='gripper')
                self.assertEqual(slip.fired_waypoints, {1})
            self.assertIsNone(slip.disabled_reason)

    def test_failure_options_only_include_configured_non_dummy_entries(self):
        collision = {'type': 'collision', 'waypoints': [0, 1]}
        slip = {'type': 'slip', 'waypoints': [1]}
        config = {'data': {'waypoints': [0, 1]},
                  'failures': [collision, {'type': 'dummy'}, slip]}
        self.assertEqual(get_available_failures(config), [collision, slip])
        self.assertEqual(get_available_failures({}), [])

    def test_generated_holds_and_live_bank_have_the_same_four_detectors(self):
        failure_definitions.load_failure_definitions()
        blocks = failure_definitions.build_hold_condition_blocks()
        expected = {'collision', 'slip', 'orientation', 'transition'}
        self.assertEqual({block['detector'] for block in blocks}, expected)
        self.assertEqual({live_detectors.detector_for_hold_block(block) for block in blocks}, expected)
        self.assertEqual(set(runner.DETECTORS), expected)
        self.assertEqual(set(samples.DETECTORS), expected)

    def test_case_enumeration_preserves_supported_injections_at_the_last_waypoint(self):
        config = {'failures': [{'type': 'collision', 'waypoints': [0, 3]},
                               {'type': 'slip', 'waypoints': [3]},
                               {'type': 'grasp', 'waypoints': [1]}]}
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'example_task.yaml').write_text(json.dumps(config))
            with mock.patch.object(runner.EV, 'CFG_DIR', folder):
                self.assertEqual(runner.enumerate_cases('example_task'), [
                    ('clean', 'none', None, ''),
                    ('collision_wp0', 'collision', 0, 'collision'),
                    ('collision_wp3', 'collision', 3, 'collision'),
                    ('slip_wp3', 'slip', 3, 'slip'),
                ])

    def test_only_supported_detector_events_become_scoring_alarms(self):
        text = ('[DETECTOR] FREEZING detected at waypoint 2 (step 10)\n'
                '[DETECTOR] COLLISION failure at waypoint 2 (step 12)\n')
        fires = samples.parse_detector_fires(text)
        self.assertEqual(fires['collision'], {2: 12})
        self.assertEqual(set(fires), {'collision', 'slip', 'orientation', 'transition'})

    def test_supported_detector_scoring_and_episode_evidence_stay_consistent(self):
        info = {'fired': True, 'fired_waypoints': [3]}
        self.assertEqual(runner.score('collision', info, False, 'collision', 3), 'TP')
        self.assertEqual(runner.score('orientation', info, False, 'collision', 3), 'CF')
        self.assertEqual(runner.score('collision', info, True, '', None), 'FP')
        self.assertEqual(runner.episode_extras(False, 'transition', 3,
                                            'finished waypoint 3 path', {}), (True, None))
        self.assertEqual(samples.remap_types('failure,ObjectNotFound||ObjectNotFound'),
                         'failure,Wrong Sequence')


if __name__ == '__main__':
    unittest.main()
