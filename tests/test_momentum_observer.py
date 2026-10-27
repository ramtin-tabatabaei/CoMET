"""Simulator-free observer regression tests: python -m unittest <module>."""
import os
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from aha_publish.detectors.collision import detector as det


class MomentumObserverTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {
            'AHA_COLLISION_MOMENTUM_VERSION': '1',
        })
        env.start()
        self.addCleanup(env.stop)
        # Unit inertia, zero gravity: prescribed momentum and actuator torque
        # give an independently known external torque.
        dynamics = patch.object(det, '_panda_dyn', SimpleNamespace(
            N=7, mass_matrix=lambda q: np.eye(7),
            gravity_torque=lambda q: np.zeros(7),
            coriolis_transpose_qd=lambda q, v: np.zeros(7),
        ))
        dynamics.start()
        self.addCleanup(dynamics.stop)

    @staticmethod
    def rows(n, moving=False):
        return [{
            'step': k, '_joint_positions': np.zeros(7),
            '_joint_velocities': np.full(7, k * .05 * 4 if moving else 0.),
            '_joint_forces': np.full(7, 0. if moving else -4.),
        } for k in range(n)]

    def test_step_response_static_and_moving(self):
        for moving in (False, True):
            logs = self.rows(30, moving)
            det.update_momentum_observer(logs, gain=25, dt=.05)
            actual = np.array([r['mo_residual'][0] for r in logs])
            expected = 4 * (1 - (-0.25) ** np.arange(30))
            np.testing.assert_allclose(actual, expected, atol=1e-12)
            self.assertLessEqual(np.abs(actual[-1] - 4), 1e-12)

    def test_incremental_trimmed_and_idempotent(self):
        full = self.rows(12, moving=True)
        det.update_momentum_observer(full)
        live = []
        for row, expected in zip(self.rows(12, moving=True), full):
            live.append(row)
            live = live[-3:]
            det.update_momentum_observer(live)
            det.update_momentum_observer(live)
            np.testing.assert_allclose(live[-1]['mo_residual'], expected['mo_residual'])

    def test_v1_preserved(self):
        with patch.dict(os.environ, {'AHA_COLLISION_MOMENTUM_VERSION': '1'}):
            logs = self.rows(4)
            det.update_momentum_observer(logs, gain=25, dt=.05)
            np.testing.assert_allclose([r['mo_residual'][0] for r in logs],
                                       [0, 5, 3.75, 4.0625])

    def test_calibration_parameters_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'task_residual_stats.json').write_text(json.dumps({
                'task': 'task', 'observer_version': '1',
                'gain': det.DEFAULT_MOMENTUM_GAIN, 'dt': det.DEFAULT_MOMENTUM_DT,
                'per_joint_max': [1.0] * 7,
            }))
            np.testing.assert_array_equal(det.load_residual_stats('task', directory), 1)
            with patch.object(det, 'DEFAULT_MOMENTUM_GAIN', 30):
                self.assertIsNone(det.load_residual_stats('task', directory))

    def test_invalid_parameters_and_version_change(self):
        for gain, dt in ((0, .05), (25, -1), (float('nan'), .05)):
            with self.assertRaises(ValueError):
                det.update_momentum_observer(self.rows(2), gain=gain, dt=dt)
        logs = self.rows(2)
        det.update_momentum_observer(logs)
        with patch.dict(os.environ, {'AHA_COLLISION_MOMENTUM_VERSION': '2'}):
            with self.assertRaisesRegex(ValueError, 'Only momentum observer version 1'):
                det.update_momentum_observer(logs)


if __name__ == '__main__':
    unittest.main()
