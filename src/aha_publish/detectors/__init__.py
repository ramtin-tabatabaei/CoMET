"""Failure detectors and their configured primitive assignments."""

import json

from aha_publish import paths


PRIMITIVE_POLICY_PATH = paths.PROJECT_ROOT / 'config/detector_primitives.json'


def load_primitive_policy():
    """Read detector primitive assignments; an empty list disables that detector."""
    try:
        policy = json.loads(PRIMITIVE_POLICY_PATH.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid JSON in {PRIMITIVE_POLICY_PATH}: {exc}') from exc
    if not isinstance(policy, dict):
        raise ValueError(f'{PRIMITIVE_POLICY_PATH} must map detector names to primitive lists')
    detectors = {'slip', 'collision', 'orientation', 'transition'}
    primitives = {'move', 'grasp', 'transport', 'place', 'release', 'push'}
    for name, selected in policy.items():
        if name not in detectors:
            raise ValueError(f'Unknown detector {name!r} in {PRIMITIVE_POLICY_PATH}')
        if (not isinstance(selected, list)
                or any(not isinstance(value, str) or value not in primitives
                       for value in selected)):
            raise ValueError(f'{PRIMITIVE_POLICY_PATH}: {name} must list primitives from '
                             f'{", ".join(sorted(primitives))}')
    return {name: set(selected) for name, selected in policy.items()}
