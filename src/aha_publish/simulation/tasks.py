"""Task discovery shared by publication commands."""
import re
from aha_publish import paths

def available_tasks():
    return sorted(p.name.removesuffix('.bt_conditions.json') for p in paths.BT_DIR.glob('*.bt_conditions.json'))

def validate_task(name):
    if not re.fullmatch(r'[a-z][a-z0-9_]*', name):
        raise ValueError(f'Invalid task name: {name!r}; use the RLBench task name.')
    return name
