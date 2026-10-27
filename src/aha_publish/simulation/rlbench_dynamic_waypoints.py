#!/usr/bin/env python3
from __future__ import annotations

from aha_publish import paths


import ast


import os
import re

from pathlib import Path

HERE = (paths.SOURCE_DIR)

# The task sources. RLBench is used from its own checkout, not vendored under
# aha/Data_Generation (which carries only backend/, not tasks/).
DEFAULT_TASKS_DIR = Path(
    os.getenv("AHA_RLBENCH_TASKS_DIR")
    or (paths.RLBENCH_ROOT / "rlbench/tasks"))

DEFAULT_REPORT_DIR = (paths.TTM_CONTEXT_DIR)

WAYPOINT_NAME = re.compile(r"^waypoint(\d+)$")

# Pose writes. set_parent is included: re-parenting invalidates a stored
# parent-local offset just as thoroughly as moving the object does.
SETTERS = frozenset({
    "set_pose", "set_position", "set_orientation", "set_quaternion",
    "set_matrix", "set_parent",
})

# The hook registrars. *_start and *_end both run while the episode is live.
ABILITY_REGISTRARS = frozenset({
    "register_waypoint_ability_start", "register_waypoint_ability_end",
})

REPEAT_REGISTRAR = "register_waypoints_should_repeat"


def _attr_path(node):
    """Dotted source text of a Name/Attribute chain, or None.

    ``self.waypoint1`` -> 'self.waypoint1', ``w2`` -> 'w2'. Anything else
    (subscripts, calls) is not a stable binding name and returns None.
    """
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _dummy_waypoint_index(node):
    """Index behind ``Dummy('waypointN')``, else None."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    if name != "Dummy" or not node.args:
        return None
    arg = node.args[0]
    if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
        return None
    match = WAYPOINT_NAME.match(arg.value)
    return int(match.group(1)) if match else None


def _setter_target(node):
    """For a call like ``X.set_pose(...)``, the dotted receiver X, else None.

    ``X.get_waypoint_object().set_position(...)`` unwraps to X as well: the
    ability hooks reach the dummy through that accessor.
    """
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr not in SETTERS:
        return None
    receiver = node.func.value
    if (isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Attribute)
            and receiver.func.attr == "get_waypoint_object"):
        receiver = receiver.func.value
    return _attr_path(receiver)


class _TaskAnalyzer(ast.NodeVisitor):
    """Collects waypoint bindings, setter writes and hook registrations."""

    def __init__(self):
        # dotted name -> waypoint index, for `self.x = Dummy('waypointN')`.
        # Attributes outlive the function that bound them.
        self.attr_bindings: dict[str, int] = {}
        # (function name, local name) -> index, for `w2 = Dummy('waypointN')`.
        self.local_bindings: dict[tuple[str, str], int] = {}
        # (function name, receiver) for every setter call seen.
        self.setter_calls: list[tuple[str, str]] = []
        # function name -> waypoint index it is registered as a hook for.
        self.hook_index: dict[str, int] = {}
        # hook functions that write through their own `waypoint` parameter.
        self.hooks_writing_self: set[str] = set()
        self.repeats = False
        self._func: list[str] = []

    # -- structure ---------------------------------------------------------
    def visit_FunctionDef(self, node):
        self._func.append(node.name)
        # The hook parameter is whatever the function's first non-self arg is
        # called; RLBench uses `waypoint` or `_`.
        args = [a.arg for a in node.args.args if a.arg != "self"]
        param = args[0] if args else None
        for sub in ast.walk(node):
            target = _setter_target(sub)
            if target is not None and param and target == param:
                self.hooks_writing_self.add(node.name)
        self.generic_visit(node)
        self._func.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    # -- bindings ----------------------------------------------------------
    def visit_Assign(self, node):
        index = _dummy_waypoint_index(node.value)
        if index is not None:
            for target in node.targets:
                name = _attr_path(target)
                if name is None:
                    continue
                if "." in name:
                    self.attr_bindings[name] = index
                elif self._func:
                    self.local_bindings[(self._func[-1], name)] = index
        self.generic_visit(node)

    # -- writes and registrations -----------------------------------------
    def visit_Call(self, node):
        target = _setter_target(node)
        if target is not None:
            self.setter_calls.append((self._func[-1] if self._func else "", target))

        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name in ABILITY_REGISTRARS and len(node.args) >= 2:
            index_arg, fn_arg = node.args[0], node.args[1]
            if isinstance(index_arg, ast.Constant) and isinstance(index_arg.value, int):
                fn_name = fn_arg.attr if isinstance(fn_arg, ast.Attribute) else getattr(
                    fn_arg, "id", None)
                if fn_name:
                    self.hook_index[fn_name] = index_arg.value
        elif name == REPEAT_REGISTRAR:
            self.repeats = True
        self.generic_visit(node)


def analyze_source(source: str, waypoint_count: int | None = None) -> dict:
    """`{'dynamic': [idx...], 'repeats': bool, 'reasons': {idx: reason}}`.

    `waypoint_count` is only needed to expand the repeat rule, which taints
    every waypoint rather than a particular one.
    """
    tree = ast.parse(source)
    analyzer = _TaskAnalyzer()
    analyzer.visit(tree)

    reasons: dict[int, str] = {}
    runtime_dynamic = set()

    # Rule 1 -- a setter called on something bound to a waypoint dummy.
    for func, receiver in analyzer.setter_calls:
        index = analyzer.attr_bindings.get(receiver)
        if index is None:
            index = analyzer.local_bindings.get((func, receiver))
        if index is None:
            continue
        # Only direct initialization writes can be explained by anchors inferred
        # across resets. Helpers/hooks may also run during execution: abstain.
        if func not in ('init_task', 'init_episode'):
            runtime_dynamic.add(index)
        reasons.setdefault(
            index, f"{func or '<module>'}() calls a setter on {receiver}")

    # Rule 2 -- an ability hook writing through its own waypoint parameter.
    for fn_name in analyzer.hooks_writing_self:
        index = analyzer.hook_index.get(fn_name)
        if index is not None:
            runtime_dynamic.add(index)
            reasons.setdefault(
                index, f"ability hook {fn_name}() writes its own waypoint")

    # Rule 3 -- a repeating waypoint sequence taints the whole sequence.
    if analyzer.repeats and waypoint_count:
        for index in range(waypoint_count):
            runtime_dynamic.add(index)
            reasons.setdefault(
                index, f"{REPEAT_REGISTRAR}: the sequence repeats with new poses")

    return {
        "dynamic": sorted(reasons),
        "runtime_dynamic": sorted(runtime_dynamic),
        "repeats": analyzer.repeats,
        "reasons": reasons,
    }


def analyze_task(task_name: str, tasks_dir=None, waypoint_count=None) -> dict | None:
    """Analyze one task by name, or None when its source is not on disk."""
    base = Path(tasks_dir or DEFAULT_TASKS_DIR)
    path = base / f"{task_name}.py"
    if not path.is_file():
        return None
    result = analyze_source(path.read_text(encoding="utf-8"), waypoint_count)
    result["source"] = str(path)
    return result


# --------------------------------------------------------------------------- #
# report stamping
# --------------------------------------------------------------------------- #


def unresolved_dynamic_waypoints(result, report):
    """Validated reset anchors explain initialization writes, never live hooks."""
    validated = {int(WAYPOINT_NAME.match(w['name']).group(1))
                 for w in report.get('waypoints', [])
                 if WAYPOINT_NAME.match(w.get('name', ''))
                 and w.get('anchor_validation_episodes', 0) >= 3
                 and w.get('position_parent') and w.get('orientation_parent')
                 and not w.get('reference_unavailable')}
    return sorted((set(result['dynamic']) - validated)
                  | set(result.get('runtime_dynamic', result['dynamic'])))
