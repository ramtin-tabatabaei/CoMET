"""Waypoint behavior tree: conditions, motion, gripper actions, and hold checks.

Robot actions finish within one tick; detector-triggered stops follow the
boundary condition check.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import py_trees
from py_trees.common import Status


def _slip_detect_extra_waypoints() -> int:
    """Extra waypoint budget before a slip-triggered stop."""
    try:
        return max(0, int(os.environ.get('AHA_SLIP_DETECT_EXTRA_WAYPOINTS', '1')))
    except (TypeError, ValueError):
        return 1


@dataclass
class BTContext:
    """Shared inputs and mutable state for one episode."""

    runner: Any
    args: Any
    scene: Any
    env_wrapper: Any
    robot: Any
    waypoints: Any
    n: int
    bt_stages: Any
    task_name: str
    chosen_cam: str
    side_camera: Any
    side_camera_window: Any

    monitor: Any
    yaml_descs: Any
    vlm_enabled: bool

    failure_active: bool
    failure_waypoint: Optional[int]

    responsible_detector: Optional[str] = None

    boundary_cache: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    deferred_hold_fires: dict = field(default_factory=dict)

    check_history: list = field(default_factory=list)

    diagnosis: Optional[dict] = None

    detect_grace_remaining: Optional[int] = None


@dataclass
class FailureReport:
    waypoint: int
    phase: str
    node: str
    failed_conditions: list
    failure_modes: list
    vlm_explanation: Any = None


def _norm(text: str) -> str:
    return ''.join((text or '').split()).lower().replace('==true', '=true').replace('==false', '=false')


def _failure_modes_for(blocks, failed_conditions):
    """Failure-mode names linked to the conditions that did not hold."""
    failed_keys = {_norm(c) for c in failed_conditions}
    modes = []
    for block in blocks or []:
        cond = block.get('condition') or block.get('original_condition') or ''
        if failed_keys and _norm(cond) not in failed_keys:
            continue
        for link in block.get('failure_links', []) or []:
            failure = link.get('failure')
            if failure and failure not in modes:
                modes.append(failure)
    return modes


def _record_failure(ctx, result, phase, i, blocks, node_name, fired=None):
    if fired:
        failed_conditions = sorted(fired)
        failure_modes = sorted(fired)
        explanation = None

    else:
        not_ok = [
            item.get('condition', '')
            for item in (result or {}).get('conditions', [])
            if item.get('status') == 'not_satisfied'
        ]
        failed_conditions = not_ok or [
            (block.get('condition') or block.get('original_condition') or '')
            for block in (blocks or [])
        ]
        failure_modes = _failure_modes_for(blocks, failed_conditions)
        explanation = (result or {}).get('explanation')

    ctx.failures.append(FailureReport(
        waypoint=i,
        phase=phase,
        node=node_name,
        failed_conditions=failed_conditions,
        failure_modes=failure_modes,
        vlm_explanation=explanation,
    ))


def _record_history(ctx, phase, i, result):
    """Keep condition-check evidence for end-of-run diagnosis."""
    if result is None:
        return
    conditions = [
        {
            'condition': item.get('condition', ''),
            'status': item.get('status', ''),
            'evidence': item.get('evidence', ''),
        }
        for item in (result.get('conditions', []) or [])
    ]
    ctx.check_history.append({
        'waypoint': i,
        'phase': phase,
        'checkpoint_passed': result.get('checkpoint_passed'),
        'conditions': conditions,
        'explanation': result.get('explanation', ''),
    })


def verdict_to_status(ctx, result, phase, i, blocks, node_name):
    """Map checkpoint verdicts using the manual/auto uncertainty policy."""
    if result is None:
        return Status.SUCCESS
    passed = result.get('checkpoint_passed')
    if passed is True:
        return Status.SUCCESS
    if passed is False:
        _record_failure(ctx, result, phase, i, blocks, node_name)
        return Status.FAILURE

    if ctx.args.mode == 'manual':
        ok = ctx.runner.ask_yes_no(
            f"  [uncertain] VLM could not decide the {phase} for waypoint {i}; "
            f"treat it as satisfied?",
            default=True,
        )
        if ok:
            return Status.SUCCESS
        _record_failure(ctx, result, phase, i, blocks, node_name)
        return Status.FAILURE

    print(
        f"  [uncertain] VLM could not decide the {phase} for waypoint {i}; "
        f"treating as satisfied."
    )
    return Status.SUCCESS


class _BTLeaf(py_trees.behaviour.Behaviour):
    def __init__(self, name, ctx, i, wp, stage):
        super().__init__(name=name)
        self.ctx = ctx
        self.i = i
        self.wp = wp
        self.stage = stage


class PreCondition(_BTLeaf):
    """Verify the waypoint's preconditions (or reuse the prior boundary verdict)."""

    def update(self):
        ctx, i, stage = self.ctx, self.i, self.stage
        runner = ctx.runner

        cached = ctx.boundary_cache.pop(i, None)
        if cached is not None:
            return cached
        runner.print_waypoint_header(self.wp, i, ctx.n - 1, stage, ctx.yaml_descs)
        runner.print_preconditions(stage, i)

        result = runner.maybe_run_vlm_condition_check(
            args=ctx.args,
            enabled=ctx.vlm_enabled,
            checkpoint_kind='pre',
            scene=ctx.scene,
            side_camera=ctx.side_camera,
            env_wrapper=ctx.env_wrapper,
            task_name=ctx.task_name,
            waypoint_index=i,
            stage=stage,
            conditions=runner.condition_strings(stage['preconditions']),
        )
        _record_history(ctx, 'precondition', i, result)
        return verdict_to_status(
            ctx, result, 'precondition', i, stage['preconditions'], self.name
        )


class MovePath(_BTLeaf):
    """Execute one complete waypoint path."""

    def update(self):
        ctx = self.ctx

        ctx.runner.bt_run_move(ctx, self.i, self.wp, self.stage)
        return Status.SUCCESS


class GripperAction(_BTLeaf):
    """Run the waypoint's gripper open/close action, or no-op when absent."""

    def update(self):
        self.ctx.runner.bt_run_gripper(self.ctx, self.i, self.wp)
        return Status.SUCCESS


class HoldMonitor(_BTLeaf):
    """Record fired live detectors, deferring fatal stop until after post/pre."""

    def update(self):
        ctx, i, stage = self.ctx, self.i, self.stage
        monitor = ctx.monitor
        if monitor is not None:
            ctx.runner.print_hold_results(stage, i, monitor)
        fired = monitor.fired_at(i) if monitor is not None else set()
        if fired:
            if not ctx.failure_active:
                return Status.SUCCESS
            responsible = getattr(ctx, 'responsible_detector', None)

            if not responsible:
                return Status.SUCCESS

            if responsible not in fired:
                return Status.SUCCESS

            fw = getattr(ctx, 'failure_waypoint', None)
            if (responsible in ('slip', 'collision')
                    and fw is not None and fw >= 0 and i < fw):
                print(
                    f"  [{responsible}] fired+confirmed at waypoint {i}, before "
                    f"the injection waypoint {fw}; recorded as a pre-injection "
                    f"false positive, not aborting."
                )
                return Status.SUCCESS

            ctx.deferred_hold_fires[i] = set(fired)
            return Status.SUCCESS
        return Status.SUCCESS


class PostCondition(_BTLeaf):
    """Check postconditions and cache the next precondition verdict."""

    def update(self):
        ctx, i, stage = self.ctx, self.i, self.stage
        runner = ctx.runner
        n = ctx.n

        def maybe_stop_for_deferred_hold(status):
            if status != Status.SUCCESS:
                return status
            fired = ctx.deferred_hold_fires.pop(i, None)
            if not fired:
                return Status.SUCCESS

            if ctx.detect_grace_remaining is None:
                ctx.detect_grace_remaining = (
                    _slip_detect_extra_waypoints()
                    if getattr(ctx, 'responsible_detector', None) == 'slip'
                    else 0
                )
            if (ctx.detect_grace_remaining > 0 and (i + 1) < n
                    and getattr(ctx, 'responsible_detector', None) == 'slip'):
                ctx.detect_grace_remaining -= 1
                ctx.deferred_hold_fires[i + 1] = fired
                print(
                    f"  [slip] not caught at waypoint {i} boundary (pre[{i + 1}]); "
                    f"giving the VLM another chance at waypoint {i + 1} "
                    f"(pre[{i + 2}])"
                )
                return Status.SUCCESS
            _record_failure(
                ctx, result=None, phase='hold', i=i,
                blocks=stage.get('hold_conditions', []),
                node_name=f"hold[{i}]", fired=fired,
            )
            return Status.FAILURE

        runner.print_postconditions(stage, i, n - 1)

        if i < n - 1:
            next_stage = runner.stage_for_waypoint(ctx.bt_stages, i + 1)
            runner.print_waypoint_header(
                ctx.waypoints[i + 1], i + 1, n - 1, next_stage, ctx.yaml_descs
            )
            runner.print_preconditions(next_stage, i + 1)
            result = runner.maybe_run_vlm_boundary_check(
                args=ctx.args,
                enabled=ctx.vlm_enabled,
                scene=ctx.scene,
                side_camera=ctx.side_camera,
                env_wrapper=ctx.env_wrapper,
                task_name=ctx.task_name,
                waypoint_index=i,
                stage=stage,
                next_waypoint_index=i + 1,
                next_stage=next_stage,
            )
            _record_history(ctx, f'postcondition+pre[{i + 1}]', i, result)
            status = verdict_to_status(
                ctx, result, 'postcondition', i,
                stage['postconditions'] + next_stage['preconditions'], self.name,
            )
            if status == Status.SUCCESS:
                ctx.boundary_cache[i + 1] = Status.SUCCESS
            return maybe_stop_for_deferred_hold(status)

        result = runner.maybe_run_vlm_condition_check(
            args=ctx.args,
            enabled=ctx.vlm_enabled,
            checkpoint_kind='post',
            scene=ctx.scene,
            side_camera=ctx.side_camera,
            env_wrapper=ctx.env_wrapper,
            task_name=ctx.task_name,
            waypoint_index=i,
            stage=stage,
            conditions=runner.condition_strings(stage['postconditions']),
        )
        _record_history(ctx, 'postcondition', i, result)
        status = verdict_to_status(
            ctx, result, 'postcondition', i, stage['postconditions'], self.name
        )
        return maybe_stop_for_deferred_hold(status)


def build_tree(ctx):
    Sequence = py_trees.composites.Sequence
    Parallel = py_trees.composites.Parallel
    policy = py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)

    wp_subtrees = []
    for i, wp in enumerate(ctx.waypoints):
        stage = ctx.runner.stage_for_waypoint(ctx.bt_stages, i)
        motion = Sequence(name=f"motion[{i}]", memory=True, children=[
            MovePath(f"move[{i}]", ctx, i, wp, stage),
            GripperAction(f"grip[{i}]", ctx, i, wp, stage),
        ])
        action = Parallel(name=f"act[{i}]", policy=policy, children=[
            motion,
            HoldMonitor(f"hold[{i}]", ctx, i, wp, stage),
        ])
        wp_subtrees.append(Sequence(name=f"wp[{i}]", memory=True, children=[
            PreCondition(f"pre[{i}]", ctx, i, wp, stage),
            action,
            PostCondition(f"post[{i}]", ctx, i, wp, stage),
        ]))
    return Sequence(name="episode", memory=True, children=wp_subtrees)


def _print_failure_summary(ctx):
    print("\n" + "=" * 70)
    if not ctx.failures:
        print("BT VERDICT: all checkpoints satisfied — no failures detected.")
        print("=" * 70)
        return
    print(f"BT VERDICT: {len(ctx.failures)} failure(s) detected — stopped at the first.")
    for report in ctx.failures:
        print("-" * 70)
        print(f"  Waypoint {report.waypoint}  [{report.phase}]  (node {report.node})")
        if report.failed_conditions:
            print("  Failed conditions:")
            for cond in report.failed_conditions:
                print(f"    - {cond}")
        if report.failure_modes:
            print(f"  Linked failure mode(s): {', '.join(report.failure_modes)}")
        if report.vlm_explanation:
            print(f"  VLM explanation: {report.vlm_explanation}")
    print("=" * 70)


def run_tree(ctx):
    """Build, display, and execute the tree; return its final status."""
    root = build_tree(ctx)
    print("\nBehavior tree:")
    print(py_trees.display.unicode_tree(root))

    root.tick_once()

    print("\n" + "=" * 70)
    print(f"BEHAVIOR TREE RESULT: {root.status.name}")
    print("=" * 70)
    print(py_trees.display.unicode_tree(root, show_status=True))
    _print_failure_summary(ctx)

    if (ctx.failures and getattr(ctx, 'vlm_enabled', False)
            and getattr(ctx.args, 'vlm_diagnose_failure', False)):
        try:
            ctx.runner.run_failure_diagnosis(ctx)
        except Exception as exc:
            print(f"  [diagnosis] failed: {exc}")
    return root.status
