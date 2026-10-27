"""Inspect a task TTM and export scene poses and detector reference geometry.

Batch task selection is handled by 01_ttm_context/main.py.
"""

import argparse
import json
import os
import sys

import numpy as np

from aha_publish import paths

RLBENCH_ROOT = paths.RLBENCH_ROOT
TTM_DIR = str(RLBENCH_ROOT / 'rlbench/task_ttms')
BASE_SCENE = str(RLBENCH_ROOT / 'rlbench/task_design.ttt')
sys.path.insert(0, str(RLBENCH_ROOT))


def as_float_list(v, p=6):
    return [round(float(x), p) for x in v]

def safe_pose_value(getter, relative_to=None):
    try:
        return getter(relative_to=relative_to)
    except Exception:
        return None

def type_name(obj):
    try:
        return obj.get_type().name
    except Exception:
        return '?'

def parent_name(obj):
    try:
        p = obj.get_parent()
        return p.get_name() if p is not None else '(root)'
    except Exception:
        return '?'

def is_waypoint_object(obj):
    """RLBench executes DUMMY waypoints as Points and PATH waypoints as
    PredefinedPaths (backend/task.py:_get_waypoints); both belong in the report."""
    return ('waypoint' in obj.get_name().lower()
            and type_name(obj) in ('DUMMY', 'PATH'))

def waypoint_world_pose(obj):
    """World pose [x, y, z, qx, qy, qz, qw] of a waypoint object.

    A cartesian-path waypoint's own frame sits mid-path, so its get_pose() is not
    the pose the arm is driven to -- PredefinedPath runs to the END of the path,
    and every other waypoint_pose helper in the repo resolves get_pose_on_path(1.0)
    for the same reason. Dummies are their own target.
    """
    from aha_publish.simulation.waypoint_chain import rpy_to_quat
    if type_name(obj) == 'PATH':
        from pyrep.objects.cartesian_path import CartesianPath
        position, euler = CartesianPath(obj.get_handle()).get_pose_on_path(1.0)
        return np.concatenate([np.asarray(position, dtype=float),
                               np.asarray(rpy_to_quat(euler), dtype=float)])
    return np.asarray(obj.get_pose(), dtype=float)

def waypoint_index(name):
    """Numeric ordering key from a waypoint name (waypoint0, waypoint10, waypoint-1, waypoint0.5)."""
    digits = ''.join(ch for ch in name if ch.isdigit() or ch in '.-')
    try:
        return float(digits)
    except ValueError:
        return 1e9


def phase1(ttm_path):
    """Read waypoint names, types, and parents from the imported task model."""
    from pyrep import PyRep
    from pyrep.const import ObjectType

    pr = PyRep()
    pr.launch(BASE_SCENE, headless=True)
    pr.start()
    root = pr.import_model(ttm_path)

    objects = [root]
    for object_type in (ObjectType.DUMMY, ObjectType.PATH):
        try:
            objects.extend(root.get_objects_in_tree(
                object_type=object_type, exclude_base=True))
        except Exception:
            pass
    waypoint_data = [
        {'name': wp.get_name(), 'type': type_name(wp), 'parent': parent_name(wp)}
        for wp in sorted(objects, key=lambda obj: obj.get_name())
        if is_waypoint_object(wp)
    ]
    pr.stop()
    pr.shutdown()
    return waypoint_data


def phase2(task_name):
    """Collect initialized scene poses and validate waypoint anchors across resets."""
    from failgen.env_wrapper import FailGenEnvWrapper
    from pyrep.const import ObjectType

    env = FailGenEnvWrapper(
        task_name=task_name,
        headless=True,
        record=False,
        save_data=False,
        save_path=str(paths.BACKEND_DATA_DIR / 'ttm_context'),
    )
    env.reset()

    sc    = env._env._scene
    task  = sc.task
    broot = task.boundary_root()
    bname = broot.get_name()
    broot_world_pos = safe_pose_value(broot.get_position)
    broot_world_ori = safe_pose_value(broot.get_orientation)

    graspable_objs = task.get_graspable_objects()
    graspable_names = {o.get_name() for o in graspable_objs}

    graspable_data = []
    for obj in graspable_objs:
        graspable_data.append({
            'name': obj.get_name(),
            'world_position': safe_pose_value(obj.get_position),
            'world_orientation': safe_pose_value(obj.get_orientation),
        })

    # Keep named scene shapes used in description context.
    all_shapes = broot.get_objects_in_tree(object_type=ObjectType.SHAPE, exclude_base=False)
    non_graspable = [o for o in all_shapes if o.get_name() not in graspable_names
                     and 'visual' not in o.get_name().lower()
                     and 'stop' not in o.get_name().lower()
                     and o.get_name() != bname]

    # Preserve the nearest graspable name used in scene context.
    ng_data = []
    for obj in non_graspable:
        obj_pos_world = np.array(obj.get_position())
        best_name = None
        best_dist = float('inf')
        for g in graspable_objs:
            g_pos = np.array(g.get_position())
            off   = obj_pos_world - g_pos
            dist  = float(np.linalg.norm(off))
            if dist < best_dist:
                best_dist, best_name = dist, g.get_name()
        ng_data.append({
            'name': obj.get_name(),
            'reference_name': best_name,
            'world_position': safe_pose_value(obj.get_position),
            'world_orientation': safe_pose_value(obj.get_orientation),
        })

    # Capture world poses for descriptions and parent-local poses for detectors.
    from aha_publish.simulation.waypoint_chain import quat_to_rpy, relative_pose

    live_waypoint_world = []
    live_wp_objs = [
        o
        for otype in (ObjectType.DUMMY, ObjectType.PATH)
        for o in broot.get_objects_in_tree(object_type=otype, exclude_base=False)
        if 'waypoint' in o.get_name().lower()
    ]
    for o in sorted(live_wp_objs, key=lambda x: (waypoint_index(x.get_name()), x.get_name())):
        parent = None
        try:
            parent = o.get_parent()
        except Exception:
            parent = None
        # Path targets and their children use the endpoint as their reference frame.
        parent_is_path = parent is not None and type_name(parent) == 'PATH'
        if type_name(o) == 'PATH' or parent_is_path:
            world = waypoint_world_pose(o)
            wpos, wori = world[:3], quat_to_rpy(world[3:7])
            if parent is None:
                lpos = lquat = None
            else:
                rel = relative_pose(waypoint_world_pose(parent), world)
                lpos, lquat = rel[:3], rel[3:7]
        else:
            wpos = safe_pose_value(o.get_position)
            wori = safe_pose_value(o.get_orientation)
            lpos = safe_pose_value(o.get_position, relative_to=parent) if parent else None
            lquat = safe_pose_value(o.get_quaternion, relative_to=parent) if parent else None
        live_waypoint_world.append({
            'name': o.get_name(),
            'type': type_name(o),
            'parent': parent_name(o),
            'world_position_xyz_m': as_float_list(wpos) if wpos is not None else None,
            'world_orientation_rpy_rad': as_float_list(wori) if wori is not None else None,
            'local_position_xyz_m': as_float_list(lpos, 9) if lpos is not None else None,
            'local_quaternion_xyzw': as_float_list(lquat, 9) if lquat is not None else None,
        })

    # Infer each component independently: an object's position can determine a
    # waypoint while its independently randomized yaw must not determine it.
    from aha_publish.simulation.waypoint_anchors import infer_waypoint_anchors
    ANCHOR_EPISODES = 5

    def tree_poses():
        poses = {}
        for o in broot.get_objects_in_tree(exclude_base=False):
            try:
                # Path waypoints contribute their endpoint, so a re-anchored
                # offset stays about the same point the local_* fields describe.
                poses[o.get_name()] = (waypoint_world_pose(o)
                                       if is_waypoint_object(o)
                                       else np.array(o.get_pose()))
            except Exception:
                pass
        try:
            poses[bname] = np.array(broot.get_pose())
        except Exception:
            pass
        return poses

    # episodes[0] is the episode every world pose in this report came from, so
    # re-anchored offsets are taken from it and stay consistent with them.
    episodes = [tree_poses()]
    # Numeric vector state referenced by pose equations belongs in the report
    # (e.g. an initial relative offset), not in runtime simulator waypoint reads.
    from aha_publish.simulation.waypoint_reference_rules import export_hook_reference_rules
    import ast
    _source = RLBENCH_ROOT / 'rlbench/tasks' / f'{task_name}.py'
    _rules = export_hook_reference_rules(_source.read_text()) if _source.exists() else {}
    _state_names = set()
    def collect_state(value):
        if isinstance(value, dict):
            for item in value.values():
                collect_state(item)
        elif isinstance(value, list):
            for item in value:
                collect_state(item)
        elif isinstance(value, str):
            try:
                for node in ast.walk(ast.parse(value, mode='eval')):
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'self':
                        _state_names.add(node.attr)
            except SyntaxError:
                pass
    collect_state(_rules)
    def numeric_state():
        result = {}
        for name in _state_names:
            value = getattr(task, name, None)
            if isinstance(value, (list, tuple, np.ndarray)):
                try:
                    arr = np.asarray(value, dtype=float)
                    if arr.size and np.isfinite(arr).all() and arr.ndim:
                        result[name] = arr.tolist()
                except (TypeError, ValueError):
                    pass
        return result
    _state_samples = [numeric_state()]
    for _ in range(ANCHOR_EPISODES - 1):
        env.reset()
        episodes.append(tree_poses())
        _state_samples.append(numeric_state())
    _reference_state = {}
    for name, value in _state_samples[0].items():
        try:
            if all(name in sample and np.shape(sample[name]) == np.shape(value)
                   and np.allclose(sample[name], value, atol=1e-5, rtol=0)
                   for sample in _state_samples[1:]):
                _reference_state[name] = value
        except (TypeError, ValueError):
            pass

    live_waypoint_world = infer_waypoint_anchors(live_waypoint_world, episodes)
    reanchored = []
    for w in live_waypoint_world:
        if w.get('reference_unavailable'):
            print(f'  WARNING: {w["name"]}: {w["reference_unavailable"]}')
        for key in ('position_parent', 'orientation_parent'):
            anchor = w.get(key)
            if anchor and anchor != w['parent']:
                reanchored.append((w['name'] + ':' + key, anchor))

    # An anchor the report does not list has no world pose to compose against,
    # so record any that is missing (close_jar's waypoint3 anchors onto
    # `success`, a proximity sensor, which is in neither object group).
    known_objects = ({o['name'] for o in graspable_data}
                     | {o['name'] for o in ng_data} | {bname})
    for _, anchor in reanchored:
        if anchor in known_objects:
            continue
        obj = next((o for o in broot.get_objects_in_tree(exclude_base=False)
                    if o.get_name() == anchor), None)
        if obj is None:
            continue
        ng_data.append({
            'name': anchor,
            'reference_name': None,
            'world_position': list(episodes[0][anchor][:3]),
            'world_orientation': list(quat_to_rpy(episodes[0][anchor][3:7])),
        })
        known_objects.add(anchor)

    env.shutdown()

    return {
        'live_waypoint_world_poses': live_waypoint_world,
        'reference_variation_index': sc._variation_index,
        'reference_state': _reference_state,
        'reference_objects': {name: pose.tolist() for name, pose in episodes[0].items()
                              if 'waypoint' not in name.lower()},
        # The task base grounds waypoint chains without becoming a described object.
        'task_boundary_root': {
            'name': bname,
            'world_pose_after_init': {
                'position_xyz_m': (
                    as_float_list(broot_world_pos) if broot_world_pos is not None else None
                ),
                'orientation_rpy_rad': (
                    as_float_list(broot_world_ori) if broot_world_ori is not None else None
                ),
            }
        },
        'placement_distribution': {
            'graspable_objects': [
                {
                    'name': obj['name'],
                    'world_pose_after_init': scene_object_pose(obj),
                }
                for obj in graspable_data
            ],
            'other_scene_objects': [
                {
                    'name': obj['name'],
                    'reference_graspable_object': obj['reference_name'],
                    'world_pose_after_init': scene_object_pose(obj),
                }
                for obj in ng_data
            ],
        },
    }


def scene_object_pose(obj):
    """Serialize the scene-object pose fields consumed by the report writer."""
    return {
        'position_xyz_m': (as_float_list(obj['world_position'])
                           if obj['world_position'] is not None else None),
        'orientation_rpy_rad': (as_float_list(obj['world_orientation'])
                                if obj['world_orientation'] is not None else None),
    }

def build_llm_report_json(task_name, waypoint_data, placement_data):
    """Lean, downstream-facing JSON: the fields that inform a task description, plus
    the parent-frame geometry the detectors need.

    Two audiences share one file:

    - The description/BT prompts read the world-frame poses (`world_*_after_init`)
      and `scene_objects`. That is the whole of what they see -- the description
      maker strips everything below before the report reaches the model.
    - The transition/orientation detectors read the per-waypoint `local_*` offset
      (position + xyzw quaternion in the parent frame) and `task_boundary_root` to
      compose a reference target that survives across episodes (the world poses do
      not; they are one episode's sample of the placement distribution).
    """
    live_wp_map = {
        w['name']: w for w in placement_data.get('live_waypoint_world_poses', [])
    }
    pd = placement_data.get('placement_distribution', {})
    base = placement_data.get('task_boundary_root') or {}

    ordered = sorted(waypoint_data, key=lambda w: (waypoint_index(w['name']), w['name']))
    waypoints_out = []
    for wp in ordered:
        lw = live_wp_map.get(wp['name'], {})
        pos = lw.get('world_position_xyz_m')
        ori = lw.get('world_orientation_rpy_rad')
        # The live parent, not phase 1's: a waypoint init_episode re-places is
        # re-anchored there, and local_* below is expressed in THAT frame.
        entry = {'name': wp['name'],
                 'type': lw.get('type') or wp.get('type'),
                 'parent': lw.get('parent') or wp['parent']}
        if pos is not None:
            entry['world_position_after_init_xyz_m'] = pos
        if ori is not None:
            entry['world_orientation_after_init_rpy_rad'] = ori
        local_pos = lw.get('local_position_xyz_m')
        local_quat = lw.get('local_quaternion_xyzw')
        if local_pos is not None and local_quat is not None:
            entry['local_position_xyz_m'] = local_pos
            entry['local_quaternion_xyzw'] = local_quat
        else:
            entry['local_offset_unavailable'] = (
                f"no parent-relative pose could be read for parent {entry['parent']!r}"
            )
        for key in ('position_parent', 'orientation_parent',
                    'anchor_validation_episodes', 'reference_unavailable'):
            if key in lw:
                entry[key] = lw[key]
        waypoints_out.append(entry)

    def _world_pose(obj):
        wp = obj.get('world_pose_after_init', {})
        return {
            'position_xyz_m': wp.get('position_xyz_m'),
            'orientation_rpy_rad': wp.get('orientation_rpy_rad'),
        }

    graspable = [
        {'name': o['name'], 'world_pose_after_init': _world_pose(o)}
        for o in pd.get('graspable_objects', [])
    ]
    fixed = [
        {
            'name': o['name'],
            'reference_graspable_object': o.get('reference_graspable_object'),
            'world_pose_after_init': _world_pose(o),
        }
        for o in pd.get('other_scene_objects', [])
    ]

    data = {
        'task_name': task_name,
        'reference_variation_index': placement_data.get('reference_variation_index', 0),
        'reference_state': placement_data.get('reference_state', {}),
        'reference_objects': placement_data.get('reference_objects', {}),
        'units': 'positions in meters [x, y, z]; orientations rpy Euler radians [roll_x, pitch_y, yaw_z]',
        'waypoints': waypoints_out,
        'scene_objects': {'graspable': graspable, 'fixed': fixed},
    }
    if base.get('name'):
        data['task_boundary_root'] = base
    # Reset-only inference must never certify waypoints changed by live hooks or
    # repeated sequences. Preserve the generic source-analysis guard for those.
    from aha_publish.simulation.rlbench_dynamic_waypoints import analyze_task, unresolved_dynamic_waypoints
    analysis = analyze_task(task_name, tasks_dir=RLBENCH_ROOT / 'rlbench/tasks',
                            waypoint_count=len(waypoints_out))
    if analysis is not None:
        data['dynamic_waypoints'] = unresolved_dynamic_waypoints(analysis, data)
        data['dynamic_waypoint_reasons'] = {
            str(i): analysis['reasons'][i] for i in data['dynamic_waypoints']}
    # Runtime consumes these equations from this report, not the task source
    # or the simulator's waypoint geometry.
    from aha_publish.simulation.waypoint_reference_rules import export_hook_reference_rules
    task_source = RLBENCH_ROOT / 'rlbench/tasks' / f'{task_name}.py'
    if task_source.exists():
        rules = export_hook_reference_rules(task_source.read_text())
        if rules['hooks'] or rules['errors']:
            data['hook_reference_rules'] = rules
            if rules['hooks'] and not rules['errors']:
                data['rule_resolved_waypoints'] = data.get('dynamic_waypoints', [])
    return json.dumps(data, indent=2)


def run_single(target, save_path=None):
    if os.path.isfile(target):
        ttm_path  = target
        task_name = os.path.basename(target).replace('.ttm', '')
    else:
        ttm_path  = os.path.join(TTM_DIR, f'{target}.ttm')
        task_name = target
        if not os.path.isfile(ttm_path):
            print(f'File not found: {ttm_path}')
            sys.exit(1)

    if save_path is None:
        save_path = f'{task_name}.llm_context.json'

    waypoint_data = phase1(ttm_path)
    placement_data = phase2(task_name)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(build_llm_report_json(
            task_name=task_name,
            waypoint_data=waypoint_data,
            placement_data=placement_data,
        ))
    print(f'\n[saved → {save_path}]')


def main():
    parser = argparse.ArgumentParser(
        description='Inspect one RLBench task TTM. Use 01_ttm_context/main.py for batch runs.',
    )
    parser.add_argument('target', help='Task name or path to a .ttm file.')
    parser.add_argument('-o', '--save', help='Output path. Default: <task>.llm_context.json')
    args = parser.parse_args()
    run_single(args.target, args.save)


if __name__ == '__main__':
    main()
