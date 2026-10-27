"""Resolve waypoint world poses from stored local offsets and reference poses.

Poses use [x, y, z, qx, qy, qz, qw]. Composition follows
T_world(node) = T_world(parent) @ T_local(node).
"""

from aha_publish import paths

import glob
import json
import os

import numpy as np

__all__ = [
    "quat_multiply",
    "quat_conjugate",
    "quat_rotate",
    "compose",
    "relative_pose",
    "pose_error",
    "rpy_to_quat",
    "spec_from_inspection_report",
    "load_chain_spec",
    "validate_spec",
    "resolve_waypoint_poses",
    "WaypointChain",
]


def _as_quat(q):
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != 4:
        raise ValueError(f"expected a 4-element xyzw quaternion, got {q.size}")
    n = np.linalg.norm(q)
    if n <= 1e-12:
        raise ValueError("degenerate (zero-norm) quaternion")
    return q / n


def quat_multiply(q1, q2):
    """Hamilton product q1 * q2, both xyzw. Applies q2 first, then q1."""
    x1, y1, z1, w1 = _as_quat(q1)
    x2, y2, z2, w2 = _as_quat(q2)
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def quat_rotate(q, v):
    """Rotate vector v by quaternion q (xyzw)."""
    x, y, z, w = _as_quat(q)
    v = np.asarray(v, dtype=float).reshape(3)
    u = np.array([x, y, z])
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def rpy_to_quat(rpy):
    """Convert intrinsic X-Y-Z Euler radians to an xyzw quaternion.

    Uses R = Rx(a) @ Ry(b) @ Rz(g), matching CoppeliaSim/PyRep.
    """
    a, b, g = (float(x) for x in np.asarray(rpy, dtype=float).reshape(3))
    ca, sa, cb, sb, cg, sg = (np.cos(a), np.sin(a), np.cos(b),
                              np.sin(b), np.cos(g), np.sin(g))
    r = np.array([
        [cb * cg, -cb * sg, sb],
        [sa * sb * cg + ca * sg, -sa * sb * sg + ca * cg, -sa * cb],
        [-ca * sb * cg + sa * sg, ca * sb * sg + sa * cg, ca * cb],
    ])
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q = np.array([(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s,
                      (r[1, 0] - r[0, 1]) / s, 0.25 * s])
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = np.array([0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s,
                      (r[2, 1] - r[1, 2]) / s])
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = np.array([(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s,
                      (r[0, 2] - r[2, 0]) / s])
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = np.array([(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s,
                      (r[1, 0] - r[0, 1]) / s])
    return q / np.linalg.norm(q)


def quat_to_rpy(q):
    """Convert an xyzw quaternion to intrinsic X-Y-Z Euler radians."""
    x, y, z, w = _as_quat(q)
    r02 = 2 * (x * z + w * y)
    r12 = 2 * (y * z - w * x)
    r22 = 1 - 2 * (x * x + y * y)
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    beta = np.arcsin(np.clip(r02, -1.0, 1.0))
    if abs(r02) < 1.0 - 1e-9:
        alpha = np.arctan2(-r12, r22)
        gamma = np.arctan2(-r01, r00)
    else:                                     # gimbal lock: fold into alpha
        alpha = np.arctan2(2 * (y * z + w * x), 1 - 2 * (x * x + z * z))
        gamma = 0.0
    return np.array([float(alpha), float(beta), float(gamma)])


def quat_conjugate(q):
    """Inverse rotation of a unit quaternion (xyzw)."""
    x, y, z, w = _as_quat(q)
    return np.array([-x, -y, -z, w])


def compose(parent_pose, local_pose):
    """Compose parent and local poses, both [x, y, z, qx, qy, qz, qw]."""
    parent = np.asarray(parent_pose, dtype=float).reshape(-1)
    local = np.asarray(local_pose, dtype=float).reshape(-1)
    if parent.size < 7 or local.size < 7:
        raise ValueError("compose() needs 7-element [xyz + xyzw quat] poses")
    position = parent[:3] + quat_rotate(parent[3:7], local[:3])
    quaternion = quat_multiply(parent[3:7], local[3:7])
    return np.concatenate([position, quaternion])


def relative_pose(parent_pose, child_pose):
    """Express a child world pose relative to its parent world pose."""
    parent = np.asarray(parent_pose, dtype=float).reshape(-1)
    child = np.asarray(child_pose, dtype=float).reshape(-1)
    if parent.size < 7 or child.size < 7:
        raise ValueError("relative_pose() needs 7-element poses")
    inv_q = quat_conjugate(parent[3:7])
    position = quat_rotate(inv_q, child[:3] - parent[:3])
    quaternion = quat_multiply(inv_q, child[3:7])
    return np.concatenate([position, quaternion])


def pose_error(pose_a, pose_b):
    """Return position error (m) and orientation error (rad), treating q and -q equally."""
    a = np.asarray(pose_a, dtype=float).reshape(-1)
    b = np.asarray(pose_b, dtype=float).reshape(-1)
    if a.size < 7 or b.size < 7:
        return float("nan"), float("nan")
    d_pos = float(np.linalg.norm(a[:3] - b[:3]))
    qa, qb = _as_quat(a[3:7]), _as_quat(b[3:7])
    dot = float(np.clip(abs(np.dot(qa, qb)), -1.0, 1.0))
    return d_pos, float(2.0 * np.arccos(dot))


DEFAULT_CHAIN_DIR = str(paths.TTM_CONTEXT_DIR)


def chain_spec_path(task_name, chain_dir=None):
    """Find the task report, allowing an optional numeric filename prefix."""
    base = chain_dir or os.getenv("AHA_WP_CHAIN_DIR") or DEFAULT_CHAIN_DIR
    exact = os.path.join(base, f"{task_name}.llm_context.json")
    if os.path.isfile(exact):
        return exact
    matches = sorted(glob.glob(
        os.path.join(base, f"*_{task_name}.llm_context.json")))
    return matches[0] if matches else exact


def _index_of(name):
    """Return the index for an exact waypointN name; decorated names return None."""
    if not name.startswith("waypoint"):
        return None
    suffix = name[len("waypoint"):]
    if not suffix.isdigit():
        return None
    return int(suffix)


def spec_from_inspection_report(report):
    """Build reference chains from an inspection report's local poses.

    Unresolved targets and entries without usable local poses stay ungrounded.
    """
    objects = {}
    scene = report.get("scene_objects") or {}
    for group in ("graspable", "fixed"):
        for obj in scene.get(group) or []:
            pose = obj.get("world_pose_after_init") or {}
            position, rpy = pose.get("position_xyz_m"), pose.get("orientation_rpy_rad")
            if position is None or rpy is None:
                continue
            objects[obj["name"]] = np.concatenate(
                [np.asarray(position, dtype=float).reshape(3), rpy_to_quat(rpy)])
    # The task base can serve as a reference frame.
    base = report.get("task_boundary_root") or {}
    base_pose = base.get("world_pose_after_init") or {}
    if base.get("name") and base_pose.get("position_xyz_m") is not None:
        objects[base["name"]] = np.concatenate([
            np.asarray(base_pose["position_xyz_m"], dtype=float).reshape(3),
            rpy_to_quat(base_pose["orientation_rpy_rad"])])

    # Dynamic targets need validated hook rules before they can resolve.
    dynamic = {int(i) for i in (report.get("dynamic_waypoints") or [])}
    rules = report.get('hook_reference_rules') or {}
    if rules.get('hooks') and not rules.get('errors'):
        dynamic.difference_update(report.get('rule_resolved_waypoints', []))
    dynamic.update(_index_of(w['name']) for w in report.get('waypoints', [])
                   if w.get('reference_unavailable') and _index_of(w['name']) is not None)

    # Preserve intermediate dummies, including non-executable waypoint names.
    nodes, waypoints, roots = {}, [], set()
    for wp in report.get("waypoints") or []:
        name = wp["name"]
        parent = wp.get("orientation_parent", wp.get("parent"))
        position_parent = wp.get("position_parent", parent)
        stored_p = wp.get("local_position_xyz_m")
        stored_q = wp.get("local_quaternion_xyzw")
        if _index_of(name) in dynamic or wp.get("reference_unavailable"):
            nodes[name] = {"parent": None, "local_position": None,
                           "local_quaternion": None, "kind": "dynamic",
                           "missing_parent": parent}
        elif parent and stored_p is not None and stored_q is not None:
            nodes[name] = {
                "parent": parent,
                "local_position": list(stored_p),
                "local_quaternion": list(stored_q),
                "kind": "waypoint",
                "position_parent": position_parent,
            }
            if parent in objects:
                roots.add(parent)
            if position_parent and not position_parent.startswith('waypoint'):
                roots.add(position_parent)
        else:
            nodes[name] = {"parent": None, "local_position": None,
                           "local_quaternion": None, "kind": "ungrounded",
                           "missing_parent": parent}

    # Discover roots after every node has been created.
    for wp in report.get("waypoints") or []:
        name = wp["name"]
        index = _index_of(name)
        if index is None:
            continue
        node = nodes.get(name) or {}
        chain = [name]
        cursor = node.get("parent") or node.get("missing_parent")
        seen = {name}
        while cursor in nodes and nodes[cursor].get("parent") and cursor not in seen:
            chain.append(cursor)
            seen.add(cursor)
            cursor = nodes[cursor]["parent"]
        if cursor:
            chain.append(cursor)
            # A missing waypoint cannot supply a root reference pose.
            if (cursor not in nodes and node.get("kind") == "waypoint"
                    and not cursor.startswith("waypoint")):
                roots.add(cursor)
        parent = node.get("parent")
        waypoints.append({"index": index, "name": name, "scene_name": name,
                          "kind": "point", "root_object": parent if parent in objects
                          else None, "chain": chain})

    return {
        "task_name": report.get("task_name"),
        "reference_variation_index": report.get("reference_variation_index"),
        "hook_reference_rules": rules,
        "reference_scene_objects": sorted(set(objects) | set(report.get('reference_objects', {}))),
        "reference_state": report.get('reference_state', {}),
        "generated_by": "spec_from_inspection_report (ttm_inspection_reports)",
        "convention": {
            "pose_layout": "[x, y, z, qx, qy, qz, qw]",
            "quaternion_order": "xyzw (PyRep/CoppeliaSim)",
            "composition": "T_world(node) = T_world(parent) . T_local(node)",
        },
        "root_objects": sorted(roots),
        "nodes": nodes,
        "waypoints": waypoints,
        "dynamic_waypoints": sorted(dynamic),
        # Track references that depend on successful hook evaluation.
        "rule_resolved_waypoints": sorted(
            {int(i) for i in (report.get('rule_resolved_waypoints') or [])}
            - set(dynamic)),
    }


def load_chain_spec(task_name, chain_dir=None):
    """Load a task's inspection report and convert it to a chain spec."""
    path = chain_spec_path(task_name, chain_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    return spec_from_inspection_report(report)


def validate_spec(spec):
    """Return structural errors in the chain, including invalid poses and parent cycles."""
    problems = []
    nodes = spec.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        return ["'nodes' is missing or empty"]

    for name, node in nodes.items():
        if not isinstance(node, dict):
            problems.append(f"node {name!r} is not an object")
            continue
        # Dynamic nodes have no static offset to validate.
        if node.get("kind") == "dynamic":
            continue
        parent = node.get("parent")
        if parent is None:
            problems.append(f"node {name!r} has no parent (ungrounded chain)")
        for key, size in (("local_position", 3), ("local_quaternion", 4)):
            value = node.get(key)
            if value is None:
                problems.append(f"node {name!r} is missing {key}")
                continue
            try:
                arr = np.asarray(value, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                problems.append(f"node {name!r} has a non-numeric {key}")
                continue
            if arr.size != size:
                problems.append(
                    f"node {name!r} {key} has {arr.size} values, expected {size}")
            elif not np.all(np.isfinite(arr)):
                problems.append(f"node {name!r} {key} contains NaN/inf")
            elif key == "local_quaternion":
                norm = float(np.linalg.norm(arr))
                if abs(norm - 1.0) > 1e-3:
                    problems.append(
                        f"node {name!r} local_quaternion is not unit "
                        f"(norm {norm:.6f}); xyzw order, w last")

    seen = set()
    for wp in spec.get("waypoints") or []:
        name = wp.get("name")
        if name not in nodes:
            problems.append(f"waypoint entry {name!r} has no matching node")
        try:
            index = int(wp["index"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"waypoint entry {name!r} has no integer 'index'")
            continue
        if index in seen:
            problems.append(f"duplicate waypoint index {index}")
        seen.add(index)

    roots = set(spec.get("root_objects") or [])
    checked = set()

    def check_parents(name, visiting):
        if name in visiting:
            problems.append(f"cyclic parent chain at {name!r}")
            return
        if name in checked:
            return
        if name not in nodes:
            if name not in roots:
                problems.append(f"frame {name!r} is not listed in root_objects")
            return
        node = nodes[name]
        if not isinstance(node, dict):
            return
        parent = node.get('parent')
        for anchor in {parent, node.get('position_parent', parent)} - {None}:
            check_parents(anchor, visiting | {name})
        checked.add(name)

    for name in nodes:
        check_parents(name, set())
    return problems


def resolve_waypoint_poses(spec, world_poses, strict=False):
    """Compose waypoint poses from a spec and supplied world poses.

    world_poses maps object names to [x, y, z, qx, qy, qz, qw]. Supplied nodes
    terminate the parent walk. Unresolved waypoints are omitted unless strict=True.
    """
    nodes = spec.get("nodes") or {}
    known = {}
    for name, pose in (world_poses or {}).items():
        arr = np.asarray(pose, dtype=float).reshape(-1)
        if arr.size >= 7:
            known[name] = arr[:7]

    resolving = set()

    def world_pose_of(name):
        if name in known:
            return known[name]
        if name in resolving:                     # cyclic parenting in the ttm
            raise ValueError(f"cyclic parent chain at {name!r}")
        node = nodes.get(name)
        if node is None:
            raise KeyError(f"no world pose and no chain entry for {name!r}")
        parent = node.get("parent")
        if not parent:
            raise KeyError(f"chain for {name!r} reaches the scene root ungrounded")
        resolving.add(name)
        try:
            parent_world = world_pose_of(parent)
            position_world = world_pose_of(node.get('position_parent', parent))
            position = (position_world[:3] + quat_rotate(
                position_world[3:7], node['local_position']))
            quaternion = quat_multiply(parent_world[3:7], node['local_quaternion'])
            pose = np.concatenate([position, quaternion])
        finally:
            resolving.discard(name)
        known[name] = pose
        return pose

    out = {}
    for wp in spec.get("waypoints") or []:
        name = wp["name"]
        try:
            out[int(wp["index"])] = world_pose_of(name)
        except (KeyError, ValueError):
            if strict:
                raise
    return out


class WaypointChain:
    """Load a task reference chain and cache resolved poses for each episode."""

    def __init__(self, spec):
        self.spec = spec
        self.poses = {}
        self.world_poses = {}

    @classmethod
    def for_task(cls, task_name, chain_dir=None):
        spec = load_chain_spec(task_name, chain_dir)
        return cls(spec) if spec else None

    @property
    def root_objects(self):
        """Names whose world pose must be supplied to resolve()."""
        return list(self.spec.get("root_objects") or [])

    def resolve(self, world_poses, strict=False, variation_index=None):
        expected = self.spec.get('reference_variation_index')
        if expected is not None and variation_index is not None and expected != variation_index:
            self.poses = {}
            raise ValueError(f"reference inspected for variation {expected}, "
                             f"episode uses {variation_index}; regenerate its inspection")
        self.world_poses = dict(world_poses)
        self.poses = resolve_waypoint_poses(self.spec, world_poses, strict=strict)
        return self.poses

    def pose(self, index):
        pose = self.poses.get(int(index))
        return None if pose is None else list(pose)

    def unresolved(self):
        """Waypoint indices the last resolve() could not ground."""
        want = {int(w["index"]) for w in (self.spec.get("waypoints") or [])}
        return sorted(want - set(self.poses))
