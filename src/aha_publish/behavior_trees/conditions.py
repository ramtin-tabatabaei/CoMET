"""Condition text parsing and predicate normalization helpers."""

from aha_publish import paths

from .failure_definitions import *

def approach_alignment_condition(obj: str) -> str:
    return f"end_effector_aligned_with({obj}) == true"

def condition_text(condition) -> str:
    if isinstance(condition, dict):
        return (
            condition.get("condition")
            or condition.get("text")
            or condition.get("description")
            or ""
        )
    return str(condition)

def normalize_failure_link(link) -> dict:
    if isinstance(link, dict):
        failure = link.get("failure") or link.get("name") or link.get("failure_mode") or ""
        reason = link.get("reason") or link.get("rationale") or link.get("explanation") or ""
        return {"failure": str(failure).strip(), "reason": str(reason)}
    return {"failure": str(link).strip(), "reason": ""}

def condition_failure_links(condition) -> list[dict]:
    if not isinstance(condition, dict):
        return []
    links = (
        condition.get("failure_links")
        or condition.get("related_failures")
        or condition.get("failures")
        or []
    )
    if isinstance(links, str):
        links = [{"failure": links, "reason": ""}]
    normalized = [normalize_failure_link(link) for link in links]
    return [link for link in normalized if link["failure"]][:1]

MEASURED_SPATIAL_FUNCTION_RE = re.compile(
    r"\b(?:" + "dist" "ance" + r"|vertical_delta|horizontal_delta)\s*\("
)

INVALID_CONDITION_ARGUMENTS = frozenset(
    {
        "",
        "selected",
        "selection",
        "object",
        "obj",
        "item",
        "thing",
        "target",
        "reference",
        "region",
        "placement_region",
        "spawn_boundary",
        "spawnboundary",
        "boundary",
        "waypoint",
    }
)

def is_invalid_condition_argument(name: str) -> bool:
    normalized = str(name or "").strip().lower()
    if normalized in INVALID_CONDITION_ARGUMENTS:
        return True
    if re.fullmatch(r"(?:wp|waypoint)\d*", normalized):
        return True
    if normalized.endswith("_boundary") or normalized.endswith("boundary"):
        return True
    return False

# Trailing tokens that are simulator scene-graph cruft, not part of the object's
# real-world name (e.g. RLBench/CoppeliaSim shapes like basket_ball_hoop_respondable
# or crackers_box_visible). Stripped generically — no per-object mapping. Instance
# numbers (block0, block_1) are deliberately NOT stripped so distinct objects stay
# distinguishable.
SIMULATOR_NAME_SUFFIXES = frozenset(
    {
        "respondable",
        "visible",
        "visual",
        "invisible",
        "collision",
        "convex",
        "dynamic",
        "static",
        "mesh",
        "shape",
        "proxy",
        "link",
        "part",
        "sub",
        "ungrouped",
        "dummy",
        "sensor",
        "joint",
        "frame",
    }
)

@lru_cache(maxsize=16384)
def clean_object_name(name: str) -> str:
    """Strip trailing simulator-internal suffixes from an object identifier.

    e.g. basket_ball_hoop_respondable -> basket_ball_hoop,
         crackers_box_visible_respondable -> crackers_box.
    Reverts to the original if cleaning would empty the name or reduce it to a
    generic/invalid argument, so it can never break a condition.
    """
    raw = str(name).strip()
    if not raw:
        return raw
    tokens = raw.split("_")
    while len(tokens) > 1 and tokens[-1].lower() in SIMULATOR_NAME_SUFFIXES:
        tokens.pop()
    cleaned = "_".join(tokens)
    if not cleaned or is_invalid_condition_argument(cleaned):
        return raw
    return cleaned

@lru_cache(maxsize=16384)
def clean_condition_object_names(text: str) -> str:
    """Rewrite every object argument inside predicate(...) calls via clean_object_name.

    Predicate names and non-parenthesized predicates (e.g. gripper_condition = Open)
    are left untouched; only the comma-separated arguments are cleaned.
    """
    def _replace(match: re.Match) -> str:
        predicate, args = match.group(1), match.group(2)
        if not args.strip():
            return match.group(0)
        cleaned = ", ".join(clean_object_name(arg) for arg in args.split(","))
        return f"{predicate}({cleaned})"

    return re.sub(r"\b(\w+)\(([^)]*)\)", _replace, str(text))

@lru_cache(maxsize=16384)
def normalize_spatial_condition_text(text: str) -> str:
    """Normalize spatial checks to qualitative predicates only."""
    clean = " ".join(str(text).strip().split())
    if not clean:
        return clean
    clean = re.sub(
        r"\bgripper_is_open\s*==\s*true\b",
        "gripper_condition = Open",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\bgripper_is_closed\s*==\s*true\b",
        "gripper_condition = Closed",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\bgripper_condition\s*(?:==|=)\s*['\"]?open['\"]?\b",
        "gripper_condition = Open",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\bgripper_condition\s*(?:==|=)\s*['\"]?closed['\"]?\b",
        "gripper_condition = Closed",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\b(?:door_closed|lid_closed|drawer_closed|object_closed)\((\w+)\)\s*==\s*(?:true|True)\b",
        r"closed(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\b((?:\w*_)?(?:door|lid|drawer)(?:_\w*)?)_closed\s*==\s*(?:true|True)\b",
        r"closed(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\b(?:door_open|lid_open|drawer_open|object_open)\((\w+)\)\s*==\s*(?:true|True)\b",
        r"open(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\b((?:\w*_)?(?:door|lid|drawer)(?:_\w*)?)_open(?:ed)?\s*==\s*(?:true|True)\b",
        r"open(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    if (
        "postconditions_satisfied" in clean
        or clean.strip().lower() == "robot_state == ready"
        or re.search(r"\bstage_\d+_completed\s*==\s*true\b", clean)
    ):
        return ""

    clean = re.sub(
        r"\b(\w+)_(on|inside|next_to)\((\w+)\)\s*==\s*true\b",
        r"\2(\1, \3) == true",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\b(?:on|inside|next_to)\(((?:\w*_)?(?:door|lid|drawer)(?:_\w*)?),\s*(?:microwave|fridge|refrigerator|oven|grill|cabinet|drawer|box|jar|laptop)\)\s*==\s*true\b",
        r"closed(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\b(\w+)_point_cloud_cluster_valid\s*==\s*true\b",
        r"Object_found(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\b(\w+)_detected\s*==\s*true\b",
        r"Object_found(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\b(\w+)_visible_in_camera\s*==\s*true\b",
        r"Object_found(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\bobject_pose_confidence\((\w+)\)\s*>\s*\d+(?:\.\d+)?\b",
        r"Object_found(\1) == True",
        clean,
    )
    clean = re.sub(
        r"\b(\w+)_pose_confidence\s*>\s*\d+(?:\.\d+)?\b",
        r"Object_found(\1) == True",
        clean,
    )
    clean = re.sub(
        r"\bselected_object_id\s*==\s*locked_target_(\w+)_id\b",
        r"selected_object(\1) == True",
        clean,
    )
    clean = re.sub(
        r"\bcontact\(gripper,\s*(\w+)\)\s*==\s*true\b",
        "gripper_condition = Closed",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\bgripper_correct_grasp\(\w+\)\s*==\s*True\b",
        "gripper_condition = Closed",
        clean,
    )
    clean = re.sub(
        r"\b(\w+)_in_gripper\s*==\s*true\b",
        r"object_in_gripper(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\bgripper_contains\((\w+)\)\s*==\s*true\b",
        r"object_in_gripper(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"\b(\w+)_attached_to_end_effector\s*==\s*true\b",
        r"object_in_gripper(\1) == True",
        clean,
        flags=re.IGNORECASE,
    )
    return remove_measured_spatial_parts(clean)

@lru_cache(maxsize=16384)
def remove_measured_spatial_parts(text: str) -> str:
    parts = [
        part.strip()
        for part in str(text).split(" and ")
        if part.strip()
    ]
    kept = [
        part
        for part in parts
        if not MEASURED_SPATIAL_FUNCTION_RE.search(part)
        and not re.search(r"\b\d+(?:\.\d+)?\s*cm\b", part, re.IGNORECASE)
        and not re.search(r"(?:<|>|<=|>=)\s*\d", part)
        and not re.search(r"\b(?:radius|diameter|width|height)\b", part)
    ]
    return " and ".join(kept)

def is_waypoint_name(name: str) -> bool:
    return bool(re.fullmatch(r"(?:waypoint|wp)\d+", str(name), re.IGNORECASE))

def first_end_effector_object(text: str) -> str | None:
    match = re.search(r"end_effector_aligned_with\((\w+)\)", text)
    if match and not is_waypoint_name(match.group(1)):
        return match.group(1)
    return None

def normalize_wrong_position_text(text: str) -> str:
    if re.search(r"\b(?:on|inside|next_to)\(\w+,\s*\w+\)\s*==\s*true\b", text):
        return text

    obj = first_end_effector_object(text)
    if obj:
        return approach_alignment_condition(obj)

    return text

def normalize_condition(condition) -> dict:
    text = normalize_spatial_condition_text(condition_text(condition))
    links = condition_failure_links(condition) or catalogue_failure_links(text)
    failure = (links[0] if links else {}).get("failure", "")
    if failure == "WrongPosition":
        text = normalize_wrong_position_text(text)
    text = clean_condition_object_names(text)
    return {
        "condition": text,
        # Always empty: reviewers add/remove separate condition blocks rather
        # than swapping alternatives (see BT_CONDITION_GUIDELINES["alternatives"]).
        "alternatives": [],
        "failure_links": links,
    }

def merge_condition_blocks(conditions: list[dict]) -> list[dict]:
    """Normalize and deduplicate checks, preserving saved review metadata."""
    seen: set[str] = set()
    unique: list[dict] = []
    for item in conditions:
        normalized = normalize_condition(item)
        text = normalized["condition"]
        if not text:
            continue
        key = " ".join(text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        if isinstance(item, dict):
            for field in ("selected", "original_condition"):
                if field in item:
                    normalized[field] = item[field]
        unique.append(normalized)
    return unique
