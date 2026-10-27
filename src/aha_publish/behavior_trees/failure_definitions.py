"""Catalogue failure links, runtime hold conditions, and scene-name mappings."""

from aha_publish import paths

from functools import lru_cache

from aha_publish.detectors import load_primitive_policy

from .io_context import *

# ------------------------------------------------------------------------------
# Hold conditions (runtime detectors)
#
# Continuous "must stay true while the stage runs" invariants, one per runtime
# detector in aha_scripts/detectors/. Unlike pre/postconditions these are NOT produced by the
# language model; the BT maker assigns them by config/detector_primitives.json.
# This list is the catalog: it is the only definition of the
# hold-condition set, and the text below is what lands in every generated BT.
# ------------------------------------------------------------------------------
DEFAULT_HOLD_CONDITION_DETECTORS = [
    {
        "detector": "collision",
        "failure": "collision",
        "condition": "no_collision() == True",
        "label": "No collision",
        "definition": "No unexpected collision or contact spike occurs while the stage executes. Backed by the collision detector (joint-torque deltas/norms compared against the task's successful-run torque stats).",
    },
    {
        "detector": "slip",
        "failure": "slip",
        "condition": "maintains_grasp() == True",
        "label": "Grasp maintained",
        "definition": "A grasped object does not slip out of the gripper while the stage executes. Backed by the slip detector (gripper touch-force loss relative to the grip-force stats while in a holding state).",
    },
    {
        "detector": "orientation",
        "failure": "orientation",
        "condition": "orientation_maintained() == True",
        "label": "Orientation maintained",
        "definition": "The waypoint/object orientation stays within the expected trend and does not drift. Backed by the orientation detector (TTM-recalculated waypoint angle trend).",
    },
    {
        "detector": "transition",
        "failure": "transition",
        "condition": "reaches_waypoint() == True",
        "label": "Waypoint reached",
        "definition": "The end effector reaches the expected waypoint pose and transitions in the correct sequence. Backed by the transition detector (end-effector distance to the expected waypoint pose).",
    },
]

HOLD_CONDITIONS_DESCRIPTION = (
    "Continuous invariants that must stay TRUE for the whole time a waypoint/stage is "
    "executing, as opposed to the pre/post checkpoints. Each one maps one-to-one to a "
    "runtime detector. Configured primitive assignments determine where they apply; "
    "these are NOT produced by the language model."
)

# Failure names the catalogue does not assign because it does not define the two
# generic gripper open/closed sequence-state predicates. The system prompt names
# this failure for them directly (see the GRIPPER OPEN/CLOSED SEQUENCE STATE
# section of SYSTEM_PROMPT), so it stays a valid link target.
NON_CATALOGUE_FAILURE_NAMES = ("ExecutionSequenceMismatch",)

# Refreshed every time failure definitions are loaded so the stage builders, which
# do not receive the definitions dict, can still see the latest catalog.
_HOLD_CONDITION_DETECTORS: list[dict] = list(DEFAULT_HOLD_CONDITION_DETECTORS)


def hold_condition_detectors(failure_definitions: dict | None = None) -> list[dict]:
    """Return the ordered hold-condition catalog.

    Prefers an explicit ``failure_definitions`` dict, then the cache populated by
    ``load_failure_definitions``, then the built-in default.
    """
    if isinstance(failure_definitions, dict):
        section = failure_definitions.get("hold_conditions") or {}
        detectors = section.get("detectors")
        if isinstance(detectors, list) and detectors:
            return detectors
    return _HOLD_CONDITION_DETECTORS or DEFAULT_HOLD_CONDITION_DETECTORS


def build_hold_condition_blocks(primitive=None, *, primitive_policy=None) -> list[dict]:
    """Per-stage hold-condition blocks built from the active detector catalog."""
    if primitive is not None and primitive_policy is None:
        primitive_policy = load_primitive_policy()
    blocks = []
    for entry in hold_condition_detectors():
        detector = entry.get("detector", "")
        if (primitive is not None and detector in primitive_policy
                and str(primitive).strip().lower() not in primitive_policy[detector]):
            continue
        failure = entry.get("failure") or detector
        definition = entry.get("definition", "")
        blocks.append(
            {
                "condition": entry.get("condition", ""),
                "detector": detector,
                "failure": failure,
                "label": entry.get("label") or detector,
                "definition": definition,
                "alternatives": [],
                "selected": True,
                "failure_links": [{"failure": failure, "reason": definition}],
            }
        )
    return blocks


def attach_hold_conditions(stages: list[dict]) -> list[dict]:
    """Assign hold detectors using each stage's primitive and the current config."""
    policy = load_primitive_policy()
    for stage in stages:
        stage["hold_conditions"] = build_hold_condition_blocks(
            stage.get("primitive") or "", primitive_policy=policy)
    return stages


def hold_conditions_for_saved_stage(saved_stage: dict, *, primitive_policy=None) -> list[dict]:
    """Apply config assignments; retain saved choices for unconfigured detectors."""
    if primitive_policy is None:
        primitive_policy = load_primitive_policy()
    saved = saved_stage.get("hold_conditions") if isinstance(saved_stage, dict) else None
    primitive = (saved_stage or {}).get("primitive") or ""
    full_blocks = build_hold_condition_blocks(primitive, primitive_policy=primitive_policy)
    if not isinstance(saved, list):
        return full_blocks
    kept_detectors = {
        str(item.get("detector") or "").strip()
        for item in saved
        if isinstance(item, dict) and item.get("selected", True) is not False
    }
    kept_detectors.discard("")
    return [block for block in full_blocks
            if block.get("detector") in primitive_policy
            or block.get("detector") in kept_detectors]


def catalogue_failure_names() -> list[str]:
    """Every failure name a generated condition may link to, in catalogue order.

    The verification-condition catalogue assigns one failure per rule, so it is
    also the authority on which failure names exist. Nothing here describes WHEN
    a failure applies: placement is stated once, by the catalogue rule itself.
    """
    from .prompts import load_condition_rules, _iter_condition_rules

    names: list[str] = []
    for _category, rule in _iter_condition_rules(load_condition_rules()):
        failure = str(rule.get("failure") or "").strip()
        if failure and failure not in names:
            names.append(failure)
    for extra in NON_CATALOGUE_FAILURE_NAMES:
        if extra not in names:
            names.append(extra)
    return names


def load_failure_definitions() -> dict:
    """Return the failure-definitions dict derived from the condition catalogue.

    The failure names come from ``generated_condition_rules.json`` and the hold
    conditions from ``DEFAULT_HOLD_CONDITION_DETECTORS`` above. No prose about
    where a condition belongs is supplied: the catalogue's placements are the
    single source of truth, so nothing can contradict them.
    """
    global _HOLD_CONDITION_DETECTORS
    _HOLD_CONDITION_DETECTORS = list(DEFAULT_HOLD_CONDITION_DETECTORS)
    return {
        "hold_conditions": {
            "description": HOLD_CONDITIONS_DESCRIPTION,
            "detectors": DEFAULT_HOLD_CONDITION_DETECTORS,
        },
        "failure_categories": {name: {"id": name} for name in catalogue_failure_names()},
    }


@lru_cache(maxsize=256)
def _predicate_template_pattern(template: str) -> re.Pattern:
    """Match a catalogue template with object arguments bound to concrete names."""
    compact = re.sub(r"\s+", "", template)
    parts = re.split(r"((?:<[^>]+>,)+\.\.\.|<[^>]+>)", compact)
    pattern = []
    for part in parts:
        if part.startswith("<") and part.endswith("..."):
            # Templates such as objects(<object_1>, <object_2>, ...) are variadic.
            pattern.append(r"[^(),]+(?:,[^(),]+)*")
        elif part.startswith("<") and part.endswith(">"):
            pattern.append(r"[^(),]+")
        else:
            pattern.append(re.escape(part))
    return re.compile("".join(pattern), re.IGNORECASE)


def catalogue_failure_links(text: str) -> list[dict]:
    """Look up a missing link from exact predicate templates in the current JSON.

    Unmatched or ambiguous predicates stay unlinked; no failure is guessed.
    """
    from .prompts import load_condition_rules, _iter_condition_rules

    compact = re.sub(r"\s+", "", text)
    if not compact:
        return []
    matches = {}
    for _category, rule in _iter_condition_rules(load_condition_rules()):
        if _predicate_template_pattern(rule["predicate_template"]).fullmatch(compact):
            matches[rule["failure"]] = str(rule.get("reason") or "")
    if len(matches) != 1:
        return []
    return [{"failure": failure, "reason": reason} for failure, reason in matches.items()]


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def scene_name_aliases(task_context) -> dict[str, str]:
    """{original_name -> name} from key_scene_objects, for renaming handles.

    key_scene_objects carries both the descriptive ``name`` and the raw
    simulator ``original_name`` (e.g. name 'basketball' / original_name 'ball').
    Only unambiguous, safe aliases are returned:
      - the alias is a bare identifier (multi-handle originals such as
        "dirt0,dirt1,dirt2" are split; anything else is dropped);
      - the alias is not itself some other object's descriptive name (renaming
        it would collide with a real object);
      - the alias does not point at two different names.
    """
    try:
        d = json.loads(task_context) if isinstance(task_context, str) else task_context
    except Exception:
        return {}
    if not isinstance(d, dict):
        return {}
    objects = [o for o in (d.get("key_scene_objects") or []) if isinstance(o, dict)]
    names = {str(o.get("name")).strip() for o in objects if o.get("name")}

    aliases: dict[str, str] = {}
    ambiguous: set[str] = set()
    for o in objects:
        name = str(o.get("name") or "").strip()
        original = o.get("original_name")
        if not name or not original:
            continue
        for token in re.split(r"[,/;\s]+", str(original)):
            token = token.strip()
            if not token or token == name or token in names:
                continue
            if not _IDENTIFIER_RE.match(token):
                continue
            for variant in {token, token.lower()} - names:
                if aliases.get(variant, name) != name:
                    ambiguous.add(variant)
                aliases[variant] = name
    for token in ambiguous:
        aliases.pop(token, None)
    return aliases


def iter_failure_definition_entries(failure_definitions: dict) -> list[tuple[str, dict]]:
    categories = failure_definitions.get("failure_categories")
    if isinstance(categories, dict):
        return [
            (str(details.get("id") or key), details)
            for key, details in categories.items()
            if isinstance(details, dict)
        ]

    return [
        (str(details.get("id") or key), details)
        for key, details in failure_definitions.items()
        if isinstance(details, dict)
    ]

def failure_definition_summaries(failure_definitions: dict) -> list[dict]:
    summaries = []
    for name, details in iter_failure_definition_entries(failure_definitions):
        summaries.append(
            {
                "id": name,
                "definition": details.get("definition", ""),
                "when_to_check": details.get("when_to_check") or details.get("when") or "",
            }
        )
    return summaries
