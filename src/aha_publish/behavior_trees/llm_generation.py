"""LLM client calls, response normalization, and generation orchestration."""

from aha_publish import paths

from contextvars import ContextVar
import json
import os
import re
import sys

# Import shared names directly from their source modules.
from .io_context import *          # also re-exports prompts + config names
from .conditions import *
from .failure_definitions import *
from .local_generation import *

# --- token-usage accounting -------------------------------------------------
# The streaming calls below used to discard the API-reported usage. We now record
# it per agent so a full BT build reports exactly what it cost (input vs output
# tokens, and a per-waypoint breakdown). Reset at the start of generate_conditions.
_TOKEN_USAGE: ContextVar[tuple[dict, ...]] = ContextVar("bt_token_usage", default=())


def _reset_token_usage() -> None:
    _TOKEN_USAGE.set(())


def _record_usage(agent: str, model: str, usage) -> None:
    """Append one {agent, model, input_tokens, output_tokens} record. `usage` is
    the OpenAI usage object; missing fields are stored as None."""
    if usage is None:
        return
    record = {
        "agent": agent,
        "model": model,
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }
    _TOKEN_USAGE.set(_TOKEN_USAGE.get() + (record,))


def format_bt_token_usage(n_waypoints: int | None = None) -> str:
    """Human-readable summary of the tokens spent in the current BT build."""
    records = _TOKEN_USAGE.get()
    if not records:
        return "[BT token usage: none recorded]"
    lines = ["[BT token usage]"]
    tot_in = tot_out = 0
    for rec in records:
        i = rec.get("input_tokens") or 0
        o = rec.get("output_tokens") or 0
        tot_in += i
        tot_out += o
        lines.append(f"  {rec['agent']:<8s} ({rec['model']}): input={i}  output={o}")
    lines.append(f"  TOTAL: input={tot_in}  output={tot_out}")
    if n_waypoints:
        lines.append(
            f"  per-waypoint ({n_waypoints} wps): "
            f"input={tot_in / n_waypoints:.0f}  output={tot_out / n_waypoints:.0f}"
        )
    return "\n".join(lines)


def build_client(provider: str):
    if provider != "openai":
        raise ValueError(f"Unsupported provider '{provider}'. Only OpenAI is supported.")
    missing = [
        package
        for package in OPENAI_REQUIRED_PACKAGES
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        packages = " ".join(missing)
        raise RuntimeError(
            f"Missing Python package(s): {packages}. Install with: "
            f"{sys.executable} -m pip install {packages}"
        )
    try:
        import openai
    except ModuleNotFoundError as e:
        package = e.name or "the missing OpenAI dependency"
        raise RuntimeError(
            f"Missing Python package '{package}'. Install with: "
            f"{sys.executable} -m pip install {package}"
        ) from e
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY.")
    os.environ["OPENAI_API_KEY"] = api_key
    return openai.OpenAI(api_key=api_key)


def generate_conditions_with_openai(client, user_message: str, system_prompt: str | None = None) -> dict:
    if system_prompt is None:
        system_prompt = get_system_prompt()
    full_text = ""
    print("[Agent 1: generating BT conditions...]\n", flush=True)
    stream = client.responses.create(
        model=GENERATOR_OPENAI_MODEL,
        max_output_tokens=CONDITION_MAX_OUTPUT_TOKENS,
        stream=True,
        text={
            "format": {
                "type": "json_schema",
                "name": "bt_conditions",
                "schema": CONDITION_RESPONSE_SCHEMA,
                "strict": True,
            }
        },
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    for event in stream:
        if hasattr(event, "type") and event.type == "response.output_text.delta":
            chunk = getattr(event, "delta", "") or ""
            print(chunk, end="", flush=True)
            full_text += chunk
        elif getattr(event, "type", "") == "response.completed":
            _record_usage("Agent1", GENERATOR_OPENAI_MODEL, getattr(event.response, "usage", None))
    print("\n", flush=True)
    return parse_json_response(full_text)

def _reviewer_enabled(reviewer: bool | None) -> bool:
    """Resolve whether Agent 2 runs: explicit flag wins, else BT_USE_REVIEWER env
    (default ON). Set BT_USE_REVIEWER=0 to disable globally."""
    if reviewer is not None:
        return reviewer
    return os.environ.get("BT_USE_REVIEWER", "1").strip().lower() not in ("0", "false", "no", "")


def build_reviewer_message(draft_stages: list, task_context: str | dict) -> str:
    """Build Agent 2's user message: a compact per-waypoint gripper/held summary
    (the authoritative ground truth) plus Agent 1's draft stages to audit."""
    try:
        ctx = json.loads(task_context) if isinstance(task_context, str) else (task_context or {})
    except Exception:
        ctx = {}
    gripper_lines = []
    for wp in (ctx.get("waypoints") or []):
        gripper_lines.append(
            f"  waypoint {wp.get('waypoint')}: gripper_state={wp.get('gripper_state')!r}, "
            f"held_object={wp.get('held_object')!r}"
        )
    objects = sorted({
        obj.get("name")
        for obj in (ctx.get("key_scene_objects") or [])
        if obj.get("name")
    })
    return (
        "Authoritative per-waypoint gripper / held-object ground truth:\n"
        + ("\n".join(gripper_lines) if gripper_lines else "  (none provided)")
        + "\n\nTask scene objects (only these names are valid):\n  "
        + (", ".join(objects) if objects else "(none listed)")
        + "\n\nAgent 1 draft stages to audit and correct:\n"
        + json.dumps({"stages": draft_stages}, indent=2)
    )


def review_conditions_with_openai(client, user_message: str) -> dict:
    full_text = ""
    print("[Agent 2: reviewing BT conditions...]\n", flush=True)
    stream = client.responses.create(
        model=OPENAI_MODEL,
        max_output_tokens=CONDITION_MAX_OUTPUT_TOKENS,
        stream=True,
        text={
            "format": {
                "type": "json_schema",
                "name": "bt_review",
                "schema": REVIEWER_RESPONSE_SCHEMA,
                "strict": True,
            }
        },
        input=[
            {"role": "system", "content": get_reviewer_system_prompt()},
            {"role": "user", "content": user_message},
        ],
    )
    for event in stream:
        if hasattr(event, "type") and event.type == "response.output_text.delta":
            chunk = getattr(event, "delta", "") or ""
            print(chunk, end="", flush=True)
            full_text += chunk
        elif getattr(event, "type", "") == "response.completed":
            _record_usage("Agent2", OPENAI_MODEL, getattr(event.response, "usage", None))
    print("\n", flush=True)
    return parse_json_response(full_text)


def review_conditions(draft_stages: list, task_context, client, provider: str) -> dict:
    """Run Agent 2 over Agent 1's raw stages and return its corrections.

    The caller handles reviewer failures by keeping the generated draft.
    """
    if provider != "openai":
        raise ValueError(f"Unsupported provider '{provider}'. Only OpenAI is supported.")
    if not draft_stages:
        return {"violations": [], "stages": draft_stages}
    user_message = build_reviewer_message(draft_stages, task_context)
    reviewed = review_conditions_with_openai(client, user_message)
    stages = reviewed.get("stages") or draft_stages
    violations = list(reviewed.get("violations", []))
    return {"violations": violations, "stages": stages}


def _print_review_report(violations: list) -> None:
    """Human-readable summary of Agent 2's findings."""
    if not violations:
        print("[Agent 2: no structural violations found]\n", flush=True)
        return
    print(f"[Agent 2: {len(violations)} violation(s) found and fixed]", flush=True)
    for v in violations:
        print(
            f"  - stage {v.get('stage')} [{v.get('rule')}]: {v.get('detail')} "
            f"-> {v.get('fix')}",
            flush=True,
        )
    print("", flush=True)


def _normalize_primitive(value) -> str | None:
    """Return the Step-1 waypoint primitive if it is one of the six allowed names."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v if v in WAYPOINT_PRIMITIVES else None


def saved_review_to_conditions(saved_data: dict) -> dict | None:
    """Load saved condition blocks without applying condition repair passes."""
    review = saved_data.get("review") or saved_data.get("generated") or {}
    saved_stages = review.get("stages") or []
    if not saved_stages:
        return None

    primitive_policy = load_primitive_policy()
    stages = []
    for index, stage in enumerate(saved_stages):
        stage_number = stage.get("stage", index)
        preconditions = merge_condition_blocks(stage.get("preconditions", []))
        postconditions = merge_condition_blocks(stage.get("postconditions", []))
        stages.append(
            {
                "stage": stage_number,
                "name": stage.get("name") or f"Stage {stage_number}",
                "summary": stage.get("summary", ""),
                "missing_description": stage.get("missing_description", ""),
                "primitive": _normalize_primitive(stage.get("primitive")),
                "preconditions": preconditions,
                "postconditions": postconditions,
                "hold_conditions": hold_conditions_for_saved_stage(
                    stage, primitive_policy=primitive_policy),
            }
        )

    return {
        "source": "saved_review",
        "saved_at": saved_data.get("saved_at"),
        "stages": stages,
    }


def _rename_condition_arguments(text: str, mapping: dict[str, str]) -> str:
    """Replace object identifiers in arguments with one lookup per token."""
    def replace_arguments(match: re.Match) -> str:
        arguments = re.sub(
            r"\b\w+\b",
            lambda token: mapping.get(token.group(0), token.group(0)),
            match.group(2),
        )
        return f"{match.group(1)}{arguments})"

    return re.sub(r"(\b\w+\s*\()([^()]*)\)", replace_arguments, text)


def _dedupe_object_found_args(text):
    def repl(m):
        seen, kept = set(), []
        for a in (x.strip() for x in m.group(1).split(",")):
            if a and a not in seen:
                seen.add(a)
                kept.append(a)
        return "Object_found(" + ", ".join(kept) + ")"
    return re.sub(r"Object_found\(([^)]*)\)", repl, text)


def normalize_scene_object_names(stages: list[dict], task_context) -> list[dict]:
    """Rewrite raw simulator handles to the descriptive key_scene_objects name.

    The BT must speak the description's ``name`` vocabulary (basketball, hoop),
    not the simulator's ``original_name`` handles (ball,
    basket_ball_hoop_respondable), since the conditions are checked visually.
    The system prompt asks for this; this pass enforces it deterministically on
    whatever the LLM emitted.

    Only predicate arguments are rewritten. Stage name/summary and the
    failure_links reasons are free text, where a handle like "lid", "target" or
    "stand" is usually an ordinary English word, so rewriting them would garble
    the prose.
    """
    rename = scene_name_aliases(task_context)
    if not rename:
        return stages
    for st in stages:
        if not isinstance(st, dict):
            continue
        for section in ("preconditions", "postconditions", "hold_conditions"):
            for c in (st.get(section) or []):
                if isinstance(c, dict) and isinstance(c.get("condition"), str):
                    c["condition"] = _dedupe_object_found_args(
                        _rename_condition_arguments(c["condition"], rename))
                if isinstance(c, dict) and isinstance(c.get("alternatives"), list):
                    c["alternatives"] = [
                        _dedupe_object_found_args(_rename_condition_arguments(a, rename))
                        if isinstance(a, str) else a
                        for a in c["alternatives"]
                    ]
    return stages


def generate_conditions(
    failure_definitions: dict,
    task_context: str,
    client,
    provider: str,
    reviewer_guidance: str = "",
    reviewer: bool | None = None,  # None -> BT_USE_REVIEWER env (default ON); True/False forces Agent 2 on/off
) -> dict:
    """Generate and optionally review conditions, normalize names, and attach monitors."""
    if provider != "openai":
        raise ValueError(f"Unsupported provider '{provider}'. Only OpenAI is supported.")
    _reset_token_usage()
    user_message = build_user_message(
        task_context,
        failure_definitions,
        reviewer_guidance,
    )
    _system_prompt = get_system_prompt()

    # Agent 1: generate draft
    draft_raw = generate_conditions_with_openai(client, user_message, _system_prompt)

    # Agent 2 reviews the draft before object-name normalization and hold monitors.
    # A reviewer error leaves the draft available for generation to continue.
    review_report = None
    if _reviewer_enabled(reviewer):
        try:
            review = review_conditions(
                draft_raw.get("stages", []), task_context, client, provider
            )
            draft_raw["stages"] = review["stages"]
            review_report = {"violations": review["violations"]}
            _print_review_report(review["violations"])
        except Exception as e:
            print(f"[Agent 2 reviewer skipped: {e}]\n", flush=True)

    # Token accounting: summarize what this build actually spent (per agent and
    # per waypoint) and carry the raw records on the result for programmatic use.
    try:
        _nwp = len(draft_raw.get("stages", []))
        print(format_bt_token_usage(_nwp), flush=True)
    except Exception:
        pass
    token_usage = {"records": list(_TOKEN_USAGE.get())}

    stages = normalize_scene_object_names(draft_raw.get("stages", []), task_context)
    attach_hold_conditions(stages)
    result = {"source": "api_generation_raw", "stages": stages}
    if review_report is not None:
        result["review"] = review_report
    result["token_usage"] = token_usage
    return result
