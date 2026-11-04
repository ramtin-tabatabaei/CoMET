# BT maker and reviewer user prompt templates

These are the user-message templates used by the current BT maker and reviewer. Braced placeholders mark content inserted at runtime. Each user message is sent alongside its separate system message; system prompts are defined in [prompts.py](src/aha_publish/behavior_trees/prompts.py).

## BT maker user prompt

Source: `build_user_message()` in [prompt_building.py](src/aha_publish/behavior_trees/prompt_building.py).

```text
TASK DESCRIPTION

{task_context}

VALID FAILURE NAMES

Link each condition to the failure name assigned by its catalogue rule in the
system prompt. These are the only names that exist:

{failures_text}{reviewer_section}

Generate the primitive classification, preconditions, and postconditions for
every waypoint.

Use the waypoint descriptions, gripper states, held-object states, scene
objects, and task relationships provided in the task description.

Follow all primitive definitions, allowed predicates, state-transition rules,
object-grounding requirements, timing constraints, failure-linking rules, and
JSON formatting requirements specified in the system prompt.

The structured gripper_state and held_object fields in the task description are
authoritative and override ambiguous free-text descriptions.

Each condition must be linked to exactly one of the failure names listed above.

Do not invent objects, predicates, waypoints, or failure modes.
```

| Placeholder | Runtime content |
| --- | --- |
| `{task_context}` | Complete task-description JSON, normally loaded from `outputs/descriptions/<task>_ALL_WAYPOINTS_COMBINED.openai.multimodal_analysis.json` and serialized with indentation. |
| `{failures_text}` | Valid failure names, one per line, each prefixed with `- `. These come from the generated predicate catalogue plus the generic `ExecutionSequenceMismatch` failure. |
| `{reviewer_section}` | Empty by default. When guidance is supplied, contains two leading newline characters, the `REVIEWER GUIDANCE` heading, a blank line, and the stripped guidance text. |

When enabled, `{reviewer_section}` expands to the following, preceded by a blank line:

```text
REVIEWER GUIDANCE

{reviewer_guidance}
```

Despite its name, this optional guidance is inserted into the **BT maker's user message**.

## Reviewer user prompt

Source: `build_reviewer_message()` in [llm_generation.py](src/aha_publish/behavior_trees/llm_generation.py).

The implementation assembles this message by joining strings. The placeholders below name those dynamic portions for documentation; they are not a separate template constant in the code.

```text
Authoritative per-waypoint gripper / held-object ground truth:
{gripper_summary}

Task scene objects (only these names are valid):
  {scene_object_names}

Agent 1 draft stages to audit and correct:
{draft_stages_json}
```

| Placeholder | Runtime content |
| --- | --- |
| `{gripper_summary}` | One line per task waypoint, in task-description order, formatted as shown below. If there are no waypoints, uses `  (none provided)`. |
| `{scene_object_names}` | Sorted, unique `name` values from `key_scene_objects`, joined with `, `. If no names are available, uses `(none listed)`. The template adds two leading spaces. |
| `{draft_stages_json}` | `json.dumps({"stages": draft_stages}, indent=2)`, containing the generator's draft stages for review. |

Each line in `{gripper_summary}` has two leading spaces and this format:

```text
  waypoint {waypoint}: gripper_state={gripper_state!r}, held_object={held_object!r}
```

The `!r` formatting uses Python representations: strings are quoted and a missing held object appears as `None`. The draft itself is JSON, so its null values appear as `null`.

The reviewer receives this compact state summary, the valid object names, and the draft stages. Its user message does not include the complete task-description JSON or image attachments.
