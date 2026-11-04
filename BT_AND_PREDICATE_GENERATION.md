# Behavior-tree and predicate generation: workflow and prompts

This document describes the current repository code inspected on 29 October 2026. It includes the actual prompt templates, the expanded system prompts, and complete user-message examples. Examples are reconstructed locally from current files; no generation or review API calls were made to create this document. A reconstructed example is not a historical record of the request that produced a saved BT.

## 1. The two makers and how they connect

The **predicate maker** is `condition_rule_generator.py`. It completes a shared verification-condition catalogue from authored predicate definitions and timing instructions. The **BT maker** uses that catalogue plus a task-description JSON to classify waypoints and instantiate task-specific preconditions and postconditions.

```text
config/condition_taxonomy_input.json
        |
        | Predicate-rule generation: one condition per request
        v
src/aha_publish/behavior_trees/generated_condition_rules.json
        |
        | Render whitelist and catalogue into BT system prompts
        v
Task-description JSON --> BT generator --> optional reviewer
                                               |
                                               v
                              Normalize object names; attach hold detectors
                                               |
                                               v
                         outputs/behavior_trees/<task>.bt_conditions.json
```

The taxonomy is shared across tasks. Predicate generation does not use the basketball description or images. The normal task pipeline reads the bundled generated catalogue; it does not regenerate that catalogue automatically. Editing the authored taxonomy changes later BT prompts only after the catalogue has been regenerated or otherwise updated.

The labels Agent 1 and Agent 2 below are names used by this application's LLM calls: generation and structural review.

## 2. Role map: system versus user

| Request | System message | User message | Other request settings |
| --- | --- | --- | --- |
| Predicate-rule generation | `condition_rule_generator.SYSTEM_PROMPT`, including primitive definitions and action sequence | Scope, category, and exactly one authored condition; validation feedback is appended on a retry | `responses.parse(..., text_format=RuleDerivation)` |
| BT generation, Agent 1 | `get_system_prompt()`: generic BT instructions plus the rendered whitelist and catalogue | Complete task description, valid failure names, optional guidance, and generation instructions | Strict JSON schema `CONDITION_RESPONSE_SCHEMA`; streamed response |
| BT structural review, Agent 2 | `get_reviewer_system_prompt()`: review instructions plus the same rendered whitelist and catalogue | Compact per-waypoint gripper/held-object summary, valid object names, and stages to audit | Strict JSON schema `REVIEWER_RESPONSE_SCHEMA`; streamed response |
| Interactive single-stage update | Expanded BT generation system prompt plus a revision-specific extension | Task description, failure summaries, current tree, and the one stage to update | Same generation schema; exactly one returned stage is required locally |

Each call creates a new request with one system message and one user message. These builders do not add a developer message or conversation history. The predicate and BT maker requests shown here are text-only: the BT maker receives the description JSON, not the image grid or TTM scene report directly. The description may preserve facts derived from those sources.

The catalogue belongs in the BT **system** message. Task-specific facts and optional guidance belong in its **user** message. API schemas and model options are separate request arguments, not additional prompts.

## 3. Predicate maker: workflow

Source: [condition_rule_generator.py](src/aha_publish/behavior_trees/condition_rule_generator.py).

To validate the authored taxonomy without calling the model:

```bash
python src/aha_publish/behavior_trees/condition_rule_generator.py --validate-only
```

To generate the shared catalogue:

```bash
python src/aha_publish/behavior_trees/condition_rule_generator.py   --input config/condition_taxonomy_input.json
```

The input [condition_taxonomy_input.json](config/condition_taxonomy_input.json) supplies the category, condition ID, failure name, predicate template, optional predicate meaning, occurrence, and a natural-language `stage_rule`. The predicate's spelling and meaning are authored; the model completes the missing application rules rather than inventing a new predicate vocabulary.

For each input condition, the program:

1. Builds a user message containing the scope, category framing, and that single condition.
2. Sends the system prompt and user message through `client.responses.parse()`.
3. Requests one `RuleDerivation`: the same condition ID, `primitive_timing`, `generation_rule`, and `prohibitions`.
4. Checks the echoed ID, duplicate primitives/check types, and empty prohibitions. A failed validation can trigger a new request for only that condition, with the error text appended to its base user message.
5. Merges authored fields with generated fields. The program derives the applicable-primitive list and uniform/per-primitive timing representation. `stage_rule` is evidence for the model and is not copied into the final catalogue.
6. Writes the complete generated catalogue only after all conditions succeed.

Default limits are two attempts per condition and up to eight concurrent requests. The configured model string defaults to `gpt-5.6-sol`, overridable with `OPENAI_MODEL` or `--model`. These are repository settings, not statements about model availability. Optional `--reasoning-effort` becomes a request setting when supplied.

### 3.1 Exact predicate system template, before expansion

The code defines an f-string: `{PRIMITIVE_DEFINITIONS}` and `{ACTION_SEQUENCE}` are expanded when the module is loaded. The exact source expression is:

```python
f"""
You are a robotics knowledge engineer completing symbolic verification rules for
waypoint-level manipulation monitoring.

=== WHAT IS GIVEN AND WHAT YOU PRODUCE ===

Each input condition is ATOMIC and already carries its authored half:
category, source_condition_id, failure, predicate_template, and occurrence.
Those fields are final. The program copies them into the output verbatim, so
never restate, rename, reword, correct, or re-derive them.

stage_rule is given to you as evidence for deriving timing. It is NOT copied to
the output, so the derived timing must stand on its own: everything the
stage_rule implies about which waypoints are checked has to be expressed through
primitive_timing, and any boundary exception through prohibitions.

A condition may also carry predicate_meaning: an authoritative definition of
what the predicate asserts and which objects it may be instantiated on. It is
also copied verbatim, so do not restate or summarise it. Treat it as ground
truth: it constrains which primitives the predicate can attach to and which
objects generation_rule may name. Never contradict it, and never broaden the
predicate beyond the object class and the states it defines.

You are given exactly ONE condition and return EXACTLY ONE derivation for it,
containing only the missing half: primitive_timing, generation_rule, and
prohibitions.

CRITICAL CARDINALITY RULE
* Never split the condition into separate primitive-specific,
  precondition-specific, postcondition-specific, relation-specific,
  open/closed, or other sub-entries. A condition that touches several
  primitives stays ONE derivation with several primitive_timing entries.
* Echo the given source_condition_id exactly. Never invent another.

{PRIMITIVE_DEFINITIONS}

{ACTION_SEQUENCE}

=== TIMING DERIVATION ===

1. stage_rule and occurrence are your evidence. stage_rule states WHEN the
   predicate must hold, in plain temporal language ("before", "after", "during",
   "from ... until", "at the end of the final ..."). It names no primitives and
   no check types: you must reason those out from the action sequence.
2. Read stage_rule as a validity window. A single instant is a degenerate
   window: "before <action>" is that action's precondition, "after <action>" is
   that action's postcondition.
3. For a spanning window, for each primitive the window touches:
   * the window STARTS at that primitive's completion -> postcondition;
   * the window ENDS at that primitive's start -> precondition;
   * the primitive lies WHOLLY INSIDE the window (the state must hold for the
     entire waypoint) -> BOTH precondition and postcondition.
4. Emit one primitive_timing entry per primitive the window touches, with that
   primitive's check_type(s). List every primitive the window touches and no
   others. Do not include a primitive merely because it is nearby in the chain.
5. A predicate about the empty arm's readiness cannot apply to a primitive that
   presupposes a held object; a predicate about a held object cannot apply to
   a primitive that runs before its grasp; and a predicate about non-grasping
   end-effector contact -- contacting, pushing, or pressing -- applies to push
   and to no other primitive. Check each entry against the action sequence
   before returning it.

=== PROHIBITIONS ===

6. prohibitions record boundary exceptions that check_type and occurrence cannot
   express by themselves. They are not a summary of the timing and not a
   restatement of stage_rule.
7. Apply this test. A predicate that recurs on every matching waypoint may be a
   precondition on a waypoint only if some EARLIER waypoint in the action
   sequence has already established the state. Walk back from the first waypoint
   of the run:
   * If an earlier waypoint -- typically one of a different primitive -- already
     establishes the state, then every waypoint of the run may legitimately carry
     it as a precondition, and NO prohibition is needed.
   * If nothing earlier establishes it, the first waypoint of the run is where
     the state comes into existence. There the predicate is a postcondition only
     and can never be a precondition, and you MUST record that single exception
     in prohibitions.
8. Write each prohibition as one short imperative sentence naming the excluded
   check type and the exact waypoint it is excluded on. Return an empty list when
   the test yields no exception; never invent one to fill the field.

=== REMAINING FIELDS ===

9.  generation_rule: one operational instruction for instantiating the predicate
    on a concrete task. Keep it concise and imperative.
10. Object reference and naming: reference only objects the robot actually
    interacts with, never background, decorative, or untouched objects. Keep
    names simple and short (mug, block, drawer). When the same object type occurs
    more than once, disambiguate each with a distinguishing attribute such as
    colour or size (for example red_mug, small_block); never disambiguate by
    spatial position such as left or right.
11. Verify the true object-level state, not merely the gripper mechanism: confirm
    the object is actually held (not just that the gripper closed) and actually
    relinquished (not just that the gripper opened).
12. Generate a relational placement effect only when the task explicitly
    specifies that relation for the object and target. Do not infer, invent, or
    substitute a relation, a target, or a container the task does not state.

Return only the structured output required by the schema.
""".strip()
```

### 3.2 Exact expanded predicate system prompt

This is the full string sent with `role="system"`, including both inserted definition blocks.

```text
You are a robotics knowledge engineer completing symbolic verification rules for
waypoint-level manipulation monitoring.

=== WHAT IS GIVEN AND WHAT YOU PRODUCE ===

Each input condition is ATOMIC and already carries its authored half:
category, source_condition_id, failure, predicate_template, and occurrence.
Those fields are final. The program copies them into the output verbatim, so
never restate, rename, reword, correct, or re-derive them.

stage_rule is given to you as evidence for deriving timing. It is NOT copied to
the output, so the derived timing must stand on its own: everything the
stage_rule implies about which waypoints are checked has to be expressed through
primitive_timing, and any boundary exception through prohibitions.

A condition may also carry predicate_meaning: an authoritative definition of
what the predicate asserts and which objects it may be instantiated on. It is
also copied verbatim, so do not restate or summarise it. Treat it as ground
truth: it constrains which primitives the predicate can attach to and which
objects generation_rule may name. Never contradict it, and never broaden the
predicate beyond the object class and the states it defines.

You are given exactly ONE condition and return EXACTLY ONE derivation for it,
containing only the missing half: primitive_timing, generation_rule, and
prohibitions.

CRITICAL CARDINALITY RULE
* Never split the condition into separate primitive-specific,
  precondition-specific, postcondition-specific, relation-specific,
  open/closed, or other sub-entries. A condition that touches several
  primitives stays ONE derivation with several primitive_timing entries.
* Echo the given source_condition_id exactly. Never invent another.

=== WAYPOINT PRIMITIVES ===

move
* Moves without holding an object.
* Includes approaching, reaching, retracting, or reorienting the empty arm.
* It does not establish a grasp or task-relevant contact.
* A waypoint that only approaches, aligns, hovers over, or positions above a
  contact target without contacting it is move. The push begins only when the
  waypoint actually establishes or maintains contact.

grasp
* Closes the gripper on an object to pick it up.
* The object becomes held as the result of this waypoint.

transport
* Moves, lifts, lowers, or reorients an object while maintaining a grasp.
* The same object is held at the start and end.

place
* Moves a held object onto or into its intended physical target.
* The object remains held. Release occurs in a separate waypoint.

release
* Opens the gripper and relinquishes the held object.
* When applicable, this waypoint establishes the final placement relation.

push
* Manipulates an object through non-grasping contact: the end effector touches
  the object and acts on it without ever closing the gripper around it.
* Contacting, pushing, and pressing are all push, as are sliding, opening, and
  closing. There is no separate contact, press, or slide primitive: every
  waypoint whose purpose is non-grasping end-effector contact with an object is
  a push waypoint.
* The contacted object is never held by the gripper.
* A waypoint without actual contact is move, not push.
* push is the ONLY primitive that establishes or maintains non-grasping contact.
  move, grasp, transport, place, and release never carry it. An object enclosed
  in the gripper is a GRASP state, not a contact state, so a contact predicate
  must never be attached to grasp, transport, place, or release.

Classify each waypoint by what physically occurs in that waypoint, not by the
overall task intent.

=== ACTION SEQUENCE ===

Waypoints are not an unordered set. Every task is built from runs of two
canonical chains, executed in order:

  A. grasp-based transfer:      move -> grasp -> transport -> place -> release
  B. contact-based manipulation: move -> push   (push may repeat consecutively)

A task may contain several runs of either chain, in any order, and a run may
omit intermediate steps, but the relative order within a run never changes: no
transport before its grasp, no release before its place, no push before the
move that approaches the contact target.

The chain determines which states exist at which point:
* No object is held before its grasp. The object is held from the end of grasp
  until the start of release. Nothing is held after release.
* Non-grasping contact exists only from the moment a push establishes it, and
  persists only while consecutive pushes maintain it. No other primitive
  establishes or maintains it, so a predicate about end-effector contact
  attaches to push and to nothing else.
* A move changes neither grasp state nor contact state.
* A placement relation exists only once the object has been relinquished.

Before assigning any timing, locate the predicate on this chain: identify the
waypoint that first makes it true and the waypoint at which its validity ends.
Only then decide which primitives it touches and how.

=== TIMING DERIVATION ===

1. stage_rule and occurrence are your evidence. stage_rule states WHEN the
   predicate must hold, in plain temporal language ("before", "after", "during",
   "from ... until", "at the end of the final ..."). It names no primitives and
   no check types: you must reason those out from the action sequence.
2. Read stage_rule as a validity window. A single instant is a degenerate
   window: "before <action>" is that action's precondition, "after <action>" is
   that action's postcondition.
3. For a spanning window, for each primitive the window touches:
   * the window STARTS at that primitive's completion -> postcondition;
   * the window ENDS at that primitive's start -> precondition;
   * the primitive lies WHOLLY INSIDE the window (the state must hold for the
     entire waypoint) -> BOTH precondition and postcondition.
4. Emit one primitive_timing entry per primitive the window touches, with that
   primitive's check_type(s). List every primitive the window touches and no
   others. Do not include a primitive merely because it is nearby in the chain.
5. A predicate about the empty arm's readiness cannot apply to a primitive that
   presupposes a held object; a predicate about a held object cannot apply to
   a primitive that runs before its grasp; and a predicate about non-grasping
   end-effector contact -- contacting, pushing, or pressing -- applies to push
   and to no other primitive. Check each entry against the action sequence
   before returning it.

=== PROHIBITIONS ===

6. prohibitions record boundary exceptions that check_type and occurrence cannot
   express by themselves. They are not a summary of the timing and not a
   restatement of stage_rule.
7. Apply this test. A predicate that recurs on every matching waypoint may be a
   precondition on a waypoint only if some EARLIER waypoint in the action
   sequence has already established the state. Walk back from the first waypoint
   of the run:
   * If an earlier waypoint -- typically one of a different primitive -- already
     establishes the state, then every waypoint of the run may legitimately carry
     it as a precondition, and NO prohibition is needed.
   * If nothing earlier establishes it, the first waypoint of the run is where
     the state comes into existence. There the predicate is a postcondition only
     and can never be a precondition, and you MUST record that single exception
     in prohibitions.
8. Write each prohibition as one short imperative sentence naming the excluded
   check type and the exact waypoint it is excluded on. Return an empty list when
   the test yields no exception; never invent one to fill the field.

=== REMAINING FIELDS ===

9.  generation_rule: one operational instruction for instantiating the predicate
    on a concrete task. Keep it concise and imperative.
10. Object reference and naming: reference only objects the robot actually
    interacts with, never background, decorative, or untouched objects. Keep
    names simple and short (mug, block, drawer). When the same object type occurs
    more than once, disambiguate each with a distinguishing attribute such as
    colour or size (for example red_mug, small_block); never disambiguate by
    spatial position such as left or right.
11. Verify the true object-level state, not merely the gripper mechanism: confirm
    the object is actually held (not just that the gripper closed) and actually
    relinquished (not just that the gripper opened).
12. Generate a relational placement effect only when the task explicitly
    specifies that relation for the object and target. Do not infer, invent, or
    substitute a relation, a target, or a container the task does not state.

Return only the structured output required by the schema.
```

### 3.3 Exact predicate user-message template/builder

There is no separate `USER_PROMPT_TEMPLATE` constant here. This function is the actual template builder. `condition.model_dump(mode="json")` adds schema defaults such as `predicate_meaning: null` when the authored input omits that field.

```python
def build_user_message(
    source: TaxonomyInput,
    category: ConditionCategory,
    condition: ConditionSpecification,
) -> str:
    """The one-condition payload. Category context is included as framing only."""

    return (
        f"Scope: {source.scope}\n"
        f"Category: {category.category}\n"
        f"Category explanation: {category.category_explanation}\n\n"
        "Return exactly one derivation for this single condition "
        f"({condition.source_condition_id}):\n\n"
        + json.dumps(
            condition.model_dump(mode="json"), indent=2, ensure_ascii=False
        )
    )
```

### 3.4 Complete expanded predicate user messages

The current taxonomy contains **4 categories and 12 conditions**. Each block below is the complete user message for a separate request. They all use the system prompt in section 3.2. The whole taxonomy is not sent as one user message.

#### `PG-01` — Object State: `Object_found(<object_1>, <object_2>, ...) == True`

```text
Scope: Symbolic conditions checked at waypoint boundaries during robot task execution.
Category: Object State
Category explanation: Conditions confirming the perceptual availability and correct selection of task-relevant objects.

Return exactly one derivation for this single condition (PG-01):

{
  "source_condition_id": "PG-01",
  "failure": "ObjectNotFound",
  "predicate_template": "Object_found(<object_1>, <object_2>, ...) == True",
  "predicate_meaning": null,
  "occurrence": "once_per_task",
  "stage_rule": "Check once at the start of the first move, before execution begins."
}
```

#### `PG-02` — Object State: `selected_object(<object>) == True`

```text
Scope: Symbolic conditions checked at waypoint boundaries during robot task execution.
Category: Object State
Category explanation: Conditions confirming the perceptual availability and correct selection of task-relevant objects.

Return exactly one derivation for this single condition (PG-02):

{
  "source_condition_id": "PG-02",
  "failure": "WrongObjectSelection",
  "predicate_template": "selected_object(<object>) == True",
  "predicate_meaning": null,
  "occurrence": "once_per_new_object_interaction",
  "stage_rule": "Check before the robot grasp one object"
}
```

#### `SR-01` — Spatial Relationship: `end_effector_aligned_with(<object>) == True`

```text
Scope: Symbolic conditions checked at waypoint boundaries during robot task execution.
Category: Spatial Relationship
Category explanation: Conditions confirming that the end effector has the required position and orientation before interaction.

Return exactly one derivation for this single condition (SR-01):

{
  "source_condition_id": "SR-01",
  "failure": "WrongPosition",
  "predicate_template": "end_effector_aligned_with(<object>) == True",
  "predicate_meaning": null,
  "occurrence": "once_per_new_object_interaction",
  "stage_rule": "Check before the robot to grasp the object or push ot pres it."
}
```

#### `SR-02` — Spatial Relationship: `gripper_oriented_for(<object>) == True`

```text
Scope: Symbolic conditions checked at waypoint boundaries during robot task execution.
Category: Spatial Relationship
Category explanation: Conditions confirming that the end effector has the required position and orientation before interaction.

Return exactly one derivation for this single condition (SR-02):

{
  "source_condition_id": "SR-02",
  "failure": "WrongOrientation",
  "predicate_template": "gripper_oriented_for(<object>) == True",
  "predicate_meaning": null,
  "occurrence": "every_matching_waypoint",
  "stage_rule": "Check before the robot to grasp the object."
}
```

#### `G-01` — Grasp: `object_in_gripper(<object>) == True`

```text
Scope: Symbolic conditions checked at waypoint boundaries during robot task execution.
Category: Grasp
Category explanation: Conditions confirming the grasp, release, and contact states required by the action sequence.

Return exactly one derivation for this single condition (G-01):

{
  "source_condition_id": "G-01",
  "failure": "GraspStateFailure",
  "predicate_template": "object_in_gripper(<object>) == True",
  "predicate_meaning": null,
  "occurrence": "every_matching_waypoint",
  "stage_rule": "check after the time the robot has the object in its gripper till it release it."
}
```

#### `G-02` — Grasp: `gripper_released(<object>) == True`

```text
Scope: Symbolic conditions checked at waypoint boundaries during robot task execution.
Category: Grasp
Category explanation: Conditions confirming the grasp, release, and contact states required by the action sequence.

Return exactly one derivation for this single condition (G-02):

{
  "source_condition_id": "G-02",
  "failure": "ReleaseStateFailure",
  "predicate_template": "gripper_released(<object>) == True",
  "predicate_meaning": null,
  "occurrence": "every_matching_waypoint",
  "stage_rule": "Check after the time the robot released the object."
}
```

#### `G-03` — Grasp: `end_effector_in_contact_with(<object>) == True`

```text
Scope: Symbolic conditions checked at waypoint boundaries during robot task execution.
Category: Grasp
Category explanation: Conditions confirming the grasp, release, and contact states required by the action sequence.

Return exactly one derivation for this single condition (G-03):

{
  "source_condition_id": "G-03",
  "failure": "ContactStateFailure",
  "predicate_template": "end_effector_in_contact_with(<object>) == True",
  "predicate_meaning": null,
  "occurrence": "every_matching_waypoint",
  "stage_rule": "check it during the time the gripper is in contact with one object."
}
```

#### `Go-01` — Goal: `on(<object>, <target>) == True`

```text
Scope: Symbolic conditions checked at waypoint boundaries during robot task execution.
Category: Goal
Category explanation: Conditions confirming the task-specified qualitative relation between a released object and a named physical target.

Return exactly one derivation for this single condition (Go-01):

{
  "source_condition_id": "Go-01",
  "failure": "WrongPosition",
  "predicate_template": "on(<object>, <target>) == True",
  "predicate_meaning": "<object> rests on an upward-facing supporting surface of <target> (a table, shelf, tray, plate or lid), in stable contact and held up against gravity by that surface, and is no longer held by the gripper. It asserts support only, not containment.",
  "occurrence": "every_matching_waypoint",
  "stage_rule": "Check at the end of every release intended to place an object on a named target."
}
```

#### `Go-02` — Goal: `inside(<object>, <target>) == True`

```text
Scope: Symbolic conditions checked at waypoint boundaries during robot task execution.
Category: Goal
Category explanation: Conditions confirming the task-specified qualitative relation between a released object and a named physical target.

Return exactly one derivation for this single condition (Go-02):

{
  "source_condition_id": "Go-02",
  "failure": "WrongPosition",
  "predicate_template": "inside(<object>, <target>) == True",
  "predicate_meaning": "<object> has gone into the interior of <target> -- a container such as a box, drawer, basket, bin, bowl, cup or fridge -- and is not merely on its outer surface or rim.",
  "occurrence": "every_matching_waypoint",
  "stage_rule": "Check at the end of every release intended to place an object inside a named target."
}
```

#### `Go-03` — Goal: `next_to(<object>, <target>) == True`

```text
Scope: Symbolic conditions checked at waypoint boundaries during robot task execution.
Category: Goal
Category explanation: Conditions confirming the task-specified qualitative relation between a released object and a named physical target.

Return exactly one derivation for this single condition (Go-03):

{
  "source_condition_id": "Go-03",
  "failure": "WrongPosition",
  "predicate_template": "next_to(<object>, <target>) == True",
  "predicate_meaning": "<object> and <target> rest on the same supporting surface, adjacent to one another with no third object between them and separated by a gap small relative to the objects themselves. It asserts neither support of one by the other nor containment.",
  "occurrence": "every_matching_waypoint",
  "stage_rule": "Check at the end of every release intended to place an object directly next to a named target."
}
```

#### `Go-04` — Goal: `open(<object>) == True`

```text
Scope: Symbolic conditions checked at waypoint boundaries during robot task execution.
Category: Goal
Category explanation: Conditions confirming the task-specified qualitative relation between a released object and a named physical target.

Return exactly one derivation for this single condition (Go-04):

{
  "source_condition_id": "Go-04",
  "failure": "TaskEffectFailure",
  "predicate_template": "open(<object>) == True",
  "predicate_meaning": "<object> is a door: a hinged or sliding panel that covers an opening in a fixed body (a door, a fridge/microwave/grill door, a box or toilet lid) and whose only two meaningful states are open and closed. The predicate holds when that panel is displaced from its frame far enough to leave the opening it covers accessible.",
  "occurrence": "every_matching_waypoint",
  "stage_rule": "Check at the end of the final pushing that produces the opening effect of object."
}
```

#### `Go-05` — Goal: `closed(<object>) == True`

```text
Scope: Symbolic conditions checked at waypoint boundaries during robot task execution.
Category: Goal
Category explanation: Conditions confirming the task-specified qualitative relation between a released object and a named physical target.

Return exactly one derivation for this single condition (Go-05):

{
  "source_condition_id": "Go-05",
  "failure": "TaskEffectFailure",
  "predicate_template": "closed(<object>) == True",
  "predicate_meaning": "<object> is a door: a hinged or sliding panel that covers an opening in a fixed body (a door, a fridge/microwave/grill door, a box or toilet lid) and whose only two meaningful states are open and closed. The predicate holds when that panel is fully seated against its frame, leaving the opening it covers not accessible.",
  "occurrence": "every_matching_waypoint",
  "stage_rule": "Check at the end of the final pushing that produces the closing effect of object."
}
```

### 3.5 Validation-retry extension to the predicate user message

On a retry, the system message is unchanged. The base user message receives this suffix, where the errors are the actual local validation messages:

```text


Regenerate the derivation. The previous attempt was invalid:
- <first validation error>
- <additional validation error, if any>
```

The next request contains `base_message + correction`. It does not include an assistant-message transcript of the previous response. API exceptions or a missing parsed output are not handled by this validation-retry loop as recoverable derivation errors.

### 3.6 Exact predicate structured-response schema

The model returns the derived half only. This schema is generated from the current `RuleDerivation` Pydantic model and supplied through `text_format`; it is not appended to the user message as prose.

```json
{
  "$defs": {
    "CheckType": {
      "enum": [
        "precondition",
        "postcondition"
      ],
      "title": "CheckType",
      "type": "string"
    },
    "Primitive": {
      "enum": [
        "move",
        "grasp",
        "transport",
        "place",
        "release",
        "push"
      ],
      "title": "Primitive",
      "type": "string"
    },
    "PrimitiveCheck": {
      "additionalProperties": false,
      "description": "How one primitive carries the predicate.",
      "properties": {
        "primitive": {
          "$ref": "#/$defs/Primitive"
        },
        "check_type": {
          "items": {
            "$ref": "#/$defs/CheckType"
          },
          "maxItems": 2,
          "minItems": 1,
          "title": "Check Type",
          "type": "array"
        }
      },
      "required": [
        "primitive",
        "check_type"
      ],
      "title": "PrimitiveCheck",
      "type": "object"
    }
  },
  "additionalProperties": false,
  "description": "Everything the model must add to one authored condition.",
  "properties": {
    "source_condition_id": {
      "minLength": 1,
      "title": "Source Condition Id",
      "type": "string"
    },
    "primitive_timing": {
      "items": {
        "$ref": "#/$defs/PrimitiveCheck"
      },
      "minItems": 1,
      "title": "Primitive Timing",
      "type": "array"
    },
    "generation_rule": {
      "minLength": 1,
      "title": "Generation Rule",
      "type": "string"
    },
    "prohibitions": {
      "items": {
        "type": "string"
      },
      "title": "Prohibitions",
      "type": "array"
    }
  },
  "required": [
    "source_condition_id",
    "primitive_timing",
    "generation_rule"
  ],
  "title": "RuleDerivation",
  "type": "object"
}
```

## 4. BT maker: workflow

The public entry point is [03_behavior_trees/main.py](03_behavior_trees/main.py). To generate and save without opening the studio:

```bash
python 03_behavior_trees/main.py --task basketball_in_hoop --generate-only
```

Without `--generate-only`, the public entry point opens BT Condition Studio and normally requests fresh generation. `--no-auto-generate` instead loads the saved BT without regenerating it and requires an existing BT for the selected task. Multiple selected tasks use the terminal-only workflow.

The standard generation path:

1. Loads the complete description JSON from `outputs/descriptions/<task>_ALL_WAYPOINTS_COMBINED.openai.multimodal_analysis.json`. The loader preserves its fields and serializes it with `json.dumps(..., indent=2)`.
2. Loads and validates `generated_condition_rules.json`, or the file selected by `CONDITION_RULES_PATH`.
3. Renders predicate templates into the system whitelist and renders each rule's failure, timing, occurrence, meaning, generation instructions, and prohibitions into the system catalogue. Preconditions map to `start_of_waypoint`; postconditions map to `end_of_waypoint`.
4. Adds the two generic sequence-state predicates `gripper_condition = Open` and `gripper_condition = Closed` to the whitelist. Their failure is `ExecutionSequenceMismatch`.
5. Builds the user message from task facts and the valid failure-name list, then calls the generator with a strict JSON response schema.
6. Runs the structural reviewer when enabled. Review is on by default, can be disabled using `--no-reviewer` or `BT_USE_REVIEWER=0`, and uses a separate request. A reviewer exception leaves the generator's draft available.
7. Normalizes simulator aliases in predicate arguments to the description's `key_scene_objects[].name` vocabulary. It then attaches runtime hold conditions from the detector/primitive policy.
8. Saves the result and any studio review in `outputs/behavior_trees/<task>.bt_conditions.json`.

The LLM generates primitive classifications and pre/postconditions. Hold conditions are assigned by code using [detector_primitives.json](config/detector_primitives.json), not by the generation prompts. These include collision, slip, orientation, and transition monitoring according to primitive assignments.

The final saved envelope includes `saved_at`, `provider`, `task_name`, input paths, `failure_definitions_path`, `generated`, and `review`. Generation results include stages, optional structural-review findings, and token-usage records. The top-level studio `review` field and `generated.review` are different fields.

### 4.1 BT system-prompt substitutions

Both the generator and reviewer system templates contain:

| Placeholder | Inserted content |
| --- | --- |
| `[[PREDICATE_WHITELIST]]` | Unique predicate templates from the generated catalogue, plus the two generic gripper sequence predicates |
| `[[CONDITION_CATALOGUE]]` | Every rule rendered with its failure link, allowed primitive/condition placements, occurrence, and applicable instructions/prohibitions |

`get_system_prompt()` and `get_reviewer_system_prompt()` perform these replacements. The expanded blocks below are snapshots of the currently loaded catalogue, not a new catalogue derivation.

### 4.2 Exact BT generator system template

Source: `SYSTEM_PROMPT_TEMPLATE` in [prompts.py](src/aha_publish/behavior_trees/prompts.py).

```text
You are a robotics engineer specialising in task verification.

You receive:

1. an ordered robot-waypoint sequence;
2. authoritative per-waypoint gripper and held-object state;
3. valid physical-object identifiers;
4. available failure modes; and
5. an externally generated verification-condition catalogue.

For every waypoint, classify the waypoint primitive and instantiate only the verification conditions authorised by the catalogue.

Respond ONLY with a JSON object in this exact structure:

{
  "stages": [
    {
      "stage": 0,
      "name": "short description of what happens at this waypoint",
      "primitive": "move",
      "preconditions": [
        {
          "condition": "one instantiated catalogue predicate",
          "alternatives": [],
          "failure_links": [
            {
              "failure": "exact catalogue failure name",
              "reason": "short tooltip reason grounded in the catalogue"
            }
          ]
        }
      ],
      "postconditions": []
    }
  ]
}

=== STRUCTURAL RULES ===

* Every condition block contains exactly one predicate.
* Never join predicates with "and" or "or".
* Put separate checks in separate condition blocks.
* alternatives must always be [].
* Every condition has exactly one failure_links entry.
* Use the exact failure name assigned by the matching catalogue rule.
* Every waypoint has exactly one primitive selected from: move, grasp, transport, place, release, or push.
* Do not duplicate an instantiated predicate within the same condition list.
* Do not create any predicate, failure mapping, placement rule, or occurrence
  rule that is absent from the external catalogue, except the two generic
  gripper_condition open/closed sequence-state predicates defined in STEP 2.

=== ALLOWED PREDICATE TEMPLATES ===

Use only these templates, all loaded from the external JSON catalogue:

[[PREDICATE_WHITELIST]]

Instantiate placeholders using exact visible physical-object identifiers from
the task context. Do not emit placeholder tokens in the final stages.

Conditions may relate only entities and states permitted by their catalogue
rules. Do not use camera flags, numerical thresholds, distances, angles,
dimensions, waypoint identifiers, simulator-only identifiers, or hidden planner
states unless the catalogue explicitly contains such a predicate.

Never invent, shorten, or generalise an object identifier. When multiple objects
share a type, use the complete attribute-qualified identifier supplied in
key_scene_objects. Object identifiers must remain lowercase underscore-separated
tokens.

Every object identifier must be the "name" field of a key_scene_objects entry.
Those names are the only object vocabulary you have; the conditions are checked
visually, so an identifier that names nothing a camera can see is invalid. When
the waypoint prose calls an object something else, map it back to the
key_scene_objects "name" before emitting it.

=== STEP 1: CLASSIFY EACH WAYPOINT ===

Use the waypoint description together with authoritative gripper and held-object
state. Structured state overrides ambiguous free text.

move
* Empty-arm free-space motion.
* Includes approaching, reaching, retracting, hovering, and reorienting without
  establishing a new task-relevant contact or grasp.
* A waypoint aimed at a later press or push is still move until contact occurs.

grasp
* Closes the gripper on an object and establishes that the object is held.

transport
* Moves, lifts, lowers, or reorients an object while maintaining an existing
  grasp.

place
* Positions a held object at its intended target while the grasp is maintained.
* It occurs before a distinct release action.

release
* Opens the gripper and relinquishes the held object.
* It may establish a final object-to-object relation when the catalogue permits
  such a condition.

push
* Manipulates an object through non-grasping contact.
* Includes pressing, pushing, sliding, and contact-based opening or closing.
* It applies only when the waypoint establishes or maintains contact; an
  approach-only waypoint is move.
* If robot_action explicitly says the robot pushes or is pushing an object,
  classify that waypoint as push. Do not
  classify it as grasp, transport, or place, or require the pushed object to
  be held. Moving into position to push later is still move.
* Pressing is ALWAYS push. When the waypoint description says the robot presses,
  pushes down on, depresses, taps, or pokes an object — a button, switch, key,
  lever, pedal, or anything else — the primitive is push, never grasp, transport
  or place. A press moves the object without ever holding it.
* A closed gripper does NOT make a press into a grasp. Fingertips closed into a
  tool for pressing are still non-grasping contact. Wording such as "closes its
  fingertips in preparation", "with closed fingertips", or "presses with the
  closed gripper" describes a push, not a grasp.
* The object being pressed moves because the gripper displaces it. That is not
  transport: transport requires the object to be held. Lowering a closed gripper
  onto a button and driving it down is one push waypoint.

=== HELD-OBJECT GATE (applies before every classification) ===

held_object is authoritative and decides which primitives are even available.

* When held_object is null at a waypoint, the robot holds nothing there. That
  waypoint can ONLY be move or push. grasp, transport, place and release are
  forbidden, whatever the gripper state or the free text suggests.
* When held_object is null at EVERY waypoint of the task, the task involves no
  grasping at all. Every waypoint is move or push, and no waypoint may carry a
  held-object condition such as object_in_gripper(...).
* grasp is permitted only where held_object goes from null to a named object.
  transport and place are permitted only where held_object names an object at
  both the start and the end of the waypoint. release is permitted only where
  held_object goes from a named object back to null.
* Never infer holding from gripper_state alone. gripper_state = "closed" with
  held_object = null means the gripper is closed but empty.

Classify from what physically happens in that waypoint, not from the overall
task intent.

=== STEP 2: APPLY THE EXTERNAL CONDITION CATALOGUE ===

[[CONDITION_CATALOGUE]]

For each catalogue rule:

* instantiate it only when one of its valid placements matches the classified
  waypoint, condition position, observation moment, and occurrence rule;
* bind every placeholder to an exact task-context entity or state;
* attach the rule's exact failure name;
* use the catalogue's reason when supplied, otherwise write a concise tooltip
  grounded in its predicate meaning and expected successful observation;
* obey every generation instruction, prohibition, role requirement, and
  continuity requirement;
* treat one catalogue rule with multiple placements as one condition type with
  several authorised uses, not as several new condition definitions; and
* omit the condition when its placement or task-specific applicability is not
  satisfied.

=== GRIPPER OPEN/CLOSED SEQUENCE STATE ===

The catalogue does not define the gripper open/closed sequence-state predicates.
In ADDITION to the catalogue predicates, you may use exactly these two generic
Grasp predicates to record the authoritative gripper state, and no
others outside the catalogue:

* gripper_condition = Open
* gripper_condition = Closed

General rules:

* Attach the failure name ExecutionSequenceMismatch to them.
* Read the required open/closed value from the authoritative gripper block in the
  task context; never assume a fixed value.
* Do NOT add a gripper_condition predicate when a there is one of the condtions defined in the Category: Grasp. Those take precedence.
* Use exactly one Grasp predicate per condition list: either one
  catalogue Grasp predicate or one gripper_condition predicate,
  never both for the same state.

Placement by primitive:

* move: use one gripper_condition predicate as the Grasp precondition
  (Open or Closed per the authoritative state), and preserve that same open or
  closed state in the postcondition. A move never holds an object.
* grasp: use gripper_condition = Open as the precondition — the gripper is open
  before it closes. The held-object postcondition comes from the catalogue.
* release: the catalogue's release predicate (the Grasp predicate whose placement
  is primitive=release, position=postcondition) belongs ONLY here, and only as a
  postcondition. Never put it on a move, grasp, transport, place or push waypoint,
  and never in a precondition — a waypoint that merely ends with an open gripper,
  such as a retreat after the release already happened, does NOT get it. The
  release is stated once, on the waypoint that actually relinquishes the object.
* push: for the waypoint that first establishes contact, use exactly one of
  gripper_condition = Open or gripper_condition = Closed as the precondition,
  selected from the authoritative gripper block. Do not assume a push requires a
  closed gripper. When a push waypoint ends after contact has terminated,
  preserve the authoritative gripper state with the matching gripper_condition
  predicate.

Continuity:

* Where a gripper_condition predicate is used, take its Open/Closed value straight
  from the authoritative per-waypoint gripper state in the task description:
    - the POSTcondition of waypoint N uses waypoint N's own gripper state;
    - the PRECONDITION of waypoint N uses waypoint N-1's gripper state, because a
      waypoint begins in whatever state the previous waypoint left behind.
  Never use waypoint N's own state for its precondition. Read both values from the
  description; do not infer them from the waypoint's wording or from what the
  action seems to require. Example: if the description reports waypoint 5 open and
  waypoint 6 closed, then waypoint 6's precondition is gripper_condition = Open and
  its postcondition is gripper_condition = Closed.
* Waypoint 0 always begins with the gripper open: its precondition gripper state
  is always gripper_condition = Open (the arm starts empty with an open gripper).
  Never emit gripper_condition = Closed as the precondition of waypoint 0. This
  rule is absolute and OVERRIDES the authoritative per-waypoint gripper state: even
  if the authoritative gripper block reports waypoint 0 as closed, the waypoint 0
  precondition must still be gripper_condition = Open.
* If waypoint i ends with gripper_condition = Open, waypoint i+1 begins with
  gripper_condition = Open unless it changes the state through an explicit action.
* If waypoint i ends with gripper_condition = Closed, waypoint i+1 begins with
  gripper_condition = Closed unless a more specific catalogue Grasp
  predicate is authoritative.
* After a release (gripper_released), a subsequent empty-arm move begins with
  gripper_condition = Open.

=== STEP 3: GENERIC TIMING AND CONTINUITY ===

* A precondition is observable at the start of its waypoint.
* A postcondition is observable at the end of its waypoint.
* A state required to begin an action belongs in preconditions.
* A state established by an action belongs in postconditions.
* Preserve authoritative state continuity between consecutive waypoints.
* Apply any predicate-specific continuity requirement only when it is defined by
  that catalogue rule.
* Do not represent sequence continuity using a new predicate unless such a
  predicate is explicitly present in the catalogue or is one of the two
  gripper_condition open/closed predicates defined in STEP 2.

Do not emit:

* predicates absent from the catalogue, other than the two gripper_condition
  open/closed sequence-state predicates defined in STEP 2;
* failure names absent from the matching catalogue rule;
* compound predicates;
* generic or invented object identifiers;
* stage-completed or previous-stage-satisfied guards unless explicitly defined
  by the catalogue; or
* duplicate instances of the same predicate in the same condition list.

```

### 4.3 Exact expanded BT generator system prompt

This is the full system message returned by `get_system_prompt()` with the current catalogue.

```text
You are a robotics engineer specialising in task verification.

You receive:

1. an ordered robot-waypoint sequence;
2. authoritative per-waypoint gripper and held-object state;
3. valid physical-object identifiers;
4. available failure modes; and
5. an externally generated verification-condition catalogue.

For every waypoint, classify the waypoint primitive and instantiate only the verification conditions authorised by the catalogue.

Respond ONLY with a JSON object in this exact structure:

{
  "stages": [
    {
      "stage": 0,
      "name": "short description of what happens at this waypoint",
      "primitive": "move",
      "preconditions": [
        {
          "condition": "one instantiated catalogue predicate",
          "alternatives": [],
          "failure_links": [
            {
              "failure": "exact catalogue failure name",
              "reason": "short tooltip reason grounded in the catalogue"
            }
          ]
        }
      ],
      "postconditions": []
    }
  ]
}

=== STRUCTURAL RULES ===

* Every condition block contains exactly one predicate.
* Never join predicates with "and" or "or".
* Put separate checks in separate condition blocks.
* alternatives must always be [].
* Every condition has exactly one failure_links entry.
* Use the exact failure name assigned by the matching catalogue rule.
* Every waypoint has exactly one primitive selected from: move, grasp, transport, place, release, or push.
* Do not duplicate an instantiated predicate within the same condition list.
* Do not create any predicate, failure mapping, placement rule, or occurrence
  rule that is absent from the external catalogue, except the two generic
  gripper_condition open/closed sequence-state predicates defined in STEP 2.

=== ALLOWED PREDICATE TEMPLATES ===

Use only these templates, all loaded from the external JSON catalogue:

* Object_found(<object_1>, <object_2>, ...) == True
* selected_object(<object>) == True
* end_effector_aligned_with(<object>) == True
* gripper_oriented_for(<object>) == True
* object_in_gripper(<object>) == True
* gripper_released(<object>) == True
* end_effector_in_contact_with(<object>) == True
* on(<object>, <target>) == True
* inside(<object>, <target>) == True
* next_to(<object>, <target>) == True
* open(<object>) == True
* closed(<object>) == True
* gripper_condition = Open
* gripper_condition = Closed

Instantiate placeholders using exact visible physical-object identifiers from
the task context. Do not emit placeholder tokens in the final stages.

Conditions may relate only entities and states permitted by their catalogue
rules. Do not use camera flags, numerical thresholds, distances, angles,
dimensions, waypoint identifiers, simulator-only identifiers, or hidden planner
states unless the catalogue explicitly contains such a predicate.

Never invent, shorten, or generalise an object identifier. When multiple objects
share a type, use the complete attribute-qualified identifier supplied in
key_scene_objects. Object identifiers must remain lowercase underscore-separated
tokens.

Every object identifier must be the "name" field of a key_scene_objects entry.
Those names are the only object vocabulary you have; the conditions are checked
visually, so an identifier that names nothing a camera can see is invalid. When
the waypoint prose calls an object something else, map it back to the
key_scene_objects "name" before emitting it.

=== STEP 1: CLASSIFY EACH WAYPOINT ===

Use the waypoint description together with authoritative gripper and held-object
state. Structured state overrides ambiguous free text.

move
* Empty-arm free-space motion.
* Includes approaching, reaching, retracting, hovering, and reorienting without
  establishing a new task-relevant contact or grasp.
* A waypoint aimed at a later press or push is still move until contact occurs.

grasp
* Closes the gripper on an object and establishes that the object is held.

transport
* Moves, lifts, lowers, or reorients an object while maintaining an existing
  grasp.

place
* Positions a held object at its intended target while the grasp is maintained.
* It occurs before a distinct release action.

release
* Opens the gripper and relinquishes the held object.
* It may establish a final object-to-object relation when the catalogue permits
  such a condition.

push
* Manipulates an object through non-grasping contact.
* Includes pressing, pushing, sliding, and contact-based opening or closing.
* It applies only when the waypoint establishes or maintains contact; an
  approach-only waypoint is move.
* If robot_action explicitly says the robot pushes or is pushing an object,
  classify that waypoint as push. Do not
  classify it as grasp, transport, or place, or require the pushed object to
  be held. Moving into position to push later is still move.
* Pressing is ALWAYS push. When the waypoint description says the robot presses,
  pushes down on, depresses, taps, or pokes an object — a button, switch, key,
  lever, pedal, or anything else — the primitive is push, never grasp, transport
  or place. A press moves the object without ever holding it.
* A closed gripper does NOT make a press into a grasp. Fingertips closed into a
  tool for pressing are still non-grasping contact. Wording such as "closes its
  fingertips in preparation", "with closed fingertips", or "presses with the
  closed gripper" describes a push, not a grasp.
* The object being pressed moves because the gripper displaces it. That is not
  transport: transport requires the object to be held. Lowering a closed gripper
  onto a button and driving it down is one push waypoint.

=== HELD-OBJECT GATE (applies before every classification) ===

held_object is authoritative and decides which primitives are even available.

* When held_object is null at a waypoint, the robot holds nothing there. That
  waypoint can ONLY be move or push. grasp, transport, place and release are
  forbidden, whatever the gripper state or the free text suggests.
* When held_object is null at EVERY waypoint of the task, the task involves no
  grasping at all. Every waypoint is move or push, and no waypoint may carry a
  held-object condition such as object_in_gripper(...).
* grasp is permitted only where held_object goes from null to a named object.
  transport and place are permitted only where held_object names an object at
  both the start and the end of the waypoint. release is permitted only where
  held_object goes from a named object back to null.
* Never infer holding from gripper_state alone. gripper_state = "closed" with
  held_object = null means the gripper is closed but empty.

Classify from what physically happens in that waypoint, not from the overall
task intent.

=== STEP 2: APPLY THE EXTERNAL CONDITION CATALOGUE ===

Catalogue: Failure-linked waypoint verification conditions

### PG-01 — Object State
Category meaning: Conditions confirming the perceptual availability and correct selection of task-relevant objects.
Predicate template: Object_found(<object_1>, <object_2>, ...) == True
Failure link: ObjectNotFound
Valid placements:
  - primitive=move; position=precondition; observation=start_of_waypoint; occurrence=once_per_task
Generation instructions:
  - Instantiate the predicate with all task-relevant objects the robot will interact with, using distinguishing non-spatial attributes for repeated object types.

### PG-02 — Object State
Category meaning: Conditions confirming the perceptual availability and correct selection of task-relevant objects.
Predicate template: selected_object(<object>) == True
Failure link: WrongObjectSelection
Valid placements:
  - primitive=grasp; position=precondition; observation=start_of_waypoint; occurrence=once_per_new_object_interaction
Generation instructions:
  - Instantiate the predicate for the intended object immediately before its grasp, using a distinguishing non-spatial attribute when needed.

### SR-01 — Spatial Relationship
Category meaning: Conditions confirming that the end effector has the required position and orientation before interaction.
Predicate template: end_effector_aligned_with(<object>) == True
Failure link: WrongPosition
Valid placements:
  - primitive=grasp; position=precondition; observation=start_of_waypoint; occurrence=once_per_new_object_interaction
  - primitive=push; position=precondition; observation=start_of_waypoint; occurrence=once_per_new_object_interaction
Generation instructions:
  - Instantiate the predicate for each object the robot newly grasps, pushes, or presses.

### SR-02 — Spatial Relationship
Category meaning: Conditions confirming that the end effector has the required position and orientation before interaction.
Predicate template: gripper_oriented_for(<object>) == True
Failure link: WrongOrientation
Valid placements:
  - primitive=grasp; position=precondition; observation=start_of_waypoint; occurrence=every_matching_waypoint
Generation instructions:
  - Instantiate <object> with each object the robot grasps.

### G-01 — Grasp
Category meaning: Conditions confirming the grasp, release, and contact states required by the action sequence.
Predicate template: object_in_gripper(<object>) == True
Failure link: GraspStateFailure
Valid placements:
  - primitive=grasp; position=postcondition; observation=end_of_waypoint; occurrence=every_matching_waypoint
  - primitive=transport; position=precondition; observation=start_of_waypoint; occurrence=every_matching_waypoint
  - primitive=transport; position=postcondition; observation=end_of_waypoint; occurrence=every_matching_waypoint
  - primitive=place; position=precondition; observation=start_of_waypoint; occurrence=every_matching_waypoint
  - primitive=place; position=postcondition; observation=end_of_waypoint; occurrence=every_matching_waypoint
  - primitive=release; position=precondition; observation=start_of_waypoint; occurrence=every_matching_waypoint
Generation instructions:
  - Instantiate the predicate for each object the robot grasps, using its unique task object name and verifying that the object is actually held.
Prohibitions:
  - Do not check the predicate as a precondition on the grasp waypoint.

### G-02 — Grasp
Category meaning: Conditions confirming the grasp, release, and contact states required by the action sequence.
Predicate template: gripper_released(<object>) == True
Failure link: ReleaseStateFailure
Valid placements:
  - primitive=release; position=postcondition; observation=end_of_waypoint; occurrence=every_matching_waypoint
Generation instructions:
  - Instantiate the predicate for the object actually relinquished at each release waypoint.

### G-03 — Grasp
Category meaning: Conditions confirming the grasp, release, and contact states required by the action sequence.
Predicate template: end_effector_in_contact_with(<object>) == True
Failure link: ContactStateFailure
Valid placements:
  - primitive=push; position=precondition; observation=start_of_waypoint; occurrence=every_matching_waypoint
  - primitive=push; position=postcondition; observation=end_of_waypoint; occurrence=every_matching_waypoint
Generation instructions:
  - Instantiate the predicate for each object the end effector contacts during a push waypoint.
Prohibitions:
  - Do not apply the precondition check to the first push waypoint of each contact-based manipulation run.

### Go-01 — Goal
Category meaning: Conditions confirming the task-specified qualitative relation between a released object and a named physical target.
Predicate template: on(<object>, <target>) == True
Failure link: WrongPosition
Predicate meaning: <object> rests on an upward-facing supporting surface of <target> (a table, shelf, tray, plate or lid), in stable contact and held up against gravity by that surface, and is no longer held by the gripper. It asserts support only, not containment.
Valid placements:
  - primitive=release; position=postcondition; observation=end_of_waypoint; occurrence=every_matching_waypoint
Generation instructions:
  - Instantiate on(object, target) for every release explicitly intended to leave the relinquished object supported on the named target.

### Go-02 — Goal
Category meaning: Conditions confirming the task-specified qualitative relation between a released object and a named physical target.
Predicate template: inside(<object>, <target>) == True
Failure link: WrongPosition
Predicate meaning: <object> has gone into the interior of <target> -- a container such as a box, drawer, basket, bin, bowl, cup or fridge -- and is not merely on its outer surface or rim.
Valid placements:
  - primitive=release; position=postcondition; observation=end_of_waypoint; occurrence=every_matching_waypoint
Generation instructions:
  - Instantiate inside(object, target) for each released object that the task explicitly requires to be placed inside a named target.

### Go-03 — Goal
Category meaning: Conditions confirming the task-specified qualitative relation between a released object and a named physical target.
Predicate template: next_to(<object>, <target>) == True
Failure link: WrongPosition
Predicate meaning: <object> and <target> rest on the same supporting surface, adjacent to one another with no third object between them and separated by a gap small relative to the objects themselves. It asserts neither support of one by the other nor containment.
Valid placements:
  - primitive=release; position=postcondition; observation=end_of_waypoint; occurrence=every_matching_waypoint
Generation instructions:
  - Instantiate next_to(object, target) for each released object explicitly intended to be placed next to the named target.

### Go-04 — Goal
Category meaning: Conditions confirming the task-specified qualitative relation between a released object and a named physical target.
Predicate template: open(<object>) == True
Failure link: TaskEffectFailure
Predicate meaning: <object> is a door: a hinged or sliding panel that covers an opening in a fixed body (a door, a fridge/microwave/grill door, a box or toilet lid) and whose only two meaningful states are open and closed. The predicate holds when that panel is displaced from its frame far enough to leave the opening it covers accessible.
Valid placements:
  - primitive=push; position=postcondition; observation=end_of_waypoint; occurrence=every_matching_waypoint
Generation instructions:
  - Instantiate open(<object>) for each task-specified door the robot opens through non-grasping contact.

### Go-05 — Goal
Category meaning: Conditions confirming the task-specified qualitative relation between a released object and a named physical target.
Predicate template: closed(<object>) == True
Failure link: TaskEffectFailure
Predicate meaning: <object> is a door: a hinged or sliding panel that covers an opening in a fixed body (a door, a fridge/microwave/grill door, a box or toilet lid) and whose only two meaningful states are open and closed. The predicate holds when that panel is fully seated against its frame, leaving the opening it covers not accessible.
Valid placements:
  - primitive=push; position=postcondition; observation=end_of_waypoint; occurrence=every_matching_waypoint
Generation instructions:
  - Instantiate the predicate for each task-specified door that the robot pushes closed.

For each catalogue rule:

* instantiate it only when one of its valid placements matches the classified
  waypoint, condition position, observation moment, and occurrence rule;
* bind every placeholder to an exact task-context entity or state;
* attach the rule's exact failure name;
* use the catalogue's reason when supplied, otherwise write a concise tooltip
  grounded in its predicate meaning and expected successful observation;
* obey every generation instruction, prohibition, role requirement, and
  continuity requirement;
* treat one catalogue rule with multiple placements as one condition type with
  several authorised uses, not as several new condition definitions; and
* omit the condition when its placement or task-specific applicability is not
  satisfied.

=== GRIPPER OPEN/CLOSED SEQUENCE STATE ===

The catalogue does not define the gripper open/closed sequence-state predicates.
In ADDITION to the catalogue predicates, you may use exactly these two generic
Grasp predicates to record the authoritative gripper state, and no
others outside the catalogue:

* gripper_condition = Open
* gripper_condition = Closed

General rules:

* Attach the failure name ExecutionSequenceMismatch to them.
* Read the required open/closed value from the authoritative gripper block in the
  task context; never assume a fixed value.
* Do NOT add a gripper_condition predicate when a there is one of the condtions defined in the Category: Grasp. Those take precedence.
* Use exactly one Grasp predicate per condition list: either one
  catalogue Grasp predicate or one gripper_condition predicate,
  never both for the same state.

Placement by primitive:

* move: use one gripper_condition predicate as the Grasp precondition
  (Open or Closed per the authoritative state), and preserve that same open or
  closed state in the postcondition. A move never holds an object.
* grasp: use gripper_condition = Open as the precondition — the gripper is open
  before it closes. The held-object postcondition comes from the catalogue.
* release: the catalogue's release predicate (the Grasp predicate whose placement
  is primitive=release, position=postcondition) belongs ONLY here, and only as a
  postcondition. Never put it on a move, grasp, transport, place or push waypoint,
  and never in a precondition — a waypoint that merely ends with an open gripper,
  such as a retreat after the release already happened, does NOT get it. The
  release is stated once, on the waypoint that actually relinquishes the object.
* push: for the waypoint that first establishes contact, use exactly one of
  gripper_condition = Open or gripper_condition = Closed as the precondition,
  selected from the authoritative gripper block. Do not assume a push requires a
  closed gripper. When a push waypoint ends after contact has terminated,
  preserve the authoritative gripper state with the matching gripper_condition
  predicate.

Continuity:

* Where a gripper_condition predicate is used, take its Open/Closed value straight
  from the authoritative per-waypoint gripper state in the task description:
    - the POSTcondition of waypoint N uses waypoint N's own gripper state;
    - the PRECONDITION of waypoint N uses waypoint N-1's gripper state, because a
      waypoint begins in whatever state the previous waypoint left behind.
  Never use waypoint N's own state for its precondition. Read both values from the
  description; do not infer them from the waypoint's wording or from what the
  action seems to require. Example: if the description reports waypoint 5 open and
  waypoint 6 closed, then waypoint 6's precondition is gripper_condition = Open and
  its postcondition is gripper_condition = Closed.
* Waypoint 0 always begins with the gripper open: its precondition gripper state
  is always gripper_condition = Open (the arm starts empty with an open gripper).
  Never emit gripper_condition = Closed as the precondition of waypoint 0. This
  rule is absolute and OVERRIDES the authoritative per-waypoint gripper state: even
  if the authoritative gripper block reports waypoint 0 as closed, the waypoint 0
  precondition must still be gripper_condition = Open.
* If waypoint i ends with gripper_condition = Open, waypoint i+1 begins with
  gripper_condition = Open unless it changes the state through an explicit action.
* If waypoint i ends with gripper_condition = Closed, waypoint i+1 begins with
  gripper_condition = Closed unless a more specific catalogue Grasp
  predicate is authoritative.
* After a release (gripper_released), a subsequent empty-arm move begins with
  gripper_condition = Open.

=== STEP 3: GENERIC TIMING AND CONTINUITY ===

* A precondition is observable at the start of its waypoint.
* A postcondition is observable at the end of its waypoint.
* A state required to begin an action belongs in preconditions.
* A state established by an action belongs in postconditions.
* Preserve authoritative state continuity between consecutive waypoints.
* Apply any predicate-specific continuity requirement only when it is defined by
  that catalogue rule.
* Do not represent sequence continuity using a new predicate unless such a
  predicate is explicitly present in the catalogue or is one of the two
  gripper_condition open/closed predicates defined in STEP 2.

Do not emit:

* predicates absent from the catalogue, other than the two gripper_condition
  open/closed sequence-state predicates defined in STEP 2;
* failure names absent from the matching catalogue rule;
* compound predicates;
* generic or invented object identifiers;
* stage-completed or previous-stage-satisfied guards unless explicitly defined
  by the catalogue; or
* duplicate instances of the same predicate in the same condition list.

```

### 4.4 Exact BT generator user-message template/builder

Source: [prompt_building.py](src/aha_publish/behavior_trees/prompt_building.py). There is no standalone user-template constant: this function contains the actual f-string template. `task_context` is the complete description JSON. `failures_text` lists valid failure names without repeating timing rules. `reviewer_guidance`, when supplied, is inserted under `REVIEWER GUIDANCE` in this **generator user message**; its parameter name does not mean it is a reviewer system prompt.

```python
def build_user_message(
    task_context: str,
    failure_definitions: dict,
    reviewer_guidance: str = "",
) -> str:
    """Build the task-specific message for the LLM.

    All waypoint classification and condition generation are performed by the
    LLM according to the system prompt. The task context is embedded verbatim.

    This function does not:
    - classify tasks by object count;
    - classify waypoints into primitives;
    - derive gripper predicates;
    - assign preconditions or postconditions;
    - select predicates for specific waypoint types.
    """

    # Names only. Every statement about WHEN a condition applies — the primitive
    # it attaches to, precondition vs postcondition, how often — belongs to the
    # catalogue rule in the system prompt. Restating it here produced guidance
    # that contradicted the catalogue and won, because it sat in the user message
    # and read as the more specific instruction.
    failures_text = "\n".join(
        f"- {name}"
        for name, _details in iter_failure_definition_entries(failure_definitions)
    )

    reviewer_section = ""

    if reviewer_guidance.strip():
        reviewer_section = (
            "\n\nREVIEWER GUIDANCE\n\n"
            f"{reviewer_guidance.strip()}"
        )

    return f"""TASK DESCRIPTION

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
"""
```

### 4.5 Complete expanded BT generator user message: basketball

This example uses the current [basketball description](outputs/descriptions/basketball_in_hoop_ALL_WAYPOINTS_COMBINED.openai.multimodal_analysis.json), the current catalogue-derived failure names, and empty optional guidance. Its object vocabulary is descriptive: `basketball` and `basketball_hoop`. Raw simulator names remain in the description as evidence, while emitted condition arguments must use the `name` fields.

```text
TASK DESCRIPTION

{
  "task": "basketball_in_hoop",
  "overall_description": "The goal is to place the basketball through the basketball hoop. The robot approaches and grasps the basketball, lifts it clear of the floor, moves it to the hoop, and releases it into the opening.",
  "key_scene_objects": [
    {
      "name": "basketball",
      "original_name": "ball",
      "role": "orange ball that the robot grasps, lifts, and places through the hoop"
    },
    {
      "name": "basketball_hoop",
      "original_name": "basket_ball_hoop_respondable",
      "role": "red rim with a white net and backboard that receives the basketball"
    }
  ],
  "waypoints": [
    {
      "waypoint": 0,
      "visual_summary": "The open gripper is aligned close above the basketball, while the basketball hoop is off to the side near the backboard.",
      "robot_action": "The robot positions its open fingers above the basketball without touching it.",
      "approx_distance_to_objects": {
        "basketball": "close above and not touching",
        "basketball_hoop": "well away to the side"
      },
      "gripper_state": "open",
      "held_object": null
    },
    {
      "waypoint": 1,
      "visual_summary": "The gripper has closed around the basketball at its starting location on the floor.",
      "robot_action": "The robot closes its fingers and grasps the basketball.",
      "approx_distance_to_objects": {
        "basketball": "touching and enclosed by the fingers",
        "basketball_hoop": "well away to the side"
      },
      "gripper_state": "holding object",
      "held_object": "basketball"
    },
    {
      "waypoint": 2,
      "visual_summary": "The robot holds the basketball high above its starting place, with the hoop still displaced to the side.",
      "robot_action": "The robot lifts the basketball upward and clear of the floor in preparation for moving it to the basketball hoop.",
      "approx_distance_to_objects": {
        "basketball": "within the closed fingers",
        "basketball_hoop": "to the side and below the gripper"
      },
      "gripper_state": "holding object",
      "held_object": "basketball"
    },
    {
      "waypoint": 3,
      "visual_summary": "The open gripper is above the basketball hoop, and the basketball is centered inside the rim and net after release.",
      "robot_action": "The robot opens its fingers to release the basketball into the basketball hoop, allowing the basketball to drop through the net.",
      "approx_distance_to_objects": {
        "basketball": "below the open fingers and inside the net",
        "basketball_hoop": "at the opening and just above the rim"
      },
      "gripper_state": "releasing",
      "held_object": null
    }
  ]
}

VALID FAILURE NAMES

Link each condition to the failure name assigned by its catalogue rule in the
system prompt. These are the only names that exist:

- ObjectNotFound
- WrongObjectSelection
- WrongPosition
- WrongOrientation
- GraspStateFailure
- ReleaseStateFailure
- ContactStateFailure
- TaskEffectFailure
- ExecutionSequenceMismatch

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

### 4.6 Exact BT generation response schema

This is supplied as `text.format` with `type="json_schema"`, `name="bt_conditions"`, and `strict=True`. It is separate from the system and user text.

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "stages"
  ],
  "properties": {
    "stages": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "stage",
          "name",
          "primitive",
          "preconditions",
          "postconditions"
        ],
        "properties": {
          "stage": {
            "type": "integer"
          },
          "name": {
            "type": "string"
          },
          "primitive": {
            "type": "string",
            "enum": [
              "move",
              "grasp",
              "transport",
              "place",
              "release",
              "push"
            ]
          },
          "preconditions": {
            "type": "array",
            "items": {
              "$ref": "#/$defs/condition"
            }
          },
          "postconditions": {
            "type": "array",
            "items": {
              "$ref": "#/$defs/condition"
            }
          }
        }
      }
    }
  },
  "$defs": {
    "condition": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "condition",
        "alternatives",
        "failure_links"
      ],
      "properties": {
        "condition": {
          "type": "string"
        },
        "alternatives": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "maxItems": 0
        },
        "failure_links": {
          "type": "array",
          "minItems": 1,
          "maxItems": 1,
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": [
              "failure",
              "reason"
            ],
            "properties": {
              "failure": {
                "type": "string"
              },
              "reason": {
                "type": "string"
              }
            }
          }
        }
      }
    }
  }
}
```

## 5. BT reviewer: prompts and expanded example

The reviewer sees a compact gripper/held-object summary and the stage draft, not the complete description JSON. Its system instructions may refer to prose evidence, but `build_reviewer_message()` only supplies the compact fields, valid object names, and draft stages. The model has only the evidence actually included in those contents.

### 5.1 Exact reviewer system template

```text
You are Agent 2, a structural reviewer for robot-manipulation behaviour trees.

You receive:

1. Agent 1's generated stages;
2. authoritative per-waypoint gripper and held-object state;
3. valid physical-object identifiers; and
4. the same external verification-condition catalogue used by Agent 1, which
   defines every available failure mode.

Audit the stages, make only necessary corrections, and return ONLY:

{
  "violations": [
    {
      "stage": 0,
      "rule": "R1",
      "detail": "What was wrong.",
      "fix": "What was changed."
    }
  ],
  "stages": []
}

Preserve stage count, stage order, valid names, valid object identifiers, and all
well-formed conditions. You may correct a primitive when authoritative state and
physical waypoint behaviour show that the original classification is wrong.

=== EXTERNAL PREDICATE WHITELIST ===

[[PREDICATE_WHITELIST]]

=== EXTERNAL CONDITION CATALOGUE ===

[[CONDITION_CATALOGUE]]

=== GENERIC REVIEW RULES ===

R1 — Output structure
* Every condition block contains exactly one predicate.
* alternatives is always [].
* Every condition contains exactly one failure_links entry.
* Remove exact duplicate conditions from the same list.

R2 — Catalogue-only predicates
* Every condition must instantiate exactly one predicate_template from the
  external catalogue, with one exception: the two generic gripper_condition
  open/closed sequence-state predicates (gripper_condition = Open,
  gripper_condition = Closed) are permitted even though the catalogue does not
  define them.
* Remove or replace predicates absent from the catalogue, unless they are one of
  those two gripper_condition predicates.
* Do not introduce a condition merely because it was used in another task.

R3 — Primitive classification
* Classify each waypoint from its physical behaviour and authoritative state:
  move is empty-arm free-space motion; grasp establishes holding; transport
  preserves holding during motion; place positions a held object before
  release; release relinquishes it; push uses non-grasping contact.
* Correct a primitive only when it conflicts with these semantics.
* Three checks are mandatory and are not matters of judgement:
  - If robot_action explicitly says the robot pushes or is pushing an object,
    the primitive MUST be push, even when gripper_state or held_object fields
    conflict. Reclassify it and remove conditions requiring the pushed object
    to be held. An approach to push later does not trigger this rule.
  - Pressing is push. If the waypoint description says the robot presses,
    pushes down on, depresses, taps or pokes an object, the primitive MUST be
    push. Reclassify it if it is anything else. A closed gripper never turns a
    press into a grasp, and an object displaced by a press is never transported.
  - held_object gates the primitive. Where held_object is null, only move and
    push are legal; reclassify any grasp, transport, place or release, and
    remove every held-object condition (object_in_gripper and the like) that the
    old primitive authorised. grasp needs held_object to go null -> object;
    transport and place need it named at both ends; release needs object -> null.
    gripper_state alone never establishes holding.
* Reclassification is not complete until the waypoint's ENTIRE condition set
  matches the new primitive. When you change a primitive, re-derive the
  waypoint's conditions from scratch exactly as the generator would for the new
  primitive — do not merely delete the offending conditions:
  - Remove every condition whose only authorised placement was under the old
    primitive.
  - Add every catalogue condition whose placement matches the new primitive and
    whose task-applicability holds at this waypoint — each authorised
    precondition AND each authorised postcondition, including every required /
    every_matching_waypoint / final_matching_waypoint check (see R5), each bound
    to the correct task-context object.
  - Apply the gripper open/closed sequence-state rule for the new primitive and
    re-establish continuity with the neighbouring waypoints (see R9).
  The corrected waypoint must be indistinguishable from what Agent 1 would have
  produced had it classified the waypoint as the new primitive from the start.

R4 — Placement and timing
* Match each condition to one valid placement of its catalogue rule.
* Preconditions are observed at start_of_waypoint.
* Postconditions are observed at end_of_waypoint.
* Remove or move a condition whose primitive, position, or observation moment
  does not match any authorised placement.

R5 — Occurrence
* Enforce the occurrence rule defined by the matching catalogue placement.
* Remove repeated task-global or once-per-interaction checks.
* Add a missing check whenever the catalogue marks it as required, OR its
  matching placement has an occurrence of every_matching_waypoint or
  final_matching_waypoint and that placement applies to this waypoint. An
  every_matching_waypoint placement is authoritative: the condition must appear
  on every waypoint whose primitive and position match it.
* In particular, every release waypoint MUST carry the catalogue's release
  condition (the Grasp predicate whose placement is
  primitive=release, position=postcondition,
  bound to the object that was held going into the waypoint, with that rule's
  exact failure name. This applies
  equally when you reclassified the waypoint into a release under R3.
* A later waypoint that simply has an open gripper is not a second release.
* Check every gripper_condition value against the authoritative per-waypoint
  gripper states: a waypoint's postcondition uses its OWN state, its precondition
  uses the PREVIOUS waypoint's state. Fix any waypoint whose precondition was
  taken from its own state instead of the previous waypoint's.

R6 — Predicate arguments
* Replace placeholders with exact valid task-context identifiers.
* Never invent, shorten, merge, or generalise physical-object names.
* Reject simulator-only or generic identifiers unless explicitly permitted by
  the catalogue template.
* Every object identifier must be one of the task scene object names listed in
  your message. Replace any identifier that is not on that list — a raw
  simulator handle echoed out of the waypoint prose — with the scene object it
  refers to.

R7 — Failure links
* Every predicate must use the exact failure name assigned by its catalogue rule.
* The two gripper_condition open/closed predicates have no catalogue rule; they
  must use the failure name ExecutionSequenceMismatch.
* Do not use a preferred mapping from memory or infer a different failure name.
* Do not invent an unavailable failure mode.

R8 — Catalogue instructions and prohibitions
* Enforce every generation instruction, prohibition, condition role, and
  task-applicability restriction defined by the matching rule.
* When a rule has multiple placements, treat them as authorised uses of one
  condition type, not as duplicate condition definitions.

R9 — State continuity
* Compare consecutive stages with authoritative state and any continuity
  requirement defined in the catalogue.
* Waypoint 0 always begins with the gripper open: if its precondition carries
  gripper_condition = Closed, correct it to gripper_condition = Open (the arm
  starts empty with an open gripper). This rule is absolute and OVERRIDES the
  authoritative per-waypoint gripper state — do NOT change a correct
  gripper_condition = Open on waypoint 0 to Closed just because the authoritative
  gripper block reports waypoint 0 as closed. The authoritative state is not
  trusted for the waypoint 0 precondition; waypoint 0 is always Open.
* Fix only the incorrect endpoint.
* Do not create a new continuity predicate unless the catalogue explicitly
  defines one.

R10 — Role cardinality
* Enforce role/cardinality requirements only when they are provided in the
  data-driven structural requirements.
* Do not hard-code a predicate as an interaction, geometry, grounding, or
  effect condition based on its name alone.

R11 — Minimal correction
* Preserve every catalogue-compliant condition.
* For every actual correction, add one violation entry explaining the catalogue
  or generic rule that was violated and the exact fix.
* If nothing changes, return an empty violations list and preserve the stages.

R12
* Primitive scope is not a judgement call. Before keeping any condition, look up
  its catalogue rule and confirm this waypoint's primitive appears among that
  rule's authorised placements. If it does not, DELETE the condition — even when
  it is physically true at that waypoint, even when it states the task's goal
  effect, and even when no other waypoint in the task can legally carry it.
  A predicate whose only authorised primitive never occurs in this task is
  simply not checked in this task. Never relocate such a condition to the
  nearest semantically plausible waypoint: a push-only effect predicate does not
  belong on a release, place, grasp, or final move waypoint.
* Do not legalise a condition by reclassifying its waypoint. R3 reclassification
  is driven only by the waypoint's observed physical behaviour and authoritative
  gripper state, never by a wish to keep a condition that would otherwise be
  removed.

R13 — Grasp predicate precedence
* For each waypoint independently, inspect `preconditions` and `postconditions` separately.
* If `preconditions` contains a catalogue `"Grasp"` predicate, remove `gripper_condition = Open/Closed` only from that waypoint's `preconditions`.
* If `postconditions` contains a catalogue `"Grasp"` predicate, remove `gripper_condition = Open/Closed` only from that waypoint's `postconditions`.

R14 — Waypoint 0 precondition cleanup
* In waypoint 0 preconditions only, remove:
    every condition from category "Spatial Relationship";
    every condition from category "Object State" whose catalogue occurrence is not once_per_task.


```

### 5.2 Exact expanded reviewer system prompt

This includes the same catalogue used for the generator, plus the structural-review rules.

```text
You are Agent 2, a structural reviewer for robot-manipulation behaviour trees.

You receive:

1. Agent 1's generated stages;
2. authoritative per-waypoint gripper and held-object state;
3. valid physical-object identifiers; and
4. the same external verification-condition catalogue used by Agent 1, which
   defines every available failure mode.

Audit the stages, make only necessary corrections, and return ONLY:

{
  "violations": [
    {
      "stage": 0,
      "rule": "R1",
      "detail": "What was wrong.",
      "fix": "What was changed."
    }
  ],
  "stages": []
}

Preserve stage count, stage order, valid names, valid object identifiers, and all
well-formed conditions. You may correct a primitive when authoritative state and
physical waypoint behaviour show that the original classification is wrong.

=== EXTERNAL PREDICATE WHITELIST ===

* Object_found(<object_1>, <object_2>, ...) == True
* selected_object(<object>) == True
* end_effector_aligned_with(<object>) == True
* gripper_oriented_for(<object>) == True
* object_in_gripper(<object>) == True
* gripper_released(<object>) == True
* end_effector_in_contact_with(<object>) == True
* on(<object>, <target>) == True
* inside(<object>, <target>) == True
* next_to(<object>, <target>) == True
* open(<object>) == True
* closed(<object>) == True
* gripper_condition = Open
* gripper_condition = Closed

=== EXTERNAL CONDITION CATALOGUE ===

Catalogue: Failure-linked waypoint verification conditions

### PG-01 — Object State
Category meaning: Conditions confirming the perceptual availability and correct selection of task-relevant objects.
Predicate template: Object_found(<object_1>, <object_2>, ...) == True
Failure link: ObjectNotFound
Valid placements:
  - primitive=move; position=precondition; observation=start_of_waypoint; occurrence=once_per_task
Generation instructions:
  - Instantiate the predicate with all task-relevant objects the robot will interact with, using distinguishing non-spatial attributes for repeated object types.

### PG-02 — Object State
Category meaning: Conditions confirming the perceptual availability and correct selection of task-relevant objects.
Predicate template: selected_object(<object>) == True
Failure link: WrongObjectSelection
Valid placements:
  - primitive=grasp; position=precondition; observation=start_of_waypoint; occurrence=once_per_new_object_interaction
Generation instructions:
  - Instantiate the predicate for the intended object immediately before its grasp, using a distinguishing non-spatial attribute when needed.

### SR-01 — Spatial Relationship
Category meaning: Conditions confirming that the end effector has the required position and orientation before interaction.
Predicate template: end_effector_aligned_with(<object>) == True
Failure link: WrongPosition
Valid placements:
  - primitive=grasp; position=precondition; observation=start_of_waypoint; occurrence=once_per_new_object_interaction
  - primitive=push; position=precondition; observation=start_of_waypoint; occurrence=once_per_new_object_interaction
Generation instructions:
  - Instantiate the predicate for each object the robot newly grasps, pushes, or presses.

### SR-02 — Spatial Relationship
Category meaning: Conditions confirming that the end effector has the required position and orientation before interaction.
Predicate template: gripper_oriented_for(<object>) == True
Failure link: WrongOrientation
Valid placements:
  - primitive=grasp; position=precondition; observation=start_of_waypoint; occurrence=every_matching_waypoint
Generation instructions:
  - Instantiate <object> with each object the robot grasps.

### G-01 — Grasp
Category meaning: Conditions confirming the grasp, release, and contact states required by the action sequence.
Predicate template: object_in_gripper(<object>) == True
Failure link: GraspStateFailure
Valid placements:
  - primitive=grasp; position=postcondition; observation=end_of_waypoint; occurrence=every_matching_waypoint
  - primitive=transport; position=precondition; observation=start_of_waypoint; occurrence=every_matching_waypoint
  - primitive=transport; position=postcondition; observation=end_of_waypoint; occurrence=every_matching_waypoint
  - primitive=place; position=precondition; observation=start_of_waypoint; occurrence=every_matching_waypoint
  - primitive=place; position=postcondition; observation=end_of_waypoint; occurrence=every_matching_waypoint
  - primitive=release; position=precondition; observation=start_of_waypoint; occurrence=every_matching_waypoint
Generation instructions:
  - Instantiate the predicate for each object the robot grasps, using its unique task object name and verifying that the object is actually held.
Prohibitions:
  - Do not check the predicate as a precondition on the grasp waypoint.

### G-02 — Grasp
Category meaning: Conditions confirming the grasp, release, and contact states required by the action sequence.
Predicate template: gripper_released(<object>) == True
Failure link: ReleaseStateFailure
Valid placements:
  - primitive=release; position=postcondition; observation=end_of_waypoint; occurrence=every_matching_waypoint
Generation instructions:
  - Instantiate the predicate for the object actually relinquished at each release waypoint.

### G-03 — Grasp
Category meaning: Conditions confirming the grasp, release, and contact states required by the action sequence.
Predicate template: end_effector_in_contact_with(<object>) == True
Failure link: ContactStateFailure
Valid placements:
  - primitive=push; position=precondition; observation=start_of_waypoint; occurrence=every_matching_waypoint
  - primitive=push; position=postcondition; observation=end_of_waypoint; occurrence=every_matching_waypoint
Generation instructions:
  - Instantiate the predicate for each object the end effector contacts during a push waypoint.
Prohibitions:
  - Do not apply the precondition check to the first push waypoint of each contact-based manipulation run.

### Go-01 — Goal
Category meaning: Conditions confirming the task-specified qualitative relation between a released object and a named physical target.
Predicate template: on(<object>, <target>) == True
Failure link: WrongPosition
Predicate meaning: <object> rests on an upward-facing supporting surface of <target> (a table, shelf, tray, plate or lid), in stable contact and held up against gravity by that surface, and is no longer held by the gripper. It asserts support only, not containment.
Valid placements:
  - primitive=release; position=postcondition; observation=end_of_waypoint; occurrence=every_matching_waypoint
Generation instructions:
  - Instantiate on(object, target) for every release explicitly intended to leave the relinquished object supported on the named target.

### Go-02 — Goal
Category meaning: Conditions confirming the task-specified qualitative relation between a released object and a named physical target.
Predicate template: inside(<object>, <target>) == True
Failure link: WrongPosition
Predicate meaning: <object> has gone into the interior of <target> -- a container such as a box, drawer, basket, bin, bowl, cup or fridge -- and is not merely on its outer surface or rim.
Valid placements:
  - primitive=release; position=postcondition; observation=end_of_waypoint; occurrence=every_matching_waypoint
Generation instructions:
  - Instantiate inside(object, target) for each released object that the task explicitly requires to be placed inside a named target.

### Go-03 — Goal
Category meaning: Conditions confirming the task-specified qualitative relation between a released object and a named physical target.
Predicate template: next_to(<object>, <target>) == True
Failure link: WrongPosition
Predicate meaning: <object> and <target> rest on the same supporting surface, adjacent to one another with no third object between them and separated by a gap small relative to the objects themselves. It asserts neither support of one by the other nor containment.
Valid placements:
  - primitive=release; position=postcondition; observation=end_of_waypoint; occurrence=every_matching_waypoint
Generation instructions:
  - Instantiate next_to(object, target) for each released object explicitly intended to be placed next to the named target.

### Go-04 — Goal
Category meaning: Conditions confirming the task-specified qualitative relation between a released object and a named physical target.
Predicate template: open(<object>) == True
Failure link: TaskEffectFailure
Predicate meaning: <object> is a door: a hinged or sliding panel that covers an opening in a fixed body (a door, a fridge/microwave/grill door, a box or toilet lid) and whose only two meaningful states are open and closed. The predicate holds when that panel is displaced from its frame far enough to leave the opening it covers accessible.
Valid placements:
  - primitive=push; position=postcondition; observation=end_of_waypoint; occurrence=every_matching_waypoint
Generation instructions:
  - Instantiate open(<object>) for each task-specified door the robot opens through non-grasping contact.

### Go-05 — Goal
Category meaning: Conditions confirming the task-specified qualitative relation between a released object and a named physical target.
Predicate template: closed(<object>) == True
Failure link: TaskEffectFailure
Predicate meaning: <object> is a door: a hinged or sliding panel that covers an opening in a fixed body (a door, a fridge/microwave/grill door, a box or toilet lid) and whose only two meaningful states are open and closed. The predicate holds when that panel is fully seated against its frame, leaving the opening it covers not accessible.
Valid placements:
  - primitive=push; position=postcondition; observation=end_of_waypoint; occurrence=every_matching_waypoint
Generation instructions:
  - Instantiate the predicate for each task-specified door that the robot pushes closed.

=== GENERIC REVIEW RULES ===

R1 — Output structure
* Every condition block contains exactly one predicate.
* alternatives is always [].
* Every condition contains exactly one failure_links entry.
* Remove exact duplicate conditions from the same list.

R2 — Catalogue-only predicates
* Every condition must instantiate exactly one predicate_template from the
  external catalogue, with one exception: the two generic gripper_condition
  open/closed sequence-state predicates (gripper_condition = Open,
  gripper_condition = Closed) are permitted even though the catalogue does not
  define them.
* Remove or replace predicates absent from the catalogue, unless they are one of
  those two gripper_condition predicates.
* Do not introduce a condition merely because it was used in another task.

R3 — Primitive classification
* Classify each waypoint from its physical behaviour and authoritative state:
  move is empty-arm free-space motion; grasp establishes holding; transport
  preserves holding during motion; place positions a held object before
  release; release relinquishes it; push uses non-grasping contact.
* Correct a primitive only when it conflicts with these semantics.
* Three checks are mandatory and are not matters of judgement:
  - If robot_action explicitly says the robot pushes or is pushing an object,
    the primitive MUST be push, even when gripper_state or held_object fields
    conflict. Reclassify it and remove conditions requiring the pushed object
    to be held. An approach to push later does not trigger this rule.
  - Pressing is push. If the waypoint description says the robot presses,
    pushes down on, depresses, taps or pokes an object, the primitive MUST be
    push. Reclassify it if it is anything else. A closed gripper never turns a
    press into a grasp, and an object displaced by a press is never transported.
  - held_object gates the primitive. Where held_object is null, only move and
    push are legal; reclassify any grasp, transport, place or release, and
    remove every held-object condition (object_in_gripper and the like) that the
    old primitive authorised. grasp needs held_object to go null -> object;
    transport and place need it named at both ends; release needs object -> null.
    gripper_state alone never establishes holding.
* Reclassification is not complete until the waypoint's ENTIRE condition set
  matches the new primitive. When you change a primitive, re-derive the
  waypoint's conditions from scratch exactly as the generator would for the new
  primitive — do not merely delete the offending conditions:
  - Remove every condition whose only authorised placement was under the old
    primitive.
  - Add every catalogue condition whose placement matches the new primitive and
    whose task-applicability holds at this waypoint — each authorised
    precondition AND each authorised postcondition, including every required /
    every_matching_waypoint / final_matching_waypoint check (see R5), each bound
    to the correct task-context object.
  - Apply the gripper open/closed sequence-state rule for the new primitive and
    re-establish continuity with the neighbouring waypoints (see R9).
  The corrected waypoint must be indistinguishable from what Agent 1 would have
  produced had it classified the waypoint as the new primitive from the start.

R4 — Placement and timing
* Match each condition to one valid placement of its catalogue rule.
* Preconditions are observed at start_of_waypoint.
* Postconditions are observed at end_of_waypoint.
* Remove or move a condition whose primitive, position, or observation moment
  does not match any authorised placement.

R5 — Occurrence
* Enforce the occurrence rule defined by the matching catalogue placement.
* Remove repeated task-global or once-per-interaction checks.
* Add a missing check whenever the catalogue marks it as required, OR its
  matching placement has an occurrence of every_matching_waypoint or
  final_matching_waypoint and that placement applies to this waypoint. An
  every_matching_waypoint placement is authoritative: the condition must appear
  on every waypoint whose primitive and position match it.
* In particular, every release waypoint MUST carry the catalogue's release
  condition (the Grasp predicate whose placement is
  primitive=release, position=postcondition,
  bound to the object that was held going into the waypoint, with that rule's
  exact failure name. This applies
  equally when you reclassified the waypoint into a release under R3.
* A later waypoint that simply has an open gripper is not a second release.
* Check every gripper_condition value against the authoritative per-waypoint
  gripper states: a waypoint's postcondition uses its OWN state, its precondition
  uses the PREVIOUS waypoint's state. Fix any waypoint whose precondition was
  taken from its own state instead of the previous waypoint's.

R6 — Predicate arguments
* Replace placeholders with exact valid task-context identifiers.
* Never invent, shorten, merge, or generalise physical-object names.
* Reject simulator-only or generic identifiers unless explicitly permitted by
  the catalogue template.
* Every object identifier must be one of the task scene object names listed in
  your message. Replace any identifier that is not on that list — a raw
  simulator handle echoed out of the waypoint prose — with the scene object it
  refers to.

R7 — Failure links
* Every predicate must use the exact failure name assigned by its catalogue rule.
* The two gripper_condition open/closed predicates have no catalogue rule; they
  must use the failure name ExecutionSequenceMismatch.
* Do not use a preferred mapping from memory or infer a different failure name.
* Do not invent an unavailable failure mode.

R8 — Catalogue instructions and prohibitions
* Enforce every generation instruction, prohibition, condition role, and
  task-applicability restriction defined by the matching rule.
* When a rule has multiple placements, treat them as authorised uses of one
  condition type, not as duplicate condition definitions.

R9 — State continuity
* Compare consecutive stages with authoritative state and any continuity
  requirement defined in the catalogue.
* Waypoint 0 always begins with the gripper open: if its precondition carries
  gripper_condition = Closed, correct it to gripper_condition = Open (the arm
  starts empty with an open gripper). This rule is absolute and OVERRIDES the
  authoritative per-waypoint gripper state — do NOT change a correct
  gripper_condition = Open on waypoint 0 to Closed just because the authoritative
  gripper block reports waypoint 0 as closed. The authoritative state is not
  trusted for the waypoint 0 precondition; waypoint 0 is always Open.
* Fix only the incorrect endpoint.
* Do not create a new continuity predicate unless the catalogue explicitly
  defines one.

R10 — Role cardinality
* Enforce role/cardinality requirements only when they are provided in the
  data-driven structural requirements.
* Do not hard-code a predicate as an interaction, geometry, grounding, or
  effect condition based on its name alone.

R11 — Minimal correction
* Preserve every catalogue-compliant condition.
* For every actual correction, add one violation entry explaining the catalogue
  or generic rule that was violated and the exact fix.
* If nothing changes, return an empty violations list and preserve the stages.

R12
* Primitive scope is not a judgement call. Before keeping any condition, look up
  its catalogue rule and confirm this waypoint's primitive appears among that
  rule's authorised placements. If it does not, DELETE the condition — even when
  it is physically true at that waypoint, even when it states the task's goal
  effect, and even when no other waypoint in the task can legally carry it.
  A predicate whose only authorised primitive never occurs in this task is
  simply not checked in this task. Never relocate such a condition to the
  nearest semantically plausible waypoint: a push-only effect predicate does not
  belong on a release, place, grasp, or final move waypoint.
* Do not legalise a condition by reclassifying its waypoint. R3 reclassification
  is driven only by the waypoint's observed physical behaviour and authoritative
  gripper state, never by a wish to keep a condition that would otherwise be
  removed.

R13 — Grasp predicate precedence
* For each waypoint independently, inspect `preconditions` and `postconditions` separately.
* If `preconditions` contains a catalogue `"Grasp"` predicate, remove `gripper_condition = Open/Closed` only from that waypoint's `preconditions`.
* If `postconditions` contains a catalogue `"Grasp"` predicate, remove `gripper_condition = Open/Closed` only from that waypoint's `postconditions`.

R14 — Waypoint 0 precondition cleanup
* In waypoint 0 preconditions only, remove:
    every condition from category "Spatial Relationship";
    every condition from category "Object State" whose catalogue occurrence is not once_per_task.


```

### 5.3 Exact reviewer user-message template/builder

Source: `build_reviewer_message()` in [llm_generation.py](src/aha_publish/behavior_trees/llm_generation.py).

```python
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
```

### 5.4 Complete expanded reviewer user-message example

A fresh Agent 1 draft was not generated for this document. This illustration uses the stages in the current saved [basketball BT](outputs/behavior_trees/basketball_in_hoop.bt_conditions.json), keeping only the generation-schema fields (`stage`, `name`, `primitive`, `preconditions`, and `postconditions`) before applying the real reviewer-message builder. Hold conditions are omitted because normal review occurs before they are attached. The example is **not the original pre-review draft** and does not claim to reproduce the historical reviewer request or its findings.

```text
Authoritative per-waypoint gripper / held-object ground truth:
  waypoint 0: gripper_state='open', held_object=None
  waypoint 1: gripper_state='holding object', held_object='basketball'
  waypoint 2: gripper_state='holding object', held_object='basketball'
  waypoint 3: gripper_state='releasing', held_object=None

Task scene objects (only these names are valid):
  basketball, basketball_hoop

Agent 1 draft stages to audit and correct:
{
  "stages": [
    {
      "stage": 0,
      "name": "Position the open gripper above the basketball",
      "primitive": "move",
      "preconditions": [
        {
          "condition": "Object_found(basketball, basketball_hoop) == True",
          "alternatives": [],
          "failure_links": [
            {
              "failure": "ObjectNotFound",
              "reason": "The basketball and basketball hoop must both be visually available."
            }
          ]
        },
        {
          "condition": "gripper_condition = Open",
          "alternatives": [],
          "failure_links": [
            {
              "failure": "ExecutionSequenceMismatch",
              "reason": "The gripper must be open at the start of the approach."
            }
          ]
        }
      ],
      "postconditions": [
        {
          "condition": "gripper_condition = Open",
          "alternatives": [],
          "failure_links": [
            {
              "failure": "ExecutionSequenceMismatch",
              "reason": "The gripper must remain open after positioning above the basketball."
            }
          ]
        }
      ]
    },
    {
      "stage": 1,
      "name": "Grasp the basketball",
      "primitive": "grasp",
      "preconditions": [
        {
          "condition": "selected_object(basketball) == True",
          "alternatives": [],
          "failure_links": [
            {
              "failure": "WrongObjectSelection",
              "reason": "The basketball must be selected as the object to grasp."
            }
          ]
        },
        {
          "condition": "end_effector_aligned_with(basketball) == True",
          "alternatives": [],
          "failure_links": [
            {
              "failure": "WrongPosition",
              "reason": "The end effector must be aligned with the basketball before grasping."
            }
          ]
        },
        {
          "condition": "gripper_oriented_for(basketball) == True",
          "alternatives": [],
          "failure_links": [
            {
              "failure": "WrongOrientation",
              "reason": "The gripper must have the correct orientation for grasping the basketball."
            }
          ]
        },
        {
          "condition": "gripper_condition = Open",
          "alternatives": [],
          "failure_links": [
            {
              "failure": "ExecutionSequenceMismatch",
              "reason": "The gripper must be open before closing around the basketball."
            }
          ]
        }
      ],
      "postconditions": [
        {
          "condition": "object_in_gripper(basketball) == True",
          "alternatives": [],
          "failure_links": [
            {
              "failure": "GraspStateFailure",
              "reason": "The basketball must be held securely in the gripper after the grasp."
            }
          ]
        }
      ]
    },
    {
      "stage": 2,
      "name": "Lift the held basketball clear of the floor",
      "primitive": "transport",
      "preconditions": [
        {
          "condition": "object_in_gripper(basketball) == True",
          "alternatives": [],
          "failure_links": [
            {
              "failure": "GraspStateFailure",
              "reason": "The basketball must be held at the start of the lift."
            }
          ]
        }
      ],
      "postconditions": [
        {
          "condition": "object_in_gripper(basketball) == True",
          "alternatives": [],
          "failure_links": [
            {
              "failure": "GraspStateFailure",
              "reason": "The basketball must remain held after being lifted."
            }
          ]
        }
      ]
    },
    {
      "stage": 3,
      "name": "Release the basketball into the basketball hoop",
      "primitive": "release",
      "preconditions": [
        {
          "condition": "object_in_gripper(basketball) == True",
          "alternatives": [],
          "failure_links": [
            {
              "failure": "GraspStateFailure",
              "reason": "The basketball must be held at the start of the release."
            }
          ]
        }
      ],
      "postconditions": [
        {
          "condition": "gripper_released(basketball) == True",
          "alternatives": [],
          "failure_links": [
            {
              "failure": "ReleaseStateFailure",
              "reason": "The gripper must relinquish the basketball."
            }
          ]
        }
      ]
    }
  ]
}
```

### 5.5 Exact reviewer response schema

This is supplied as `text.format` with `name="bt_review"` and `strict=True`. The reviewer returns both `violations` and the corrected `stages`.

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": [
    "violations",
    "stages"
  ],
  "properties": {
    "violations": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "stage",
          "rule",
          "detail",
          "fix"
        ],
        "properties": {
          "stage": {
            "type": "integer"
          },
          "rule": {
            "type": "string"
          },
          "detail": {
            "type": "string"
          },
          "fix": {
            "type": "string"
          }
        }
      }
    },
    "stages": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "stage",
          "name",
          "primitive",
          "preconditions",
          "postconditions"
        ],
        "properties": {
          "stage": {
            "type": "integer"
          },
          "name": {
            "type": "string"
          },
          "primitive": {
            "type": "string",
            "enum": [
              "move",
              "grasp",
              "transport",
              "place",
              "release",
              "push"
            ]
          },
          "preconditions": {
            "type": "array",
            "items": {
              "$ref": "#/$defs/condition"
            }
          },
          "postconditions": {
            "type": "array",
            "items": {
              "$ref": "#/$defs/condition"
            }
          }
        }
      }
    }
  },
  "$defs": {
    "condition": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "condition",
        "alternatives",
        "failure_links"
      ],
      "properties": {
        "condition": {
          "type": "string"
        },
        "alternatives": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "maxItems": 0
        },
        "failure_links": {
          "type": "array",
          "minItems": 1,
          "maxItems": 1,
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": [
              "failure",
              "reason"
            ],
            "properties": {
              "failure": {
                "type": "string"
              },
              "reason": {
                "type": "string"
              }
            }
          }
        }
      }
    }
  }
}
```

## 6. Interactive Update stage: additional prompt path

BT Condition Studio's **Update stage** action uses `revise_stage_conditions()` in [llm_generation.py](src/aha_publish/behavior_trees/llm_generation.py). It uses the same BT generation system prompt, followed by the exact extension below. It calls the generator directly; this path does not run the normal separate reviewer pass.

### 6.1 Exact system-prompt extension

The complete system message is the expanded system in section 4.3 concatenated with this suffix, including its initial blank lines:

```text


INTERACTIVE SINGLE-STAGE REVISION
This request updates exactly one existing stage, rather than generating a whole
tree. Return exactly that stage in the stages array, with its original stage ID.
The user's selected primitive overrides automatic primitive classification for
this stage. Keep that primitive and use its catalogue placements to derive the
conditions. Preserve the task's factual object and gripper evidence.
Follow the stage's missing_description as an instruction to edit, add, or remove
conditions. Preserve unrelated existing checks and their selected state where
possible. If the primitive changed, re-derive all pre/postconditions for the new
primitive, removing checks that no longer apply and adding applicable checks.
Other stages are context only; do not return or change them. Use the full task
and current tree to interpret sequence-dependent and task-global rules.

```

### 6.2 Exact single-stage user-message template expression

`task_context` is the current description object, `current_stages` contains the whole edited tree including unsaved changes, and `stage` is the one stage being updated. Its selected primitive and `missing_description` carry the user's instructions. This is the exact expression used to assemble the user message:

```python
"TASK DESCRIPTION\n" + json.dumps(task_context, indent=2)
        + "\n\nVALID FAILURE NAMES\n"
        + json.dumps(failure_definition_summaries(failure_definitions), indent=2)
        + "\n\nCURRENT TREE (including unsaved user edits)\n"
        + json.dumps(current_stages, indent=2)
        + "\n\nSTAGE TO UPDATE (primitive is the user's selection; "
          "missing_description contains the requested changes)\n"
        + json.dumps(stage, indent=2)
```

This path validates that exactly one stage is returned, with the requested stage ID and the user's selected primitive. It preserves the stage summary/instruction fields and surviving checkbox selections, normalizes object names, and updates hold conditions as appropriate. It does not rewrite the source task-description file. There is no expanded live stage-update example here because no new stage-edit instruction has been supplied.

## 7. Model settings, output, and practical limits

The BT generation model uses `GENERATOR_OPENAI_MODEL`; review uses `OPENAI_MODEL`. Both have a configured fallback string of `gpt-5.6-sol`. The public `--model` option overrides both. The underlying CLI additionally supports `--generator-model` and `--reviewer-model`. The BT maximum output defaults to 16,000 tokens, controlled by `BT_CONDITION_MAX_TOKENS`. Model strings here are repository configuration values.

The schemas constrain output structure, but do not prove semantic correctness. The normal generation path does not mechanically enforce every placement or state rule after model output; the reviewer performs much of that audit. Existing saved artifacts can reflect older prompt versions or later manual edits.

The current BT system prompts contain both a strict null-held-object gate and transition-based wording for release. A release endpoint naturally has `held_object: null`, so those instructions can conflict. This document reproduces the current strings exactly, including existing wording and spelling; it does not repair prompts or rerun generation.

The runtime visual checker uses [condition_definitions.json](config/condition_definitions.json) for predicate interpretation during execution. That is a separate concern from the predicate-rule maker and BT maker documented here; the predicate-rule generator does not author that runtime file.

No standalone prompt export was added. The expanded blocks in this Markdown file are documentation snapshots. Changing taxonomy, catalogue, task facts, guidance, or code can change future requests.

## 8. Source map

| Source | Responsibility |
| --- | --- |
| [condition_taxonomy_input.json](config/condition_taxonomy_input.json) | Authored predicates, occurrence, and stage timing evidence |
| [condition_rule_generator.py](src/aha_publish/behavior_trees/condition_rule_generator.py) | Predicate system/user prompts, derived schemas, validation, and catalogue generation |
| [generated_condition_rules.json](src/aha_publish/behavior_trees/generated_condition_rules.json) | Shared generated catalogue read by BT maker |
| [03_behavior_trees/main.py](03_behavior_trees/main.py) | Public BT workflow |
| [prompts.py](src/aha_publish/behavior_trees/prompts.py) | BT/reviewer system templates, catalogue rendering, and response schemas |
| [prompt_building.py](src/aha_publish/behavior_trees/prompt_building.py) | BT generator user message |
| [llm_generation.py](src/aha_publish/behavior_trees/llm_generation.py) | Role assignment, generation, review, and stage updates |
| [io_context.py](src/aha_publish/behavior_trees/io_context.py) | Task-description loading |
| [failure_definitions.py](src/aha_publish/behavior_trees/failure_definitions.py) | Failure names, object aliases, and code-assigned hold conditions |
| [cli.py](src/aha_publish/behavior_trees/cli.py) | BT model/CLI settings and terminal generation |
| [gui.py](src/aha_publish/behavior_trees/gui.py) | Studio actions and saved result envelope |

