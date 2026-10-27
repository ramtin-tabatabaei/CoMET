"""Generation and saved reviews retain condition choices made by the model or user."""

from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest import mock

from aha_publish.behavior_trees import gui, io_context, llm_generation as generation


def condition(text, failure="ExecutionSequenceMismatch"):
    return {
        "condition": text,
        "alternatives": [],
        "failure_links": [{"failure": failure, "reason": "Saved check."}],
    }


def example_stages():
    # These combinations exercise rules that the retired repair passes rewrote.
    return [
        {
            "stage": 0,
            "name": "Move to the button",
            "primitive": "move",
            "preconditions": [
                condition("Object_found(red_button) == True", "ObjectNotFound"),
                condition("selected_object(red_button) == True", "WrongObjectSelection"),
                condition("end_effector_aligned_with(red_button) == true", "WrongPosition"),
                condition("catalogue_extension(red_button) == True"),
            ],
            "postconditions": [],
        },
        {
            "stage": 1,
            "name": "Push the button",
            "primitive": "push",
            "preconditions": [condition("object_for_press(red_button) == True")],
            "postconditions": [
                condition("object_in_gripper(red_button) == True"),
                condition("gripper_condition = Closed"),
            ],
        },
        {
            "stage": 2,
            "name": "Release the cup",
            "primitive": "release",
            "preconditions": [
                condition("object_in_gripper(cup) == True"),
                condition("inside(cup, tray) == true", "WrongPosition"),
            ],
            "postconditions": [condition("gripper_released(cup) == True")],
        },
        {
            "stage": 3,
            "name": "Transport the cup",
            "primitive": "transport",
            "preconditions": [condition("object_in_gripper(cup) == True")],
            "postconditions": [condition("object_in_gripper(other_cup) == True")],
        },
    ]


class BehaviorTreeGenerationTests(unittest.TestCase):
    def test_context_reload_sees_same_timestamp_edits_and_ignores_caller_mutations(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "context.json"
            path.write_text('{"task": "first"}')
            initial_stat = path.stat()
            loaded = io_context.load_task_context_object(path)
            loaded["task"] = "caller mutation"
            self.assertEqual(json.loads(io_context.load_task_context(path)), {"task": "first"})
            path.write_text('{"task": "other"}')
            os.utime(path, ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns))
            self.assertEqual(json.loads(io_context.load_task_context(path)), {"task": "other"})

    def assert_conditions_equal(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for before, after in zip(expected, actual):
            self.assertEqual(before["primitive"], after["primitive"])
            self.assertEqual(before["preconditions"], after["preconditions"])
            self.assertEqual(before["postconditions"], after["postconditions"])

    def test_generation_preserves_draft_or_reviewer_conditions(self):
        for reviewer in (False, True):
            with self.subTest(reviewer=reviewer):
                draft = example_stages()
                reviewed = deepcopy(draft)
                reviewed[1]["postconditions"] = [condition("object_for_press(red_button) == True")]
                violations = [{"stage": 1, "rule": "review", "detail": "Press contact.", "fix": "Changed check."}]
                with mock.patch.object(
                    generation, "generate_conditions_with_openai", return_value={"stages": deepcopy(draft)}
                ), mock.patch.object(
                    generation, "review_conditions_with_openai",
                    return_value={"stages": deepcopy(reviewed), "violations": violations},
                ) as review_call, redirect_stdout(StringIO()):
                    result = generation.generate_conditions({}, "{}", object(), "openai", reviewer=reviewer)

                self.assert_conditions_equal(reviewed if reviewer else draft, result["stages"])
                self.assertEqual(result["source"], "api_generation_raw")
                self.assertEqual(result["token_usage"], {"records": []})
                for stage in result["stages"]:
                    self.assertEqual(stage["hold_conditions"],
                                     generation.build_hold_condition_blocks(stage["primitive"]))
                if reviewer:
                    review_call.assert_called_once()
                    self.assertEqual(result["review"]["violations"], violations)
                else:
                    review_call.assert_not_called()
                    self.assertNotIn("review", result)

    def test_generation_still_normalizes_simulator_handles(self):
        stages = [{
            "stage": 0, "name": "Grasp the ball", "primitive": "grasp",
            "preconditions": [condition("selected_object(ball_handle) == True", "WrongObjectSelection")],
            "postconditions": [condition("object_in_gripper(ball_handle) == True")],
        }]
        context = json.dumps({"key_scene_objects": [{"name": "basketball", "original_name": "ball_handle"}]})
        with mock.patch.object(
            generation, "generate_conditions_with_openai", return_value={"stages": stages}
        ), redirect_stdout(StringIO()):
            result = generation.generate_conditions({}, context, object(), "openai", reviewer=False)
        stage = result["stages"][0]
        self.assertEqual(stage["preconditions"][0]["condition"], "selected_object(basketball) == True")
        self.assertEqual(stage["postconditions"][0]["condition"], "object_in_gripper(basketball) == True")

    def test_saved_tree_reopens_without_repairing_conditions(self):
        stages = example_stages()
        stages[0]["hold_conditions"] = []
        stages[1]["hold_conditions"] = [{"detector": "collision"}]
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(
                provider="openai", task_name="example_task",
                task_context=Path(folder) / "context.json",
                review_output=Path(folder) / "example_task.bt_conditions.json",
            )
            args.task_context.write_text("{}")
            for payload_key in ("generated", "review"):
                with self.subTest(payload_key=payload_key):
                    gui.save_review_file(args, {payload_key: {"stages": stages}})
                    before = args.review_output.read_bytes()
                    result = gui.load_saved_conditions(args)
                    self.assert_conditions_equal(stages, result["stages"])
                    self.assertEqual(result["source"], "saved_review")
                    self.assertEqual(result["stages"][0]["hold_conditions"], [])
                    self.assertEqual([item["detector"] for item in result["stages"][1]["hold_conditions"]], ["collision"])
                    self.assertEqual(result["stages"][2]["hold_conditions"], generation.build_hold_condition_blocks())
                    self.assertEqual(args.review_output.read_bytes(), before)

    def test_scene_names_replace_arguments_without_renaming_predicates(self):
        stages = [{
            "name": "open the ball container",
            "preconditions": [{
                **condition("open(open) == True and Object_found(ball, ball_copy) == True"),
                "alternatives": ["inside(ball, open) == true"],
            }],
        }]
        context = {"key_scene_objects": [
            {"name": "cabinet", "original_name": "open"},
            {"name": "basketball", "original_name": "ball, ball_copy"},
        ]}
        failure_links = deepcopy(stages[0]["preconditions"][0]["failure_links"])
        result = generation.normalize_scene_object_names(stages, context)[0]
        block = result["preconditions"][0]
        self.assertEqual(block["condition"], "open(cabinet) == True and Object_found(basketball) == True")
        self.assertEqual(block["alternatives"], ["inside(basketball, cabinet) == true"])
        self.assertEqual(block["failure_links"], failure_links)
        self.assertEqual(result["name"], "open the ball container")

    def test_overlapping_generations_keep_separate_token_totals(self):
        before_usage = Barrier(2)
        after_usage = Barrier(2)

        def generate(input_tokens):
            def stream(**request):
                yield SimpleNamespace(type="response.output_text.delta", delta=json.dumps({"stages": example_stages()}))
                before_usage.wait(timeout=5)
                yield SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=3)),
                )
                after_usage.wait(timeout=5)

            client = SimpleNamespace(responses=SimpleNamespace(create=stream))
            return generation.generate_conditions({}, "{}", client, "openai", reviewer=False)

        with redirect_stdout(StringIO()), ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(generate, input_tokens) for input_tokens in (10, 20)]
            results = [future.result(timeout=10) for future in futures]

        for input_tokens, result in zip((10, 20), results):
            records = result["token_usage"]["records"]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["input_tokens"], input_tokens)
            self.assertEqual(records[0]["output_tokens"], 3)

    def test_saved_tree_keeps_selections_and_original_text_after_repeated_saves(self):
        stage = {
            "stage": 0, "name": "Open gripper", "primitive": "release",
            "preconditions": [{
                **condition("Object_found(red_button) == True", "ObjectNotFound"),
                "selected": False,
                "original_condition": "Object_found(button_handle) == True",
            }],
            "postconditions": [{
                **condition("gripper_condition = Open"),
                "selected": True,
                "original_condition": "gripper_is_open == true",
            }],
        }
        saved_stage = deepcopy(stage)
        for section in ("preconditions", "postconditions"):
            duplicate = deepcopy(stage[section][0])
            duplicate["condition"] = duplicate["condition"].upper()
            duplicate["selected"] = not duplicate["selected"]
            saved_stage[section].append(duplicate)

        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(
                provider="openai", task_name="example_task",
                task_context=Path(folder) / "context.json",
                review_output=Path(folder) / "custom_output.json",
            )
            args.task_context.write_text("{}")
            for payload_key in ("generated", "review"):
                with self.subTest(payload_key=payload_key):
                    gui.save_review_file(args, {payload_key: {"stages": [saved_stage]}})
                    for _ in range(2):
                        loaded = gui.load_saved_conditions(args)
                        self.assert_conditions_equal([stage], loaded["stages"])
                        self.assertEqual(loaded["source_path"], str(args.review_output))
                        gui.save_review_file(args, {"review": loaded})

    def test_saved_tree_ignores_missing_invalid_and_other_task_files(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(
                task_context=Path(folder) / "context.json",
                review_output=Path(folder) / "custom_output.json",
            )
            self.assertIsNone(gui.load_saved_conditions(args))
            args.review_output.write_text("invalid JSON")
            self.assertIsNone(gui.load_saved_conditions(args))
            args.review_output.write_text(json.dumps({
                "task_context_path": str(Path(folder) / "other_context.json"),
                "review": {"stages": example_stages()},
            }))
            self.assertIsNone(gui.load_saved_conditions(args))

    def test_reviewer_failure_keeps_the_generated_draft(self):
        stages = example_stages()
        with mock.patch.object(
            generation, "generate_conditions_with_openai", return_value={"stages": deepcopy(stages)}
        ), mock.patch.object(
            generation, "review_conditions_with_openai", side_effect=RuntimeError("Reviewer unavailable")
        ), redirect_stdout(StringIO()):
            result = generation.generate_conditions({}, "{}", object(), "openai", reviewer=True)
        self.assert_conditions_equal(stages, result["stages"])
        self.assertNotIn("review", result)


if __name__ == "__main__":
    unittest.main()
