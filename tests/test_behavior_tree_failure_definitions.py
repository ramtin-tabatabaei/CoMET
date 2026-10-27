"""Failure links follow user catalogue data; task descriptions reach prompts intact."""

from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

from aha_publish.behavior_trees import conditions, failure_definitions, prompts, llm_generation
from aha_publish.behavior_trees.prompt_building import build_user_message


class FailureDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "user_rules.json"
        self.rules = {
            "categories": [{
                "category": "User conditions",
                "rules": [
                    {
                        "source_condition_id": "custom-hold",
                        "predicate_template": "secure_hold(<object>) == True",
                        "failure": "UserHoldingFailure",
                        "reason": "The user's holding definition.",
                        "placements": [{"primitive": "grasp", "condition_position": "postcondition"}],
                    },
                    {
                        "source_condition_id": "custom-objects",
                        "predicate_template": "objects_available(<object_1>, <object_2>, ...) == True",
                        "failure": "UserObjectsFailure",
                        "placements": [{"primitive": "move", "condition_position": "precondition"}],
                    },
                ],
            }],
        }
        self.save_rules()
        self.patch = mock.patch.object(prompts, "CONDITION_RULES_PATH", self.path)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def save_rules(self):
        self.path.write_text(json.dumps(self.rules))

    def test_missing_links_follow_changes_to_the_user_catalogue(self):
        block = {"condition": "secure_hold(orientation_target) == True"}
        link = conditions.normalize_condition(block)["failure_links"]
        self.assertEqual(link, [{"failure": "UserHoldingFailure", "reason": "The user's holding definition."}])

        self.rules["categories"][0]["rules"][0]["failure"] = "UpdatedUserFailure"
        self.save_rules()
        self.assertEqual(conditions.normalize_condition(block)["failure_links"][0]["failure"], "UpdatedUserFailure")

    def test_new_predicate_names_work_without_python_changes(self):
        self.rules["categories"][0]["rules"][0]["predicate_template"] = "verified_control(<object>) == Ready"
        self.save_rules()
        self.assertEqual(failure_definitions.catalogue_failure_links("secure_hold(cup) == True"), [])
        links = failure_definitions.catalogue_failure_links("verified_control(switch) == Ready")
        self.assertEqual(links[0]["failure"], "UserHoldingFailure")
        self.assertEqual(failure_definitions.catalogue_failure_links("verified_control(switch) == False"), [])

    def test_variadic_templates_match_one_or_many_objects(self):
        for arguments in ("cup", "cup, tray", "cup, tray, lid"):
            with self.subTest(arguments=arguments):
                links = failure_definitions.catalogue_failure_links(f"objects_available({arguments}) == True")
                self.assertEqual(links, [{"failure": "UserObjectsFailure", "reason": ""}])

    def test_unknown_or_ambiguous_conditions_are_not_given_guessed_failures(self):
        for text in ("unknown_orientation(target) == True", "object_in_gripper(cup) == True", "secure_hold(cup, lid) == True"):
            with self.subTest(text=text):
                self.assertEqual(conditions.normalize_condition({"condition": text})["failure_links"], [])
        duplicate = deepcopy(self.rules["categories"][0]["rules"][0])
        duplicate.update(source_condition_id="ambiguous", failure="ConflictingUserFailure")
        self.rules["categories"][0]["rules"].append(duplicate)
        self.save_rules()
        self.assertEqual(failure_definitions.catalogue_failure_links("secure_hold(cup) == True"), [])

    def test_explicit_failure_names_are_kept_without_alias_translation(self):
        for name in ("missed_grasp", "wrong_position", "CustomUserFailure"):
            with self.subTest(name=name):
                block = {"condition": "secure_hold(cup) == True", "failure_links": [{"failure": name, "reason": "User link."}]}
                self.assertEqual(conditions.normalize_condition(block)["failure_links"], block["failure_links"])

    def test_blank_links_use_catalogue_assignments(self):
        block = {"condition": "secure_hold(cup) == True", "failure_links": [{"failure": "", "reason": ""}]}
        self.assertEqual(conditions.normalize_condition(block)["failure_links"][0]["failure"], "UserHoldingFailure")

    def test_catalogue_validation_is_cached_without_sharing_mutable_rules(self):
        with mock.patch.object(prompts, "validate_condition_rules", wraps=prompts.validate_condition_rules) as validate:
            first = prompts.load_condition_rules()
            first["categories"][0]["rules"][0]["failure"] = "CallerEdit"
            second = prompts.load_condition_rules()
            self.assertEqual(second, self.rules)
            validate.assert_called_once()

    def test_cached_catalogue_detects_same_size_edits_with_preserved_timestamp(self):
        with mock.patch.object(prompts, "validate_condition_rules", wraps=prompts.validate_condition_rules) as validate:
            prompts.load_condition_rules()
            previous_stat = self.path.stat()
            self.rules["categories"][0]["rules"][0]["failure"] = "UpdatedUserFailure"
            self.save_rules()
            self.assertEqual(self.path.stat().st_size, previous_stat.st_size)
            os.utime(self.path, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
            self.assertEqual(prompts.load_condition_rules(), self.rules)
            self.assertEqual(validate.call_count, 2)

    def test_cached_catalogue_does_not_hide_invalid_or_missing_files(self):
        prompts.load_condition_rules()
        self.path.write_text("{invalid JSON")
        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            prompts.load_condition_rules()
        self.path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "catalogue not found"):
            prompts.load_condition_rules()

    def test_reviewer_uses_the_updated_catalogue_without_restarting(self):
        events = [SimpleNamespace(type="response.output_text.delta", delta='{"violations": [], "stages": []}')]
        create = mock.Mock(return_value=events)
        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        with redirect_stdout(StringIO()):
            llm_generation.review_conditions_with_openai(client, "Draft to review")
            before = create.call_args.kwargs["input"][0]["content"]
            self.rules["categories"][0]["rules"][0]["predicate_template"] = "verified_control(<object>) == Ready"
            self.save_rules()
            llm_generation.review_conditions_with_openai(client, "Draft to review")
            after = create.call_args.kwargs["input"][0]["content"]
        self.assertIn("secure_hold(<object>) == True", before)
        self.assertIn("verified_control(<object>) == Ready", after)
        self.assertNotIn("secure_hold(<object>)", after)


class PromptContextTests(unittest.TestCase):
    def test_full_json_description_is_embedded_without_preprocessing(self):
        context = json.dumps({
            "key_scene_objects": [{"name": "basketball", "original_name": "ball"}],
            "waypoints": [{
                "waypoint": 0,
                "distance_m": 0.2,
                "offset": [0.1, 0.2, 0.3],
                "tolerance": 0.01,
                "approx_distance_to_objects": {"ball": "20 cm"},
                "waypoint_relationship_used": "ball above hoop",
            }],
        }, indent=4)
        message = build_user_message(context, {"failure_categories": {"UserFailure": {"id": "UserFailure"}}})
        embedded = message.split("TASK DESCRIPTION\n\n", 1)[1].split("\n\nVALID FAILURE NAMES", 1)[0]
        self.assertEqual(embedded, context)
        self.assertIn("- UserFailure", message)

    def test_plain_text_description_is_also_preserved(self):
        context = "original_name ball; distance 0.2 m; offset 20 cm; tolerance 0.01"
        self.assertIn(context, build_user_message(context, {}))


if __name__ == "__main__":
    unittest.main()
