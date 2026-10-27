"""The catalogue generator writes JSON that the BT prompt loader can consume."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from aha_publish.behavior_trees import condition_rule_generator as generator, prompts


class ConditionRuleGeneratorTests(unittest.TestCase):
    def test_generation_writes_only_a_loadable_json_catalogue(self):
        source = {
            "taxonomy_name": "User rules",
            "scope": "Waypoint checks",
            "categories": [{
                "category": "User conditions",
                "category_explanation": "Conditions authored by the user.",
                "rules": [{
                    "source_condition_id": "user-1",
                    "failure": "UserFailure",
                    "predicate_template": "secure_hold(<object>) == True",
                    "occurrence": "every_matching_waypoint",
                    "stage_rule": "Check after grasping the object.",
                }],
            }],
        }
        derivations = generator.DerivationSet.model_validate({"derivations": [{
            "source_condition_id": "user-1",
            "primitive_timing": [{"primitive": "grasp", "check_type": ["postcondition"]}],
            "generation_rule": "Use the task-specified object name.",
            "prohibitions": [],
        }]})
        sdk = SimpleNamespace(OpenAI=mock.Mock(return_value=mock.sentinel.client))
        with tempfile.TemporaryDirectory() as folder:
            source_path = Path(folder) / "taxonomy.json"
            source_path.write_text(json.dumps(source))
            output_path = Path(folder) / "catalogue/generated_condition_rules.json"
            with mock.patch.object(generator, "DEFAULT_OUTPUT_JSON", output_path), mock.patch(
                "sys.argv", ["condition-rule-generator", "--input", str(source_path)]
            ), mock.patch.dict("sys.modules", {"openai": sdk}), mock.patch.dict(
                "os.environ", {"OPENAI_API_KEY": "test-key"}
            ), mock.patch.object(
                generator, "generate_derivations", return_value=derivations
            ) as derive, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(generator.main(), 0)

            self.assertEqual(list(output_path.parent.iterdir()), [output_path])
            loaded = prompts.load_condition_rules(output_path)
            rule = loaded["categories"][0]["rules"][0]
            self.assertEqual(rule["failure"], "UserFailure")
            self.assertEqual(rule["predicate_template"], "secure_hold(<object>) == True")
            self.assertEqual(rule["timing"]["check_type"], "postcondition")
            self.assertIs(derive.call_args.kwargs["client"], mock.sentinel.client)
            derive.assert_called_once()


if __name__ == "__main__":
    unittest.main()
