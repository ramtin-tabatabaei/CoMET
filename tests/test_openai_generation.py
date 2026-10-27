"""OpenAI client construction, description requests, and generation options."""

from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest import mock

from aha_publish.behavior_trees import cli as bt_cli, llm_generation
from aha_publish.descriptions import cli as description_cli, providers, runner


class OpenAIGenerationTests(unittest.TestCase):
    def test_both_client_factories_use_the_openai_environment_key(self):
        for module in (llm_generation, providers):
            with self.subTest(module=module.__name__):
                sdk = SimpleNamespace(OpenAI=mock.Mock(return_value=mock.sentinel.client))
                with mock.patch.dict("sys.modules", {"openai": sdk}), mock.patch.dict(
                    "os.environ", {"OPENAI_API_KEY": "test-key"}
                ), mock.patch.object(llm_generation.importlib.util, "find_spec", return_value=object()):
                    self.assertIs(module.build_client("openai"), mock.sentinel.client)
                sdk.OpenAI.assert_called_once_with(api_key="test-key")

    def test_programmatic_calls_reject_unsupported_providers_before_api_work(self):
        for module in (llm_generation, providers):
            with self.subTest(module=module.__name__):
                with self.assertRaisesRegex(ValueError, "Only OpenAI is supported"):
                    module.build_client("unsupported")
        with mock.patch.object(llm_generation, "generate_conditions_with_openai") as generate:
            with self.assertRaisesRegex(ValueError, "Only OpenAI is supported"):
                llm_generation.generate_conditions({}, "{}", None, "unsupported")
            generate.assert_not_called()
        with mock.patch.object(llm_generation, "review_conditions_with_openai") as review:
            with self.assertRaisesRegex(ValueError, "Only OpenAI is supported"):
                llm_generation.review_conditions([], "{}", None, "unsupported")
            review.assert_not_called()
        with mock.patch.object(runner, "discover_waypoints") as discover:
            with self.assertRaisesRegex(ValueError, "Only OpenAI is supported"):
                runner.analyze_task(None, SimpleNamespace(provider="unsupported"), None)
            discover.assert_not_called()

    def test_description_request_keeps_images_model_and_usage(self):
        response = SimpleNamespace(output_text='{"task": "example_task"}', usage=SimpleNamespace(input_tokens=17, output_tokens=4))
        client = SimpleNamespace(responses=SimpleNamespace(create=mock.Mock(return_value=response)))
        task = SimpleNamespace(image_paths=[SimpleNamespace(path="grid.png")])
        args = SimpleNamespace(include_grid_image=True, openai_model="custom-model", max_output_tokens=512)
        with mock.patch.object(providers, "load_image_as_base64", return_value=("image-data", "image/png")):
            result, usage = providers.analyze_with_openai(task, args, "Task description", client)
        self.assertEqual(result, {"task": "example_task"})
        self.assertEqual((usage.input_tokens, usage.output_tokens, usage.calls), (17, 4, 1))
        request = client.responses.create.call_args.kwargs
        self.assertEqual(request["model"], "custom-model")
        self.assertEqual(request["max_output_tokens"], 512)
        content = request["input"][1]["content"]
        self.assertEqual(content, [
            {"type": "input_text", "text": "Task description"},
            {"type": "input_image", "image_url": "data:image/png;base64,image-data"},
        ])

    def test_internal_clis_accept_only_openai(self):
        for module in (bt_cli, description_cli):
            with self.subTest(module=module.__name__):
                with mock.patch("sys.argv", ["generator", "--provider", "openai"]):
                    self.assertEqual(module.parse_args().provider, "openai")
                with mock.patch("sys.argv", ["generator", "--provider", "unsupported"]), redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        module.parse_args()
                    self.assertEqual(error.exception.code, 2)

    def test_bt_model_overrides_keep_generator_and_reviewer_separate(self):
        args = SimpleNamespace(model="shared-model", generator_model="generator-model", reviewer_model="reviewer-model")
        with mock.patch.object(llm_generation, "GENERATOR_OPENAI_MODEL"), mock.patch.object(llm_generation, "OPENAI_MODEL"):
            bt_cli.apply_model_overrides(args)
            self.assertEqual(llm_generation.GENERATOR_OPENAI_MODEL, "generator-model")
            self.assertEqual(llm_generation.OPENAI_MODEL, "reviewer-model")


if __name__ == "__main__":
    unittest.main()
