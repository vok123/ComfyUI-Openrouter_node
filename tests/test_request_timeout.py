import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class FakeRequestException(Exception):
    def __init__(self, message="", response=None):
        super().__init__(message)
        self.response = response


class FakeTimeout(FakeRequestException):
    pass


def install_dependency_stubs():
    requests_module = types.ModuleType("requests")
    exceptions = types.SimpleNamespace(
        RequestException=FakeRequestException,
        Timeout=FakeTimeout,
    )
    requests_module.exceptions = exceptions
    requests_module.get = Mock()
    requests_module.post = Mock()
    sys.modules["requests"] = requests_module

    torch_module = types.ModuleType("torch")
    torch_module.float32 = "float32"
    torch_module.Tensor = type("Tensor", (), {})
    torch_module.zeros = Mock(return_value="placeholder-image")
    sys.modules["torch"] = torch_module

    tiktoken_module = types.ModuleType("tiktoken")
    tiktoken_module.get_encoding = Mock()
    sys.modules["tiktoken"] = tiktoken_module

    pil_module = types.ModuleType("PIL")
    image_module = types.ModuleType("PIL.Image")
    image_module.open = Mock()
    image_module.fromarray = Mock()
    pil_module.Image = image_module
    sys.modules["PIL"] = pil_module
    sys.modules["PIL.Image"] = image_module

    sys.modules.setdefault("numpy", types.ModuleType("numpy"))


def load_node_module():
    install_dependency_stubs()
    root = Path(__file__).resolve().parents[1]
    package_name = "comfy_openrouter_node_test"

    package = types.ModuleType(package_name)
    package.__path__ = [str(root)]
    sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.node",
        root / "node.py",
        submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RequestTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.node_module = load_node_module()
        self.node = self.node_module.OpenRouterNode()
        self.node.fetch_credits = Mock(return_value="Remaining: $1.000")
        self.node.count_tokens = Mock(return_value=1)

    def call_generate_response(self, request_timeout=120, reasoning_effort="auto"):
        return self.node.generate_response(
            api_key="test-key",
            system_prompt="system",
            user_message_box="hello",
            model="openai/gpt-4o",
            web_search=False,
            cheapest=False,
            fastest=False,
            temperature=1.0,
            pdf_engine="auto",
            chat_mode=False,
            request_timeout=request_timeout,
            reasoning_effort=reasoning_effort,
        )

    def test_main_openrouter_request_uses_configured_timeout(self):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "done"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }
        self.node_module.requests.post.return_value = response

        result = self.call_generate_response(request_timeout=45)

        self.assertEqual(result[0], "done")
        self.node_module.requests.post.assert_called_once()
        self.assertEqual(self.node_module.requests.post.call_args.kwargs["timeout"], 45)

    def test_default_reasoning_effort_omits_reasoning_override(self):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "done"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }
        self.node_module.requests.post.return_value = response

        result = self.call_generate_response()

        self.assertEqual(result[0], "done")
        payload = self.node_module.requests.post.call_args.kwargs["json"]
        self.assertNotIn("reasoning", payload)

    def test_explicit_reasoning_effort_is_sent(self):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "done"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }
        self.node_module.requests.post.return_value = response

        result = self.call_generate_response(reasoning_effort="high")

        self.assertEqual(result[0], "done")
        payload = self.node_module.requests.post.call_args.kwargs["json"]
        self.assertEqual(payload["reasoning"], {"effort": "high"})

    def test_invalid_reasoning_effort_falls_back_to_auto(self):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "done"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }
        self.node_module.requests.post.return_value = response

        result = self.call_generate_response(reasoning_effort="unsupported")

        self.assertEqual(result[0], "done")
        payload = self.node_module.requests.post.call_args.kwargs["json"]
        self.assertNotIn("reasoning", payload)

    def test_is_changed_includes_reasoning_effort(self):
        base_args = dict(
            api_key="test-key",
            system_prompt="system",
            user_message_box="hello",
            model="openai/gpt-4o",
            web_search=False,
            cheapest=False,
            fastest=False,
            temperature=1.0,
            pdf_engine="auto",
            chat_mode=False,
            request_timeout=120,
        )

        low_key = self.node_module.OpenRouterNode.IS_CHANGED(**base_args, reasoning_effort="low")
        high_key = self.node_module.OpenRouterNode.IS_CHANGED(**base_args, reasoning_effort="high")

        self.assertNotEqual(low_key, high_key)

    def test_timeout_exception_returns_clear_error(self):
        self.node_module.requests.post.side_effect = FakeTimeout("request timed out")

        result = self.call_generate_response(request_timeout=2)

        self.assertIn("API Request Error", result[0])
        self.assertIn("request timed out", result[0])
        self.assertEqual(result[2], "Stats N/A due to error")
        self.assertEqual(result[3], "Credits N/A due to error")

    def test_fetch_credits_uses_configured_timeout(self):
        self.node.fetch_credits = self.node_module.OpenRouterNode().fetch_credits
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "data": {
                "total_credits": 10.0,
                "total_usage": 3.25,
            }
        }
        self.node_module.requests.get.return_value = response

        credits = self.node.fetch_credits("test-key", timeout=45)

        self.assertEqual(credits, "Remaining: $6.750")
        self.node_module.requests.get.assert_called_once()
        self.assertEqual(self.node_module.requests.get.call_args.kwargs["timeout"], 45)


class ImageGenerationTests(unittest.TestCase):
    def setUp(self):
        self.node_module = load_node_module()
        self.node = self.node_module.OpenRouterNode()
        self.node.fetch_credits = Mock(return_value="Remaining: $1.000")
        self.node.base64_to_image = Mock(return_value="decoded-image")
        self.node.image_to_base64 = Mock(return_value="fake-base64")
        cls = self.node_module.OpenRouterNode
        cls.image_capabilities_cache = dict(cls.fallback_image_model_capabilities)
        cls.image_capabilities_last_fetch_time = 10**12

    def call_generate_response(self, model="bytedance-seed/seedream-5-0-pro", **kwargs):
        args = dict(
            api_key="test-key",
            system_prompt="system",
            user_message_box="a cat in space",
            model=model,
            web_search=True,
            cheapest=True,
            fastest=False,
            temperature=1.0,
            pdf_engine="auto",
            chat_mode=False,
            request_timeout=45,
            aspect_ratio="16:9 (1344x768)",
            image_resolution="2K",
            seed=42,
        )
        args.update(kwargs)
        return self.node.generate_response(**args)

    def test_image_models_use_images_endpoint(self):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "created": 1748372400,
            "data": [{"b64_json": "abc123"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 10, "cost": 0.04},
        }
        self.node_module.requests.post.return_value = response

        result = self.call_generate_response()

        self.assertEqual(result[0], "Image generated with bytedance-seed/seedream-5-0-pro")
        self.assertEqual(result[1], "decoded-image")
        self.node_module.requests.post.assert_called_once()
        self.assertEqual(
            self.node_module.requests.post.call_args.args[0],
            "https://openrouter.ai/api/v1/images",
        )
        payload = self.node_module.requests.post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "bytedance-seed/seedream-5-0-pro")
        self.assertEqual(payload["prompt"], "a cat in space")
        self.assertEqual(payload["aspect_ratio"], "16:9")
        self.assertEqual(payload["resolution"], "2K")
        self.assertEqual(payload["seed"], 42)
        self.assertNotIn("messages", payload)
        self.assertNotIn("reasoning", payload)

    def test_image_models_do_not_apply_chat_modifiers(self):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "created": 1748372400,
            "data": [{"b64_json": "abc123"}],
        }
        self.node_module.requests.post.return_value = response

        self.call_generate_response(model="qwen/qwen-image-3-pro")

        payload = self.node_module.requests.post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "qwen/qwen-image-3-pro")

    def test_image_models_omit_unsupported_default_params(self):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "created": 1748372400,
            "data": [{"b64_json": "abc123"}],
        }
        self.node_module.requests.post.return_value = response

        self.call_generate_response(
            model="qwen/qwen-image-3",
            aspect_ratio="auto",
            image_resolution="1K",
            seed=0,
        )

        payload = self.node_module.requests.post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "qwen/qwen-image-3")
        self.assertEqual(payload["prompt"], "a cat in space")
        self.assertEqual(payload["resolution"], "1K")
        self.assertNotIn("aspect_ratio", payload)
        self.assertNotIn("seed", payload)
        self.assertNotIn("n", payload)

    def test_seedream_lite_clamps_1k_resolution_to_2k(self):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "created": 1748372400,
            "data": [{"b64_json": "abc123"}],
        }
        self.node_module.requests.post.return_value = response

        self.call_generate_response(
            model="bytedance-seed/seedream-5-0-lite",
            aspect_ratio="auto",
            image_resolution="1K",
            seed=0,
        )

        payload = self.node_module.requests.post.call_args.kwargs["json"]
        self.assertEqual(payload["resolution"], "2K")
        self.assertEqual(payload["aspect_ratio"], "auto")
        self.assertNotIn("seed", payload)

    def test_seedream_pro_clamps_4k_resolution_to_2k(self):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "created": 1748372400,
            "data": [{"b64_json": "abc123"}],
        }
        self.node_module.requests.post.return_value = response

        self.call_generate_response(
            model="bytedance-seed/seedream-5-0-pro",
            image_resolution="4K",
        )

        payload = self.node_module.requests.post.call_args.kwargs["json"]
        self.assertEqual(payload["resolution"], "2K")

    def test_grok_omits_unsupported_seed(self):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "created": 1748372400,
            "data": [{"b64_json": "abc123"}],
        }
        self.node_module.requests.post.return_value = response

        self.call_generate_response(
            model="x-ai/grok-imagine-image-2.0",
            image_resolution="1K",
            seed=42,
        )

        payload = self.node_module.requests.post.call_args.kwargs["json"]
        self.assertEqual(payload["resolution"], "1K")
        self.assertNotIn("seed", payload)

    def test_qwen_clamps_unsupported_aspect_ratio(self):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "created": 1748372400,
            "data": [{"b64_json": "abc123"}],
        }
        self.node_module.requests.post.return_value = response

        self.call_generate_response(
            model="qwen/qwen-image-3",
            aspect_ratio="21:9 (1536x672)",
            image_resolution="4K",
            seed=7,
        )

        payload = self.node_module.requests.post.call_args.kwargs["json"]
        self.assertEqual(payload["aspect_ratio"], "2:1")
        self.assertEqual(payload["resolution"], "2K")
        self.assertEqual(payload["seed"], 7)

    def test_grok_truncates_input_references(self):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "created": 1748372400,
            "data": [{"b64_json": "abc123"}],
        }
        self.node_module.requests.post.return_value = response

        self.call_generate_response(
            model="x-ai/grok-imagine-image-2.0",
            image_1="a",
            image_2="b",
            image_3="c",
            image_4="d",
        )

        payload = self.node_module.requests.post.call_args.kwargs["json"]
        self.assertEqual(len(payload["input_references"]), 3)

    def test_image_models_send_input_references(self):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "created": 1748372400,
            "data": [{"b64_json": "abc123"}],
        }
        self.node_module.requests.post.return_value = response

        self.call_generate_response(image_1="tensor-a", image_2="tensor-b")

        payload = self.node_module.requests.post.call_args.kwargs["json"]
        self.assertEqual(
            payload["input_references"],
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,fake-base64"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,fake-base64"},
                },
            ],
        )

    def test_text_models_still_use_chat_completions(self):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "done"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }
        self.node_module.requests.post.return_value = response

        result = self.call_generate_response(model="openai/gpt-4o", cheapest=False, web_search=False)

        self.assertEqual(result[0], "done")
        self.assertEqual(
            self.node_module.requests.post.call_args.args[0],
            "https://openrouter.ai/api/v1/chat/completions",
        )


class DynamicImageKeyTests(unittest.TestCase):
    def setUp(self):
        self.node_module = load_node_module()
        self.node = self.node_module.OpenRouterNode

    def test_dynamic_image_keys_ignore_image_resolution(self):
        self.assertFalse(self.node.is_dynamic_image_key("image_resolution"))
        self.assertFalse(self.node.is_dynamic_image_key("image"))
        self.assertFalse(self.node.is_dynamic_image_key("aspect_ratio"))
        self.assertTrue(self.node.is_dynamic_image_key("image_1"))
        self.assertTrue(self.node.is_dynamic_image_key("image_12"))
        self.assertEqual(
            self.node.dynamic_image_keys({
                "image_2": "b",
                "image_resolution": "1K",
                "image_1": "a",
                "aspect_ratio": "1:1",
            }),
            ["image_1", "image_2"],
        )


if __name__ == "__main__":
    unittest.main()
