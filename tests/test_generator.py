import os
import unittest
from unittest.mock import patch

from shared import generator


class GeneratorConfigurationTests(unittest.TestCase):
    def test_openai_is_the_default_provider(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(generator.provider_name(), "openai")
            self.assertEqual(generator.model_name(), generator.DEFAULT_OPENAI_MODEL)

    def test_huggingface_model_can_be_configured(self):
        env = {"LLM_PROVIDER": "huggingface", "HF_MODEL": "example/local-model"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(generator.provider_name(), "huggingface")
            self.assertEqual(generator.model_name(), "example/local-model")

    def test_unknown_provider_is_rejected_before_generation(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "unknown"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "LLM_PROVIDER"):
                generator.generate_chat([{"role": "user", "content": "Hello"}])

    def test_huggingface_does_not_require_an_api_key(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "huggingface"}, clear=True):
            self.assertIsNone(generator.configuration_error())


if __name__ == "__main__":
    unittest.main()
