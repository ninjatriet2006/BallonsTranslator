"""Unit tests for modular LLM providers, presets, and aliases."""

import unittest
from ballontranslator.utils.project.llm_profiles import (
    PROVIDER_DEFAULTS,
    PROVIDER_ALIASES,
    canonical_provider,
    default_profile,
    LLMProfile,
    PROTOCOL_OPENAI_COMPLETION,
    PROTOCOL_ANTHROPIC_MESSAGES,
)
from ballontranslator.utils.project.llm_providers import (
    get_registered_providers,
    OpenAIProvider,
    DeepSeekProvider,
    GeminiProvider,
    AnthropicProvider,
    AgnesProvider,
    OpenRouterProvider,
    OllamaProvider,
    LMStudioProvider,
)


class TestModularLLMProviders(unittest.TestCase):
    """Test suite for provider presets and modular definitions."""

    def test_all_providers_registered(self):
        providers = get_registered_providers()
        expected = [
            "OpenAI",
            "DeepSeek",
            "Gemini",
            "Anthropic",
            "Agnes AI",
            "OpenRouter",
            "Ollama",
            "LM Studio",
        ]
        for name in expected:
            self.assertIn(name, providers, f"Provider '{name}' should be registered")
            self.assertIn(name, PROVIDER_DEFAULTS, f"'{name}' should be in PROVIDER_DEFAULTS")

    def test_provider_aliases(self):
        self.assertEqual(canonical_provider("Google"), "Gemini")
        self.assertEqual(canonical_provider("Vyceai"), "Agnes AI")
        self.assertEqual(canonical_provider("Agnes"), "Agnes AI")
        self.assertEqual(canonical_provider("agnes-ai"), "Agnes AI")
        self.assertEqual(canonical_provider("Claude"), "Anthropic")

    def test_agnes_provider_preset(self):
        profile = default_profile("Agnes AI")
        self.assertEqual(profile.name, "Agnes AI")
        self.assertEqual(profile.base_url, "https://vyceai.com/v1")
        self.assertEqual(profile.model, "agnes-3.0-flash")
        self.assertTrue(profile.support_vision)
        self.assertEqual(profile.vision_model, "agnes-3.0-flash")
        self.assertTrue(profile.support_image)
        self.assertEqual(profile.image_model, "agnes-image-2.0-flash")
        self.assertEqual(profile.protocol, PROTOCOL_OPENAI_COMPLETION)

    def test_anthropic_provider_preset(self):
        profile = default_profile("Anthropic")
        self.assertEqual(profile.name, "Anthropic")
        self.assertEqual(profile.base_url, "https://api.anthropic.com/v1")
        self.assertEqual(profile.protocol, PROTOCOL_ANTHROPIC_MESSAGES)
        self.assertTrue(profile.support_vision)
        self.assertIn("claude-3-7-sonnet-latest", profile.model_options)

    def test_deepseek_provider_preset(self):
        profile = default_profile("DeepSeek")
        self.assertEqual(profile.name, "DeepSeek")
        self.assertEqual(profile.base_url, "https://api.deepseek.com")
        self.assertFalse(profile.support_vision)
        self.assertFalse(profile.support_image)
        self.assertIn("deepseek-v4-flash", profile.model_options)


if __name__ == "__main__":
    unittest.main()
