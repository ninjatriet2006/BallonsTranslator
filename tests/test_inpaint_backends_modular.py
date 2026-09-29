"""Unit tests for modular inpaint backends."""

import io
import unittest
from PIL import Image

from ballontranslator.modules.inpaint.llm.backends import (
    get_image_backend,
    AgnesImageBackend,
    OpenAIImageBackend,
    GeminiImageBackend,
    OpenRouterImageBackend,
)
from ballontranslator.utils.project.llm_profiles import LLMProfile


class TestModularInpaintBackends(unittest.TestCase):
    """Test suite for modular inpaint backends."""

    def test_agnes_backend_matching(self):
        backend1 = get_image_backend(None, model="agnes-image-2.0-flash")
        self.assertIsInstance(backend1, AgnesImageBackend)

        backend2 = get_image_backend(None, base_url="https://vyceai.com/v1")
        self.assertIsInstance(backend2, AgnesImageBackend)

        backend3 = get_image_backend(None, base_url="https://apihub.agnes-ai.com/v1")
        self.assertIsInstance(backend3, AgnesImageBackend)

        profile = LLMProfile(name="Vyceai")
        backend4 = get_image_backend(profile)
        self.assertIsInstance(backend4, AgnesImageBackend)

    def test_openai_backend_matching(self):
        profile = LLMProfile(name="OpenAI", base_url="https://api.openai.com/v1")
        backend = get_image_backend(profile)
        self.assertIsInstance(backend, OpenAIImageBackend)

    def test_gemini_backend_matching(self):
        profile = LLMProfile(name="Gemini", base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        backend = get_image_backend(profile)
        self.assertIsInstance(backend, GeminiImageBackend)

    def test_openrouter_backend_matching(self):
        profile = LLMProfile(name="OpenRouter", base_url="https://openrouter.ai/api/v1")
        backend = get_image_backend(profile)
        self.assertIsInstance(backend, OpenRouterImageBackend)

    def test_agnes_endpoint_and_aspect_ratio_size(self):
        endpoint = AgnesImageBackend.get_endpoint("https://vyceai.com/v1")
        self.assertEqual(endpoint, "https://vyceai.com/v1/images/generations")

        # Landscape image test
        img = Image.new("RGB", (1200, 600), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        profile = LLMProfile(name="Agnes AI", image_model="agnes-image-2.0-flash")
        args = AgnesImageBackend.build_api_args(profile, buf)
        self.assertEqual(args["model"], "agnes-image-2.0-flash")
        self.assertEqual(args["size"], "1024x768")
        self.assertIn("extra_body", args)
        self.assertIn("image", args["extra_body"])

        # Portrait image test
        img_portrait = Image.new("RGB", (600, 1200), color=(255, 255, 255))
        buf_p = io.BytesIO()
        img_portrait.save(buf_p, format="PNG")
        buf_p.seek(0)

        args_p = AgnesImageBackend.build_api_args(profile, buf_p)
        self.assertEqual(args_p["size"], "768x1024")


if __name__ == "__main__":
    unittest.main()
