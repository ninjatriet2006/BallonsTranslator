"""Ollama local model provider definition."""

from .base import BaseProvider, ProviderPreset, register_provider


@register_provider
class OllamaProvider(BaseProvider):
    """Local Ollama instance provider."""

    preset = ProviderPreset(
        id="ollama",
        name="Ollama",
        protocol="openai_completion",
        base_url="http://localhost:11434/v1/",
        require_api_key=False,
        model="llama3.1",
        model_options=["llama3.1", "qwen2.5", "mistral"],
        support_vision=True,
        vision_model="llama3.1",
        vision_model_options=["llama3.1", "qwen2.5", "mistral"],
        vision_detail_level="auto",
        support_image=False,
    )
