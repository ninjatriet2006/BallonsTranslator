"""OpenRouter aggregator provider definition."""

from .base import BaseProvider, ProviderPreset, register_provider


@register_provider
class OpenRouterProvider(BaseProvider):
    """OpenRouter API provider offering diverse model access."""

    preset = ProviderPreset(
        id="openrouter",
        name="OpenRouter",
        protocol="openai_completion",
        base_url="https://openrouter.ai/api/v1",
        require_api_key=True,
        model="openai/gpt-5.5",
        model_options=[
            "openai/gpt-5.5", "openai/gpt-5.4", "openai/gpt-4o",
            "anthropic/claude-sonnet-4", "~anthropic/claude-sonnet-latest", "~x-ai/grok-latest",
            "~google/gemini-flash-latest", "~google/gemini-pro-latest",
            "qwen/qwen3.7-plus", "qwen/qwen3.7-max", "qwen/qwen3.6-plus",
        ],
        support_vision=True,
        vision_model="openai/gpt-5.5",
        vision_model_options=[
            "openai/gpt-5.5", "openai/gpt-5.4", "openai/gpt-4o",
            "anthropic/claude-sonnet-4", "~anthropic/claude-sonnet-latest", "~x-ai/grok-latest",
            "~google/gemini-flash-latest", "~google/gemini-pro-latest",
            "qwen/qwen3.7-plus", "qwen/qwen3.7-max", "qwen/qwen3.6-plus",
        ],
        vision_detail_level="auto",
        support_image=True,
        image_base_url="https://openrouter.ai/api/v1/images",
        image_model="black-forest-labs/flux.2-klein-4b",
        image_model_options=["black-forest-labs/flux.2-klein-4b"],
    )
