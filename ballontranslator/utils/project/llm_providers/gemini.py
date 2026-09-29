"""Google Gemini provider definition."""

from .base import BaseProvider, ProviderPreset, register_provider


@register_provider
class GeminiProvider(BaseProvider):
    """Google Gemini multi-modal and fast inference provider."""

    preset = ProviderPreset(
        id="google",
        name="Gemini",
        protocol="openai_completion",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        require_api_key=True,
        model="gemini-3.1-flash-lite",
        model_options=[
            "gemini-3.1-flash-lite", "gemini-flash-latest", "gemini-pro-latest",
            "gemma-4-31b-it", "gemma-4-26b-a4b-it",
        ],
        thinking_level="medium",
        support_vision=True,
        vision_model="gemini-3.1-flash-lite",
        vision_model_options=["gemini-3.1-flash-lite", "gemini-flash-latest", "gemini-pro-latest"],
        vision_detail_level="auto",
        support_image=True,
        image_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        image_model="gemini-3.1-flash-lite-image",
        image_model_options=["gemini-3.1-flash-lite-image"],
        json_schema_response_format=True,
        aliases=["Google"],
    )
