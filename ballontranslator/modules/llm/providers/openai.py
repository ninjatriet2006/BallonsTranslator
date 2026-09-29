"""OpenAI provider definition."""

from .base import BaseProvider, ProviderPreset, register_provider


@register_provider
class OpenAIProvider(BaseProvider):
    """OpenAI flagship models provider."""

    preset = ProviderPreset(
        id="openai",
        name="OpenAI",
        protocol="openai_completion",
        base_url="https://api.openai.com/v1",
        require_api_key=True,
        model="gpt-5.5",
        model_options=[
            "gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
            "gpt-5.5", "gpt-5.4", "gpt-5.4-mini",
            "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini",
        ],
        support_vision=True,
        vision_model="gpt-5.5",
        vision_model_options=[
            "gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
            "gpt-5.5", "gpt-5.4", "gpt-5.4-mini",
            "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini",
        ],
        vision_detail_level="auto",
        support_image=True,
        image_base_url="https://api.openai.com/v1/images/edits",
        image_model="gpt-image-2",
        image_model_options=["gpt-image-2"],
    )
