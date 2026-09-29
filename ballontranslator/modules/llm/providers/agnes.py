"""Agnes AI / Vyceai provider definition."""

from .base import BaseProvider, ProviderPreset, register_provider


@register_provider
class AgnesProvider(BaseProvider):
    """Agnes AI / Vyceai models provider with native vision and inpainting support."""

    preset = ProviderPreset(
        id="agnes",
        name="Agnes AI",
        protocol="openai_completion",
        base_url="https://vyceai.com/v1",
        require_api_key=True,
        model="agnes-3.0-flash",
        model_options=[
            "agnes-3.0-flash",
            "agnes-3.0-pro",
        ],
        support_vision=True,
        vision_model="agnes-3.0-flash",
        vision_model_options=[
            "agnes-3.0-flash",
            "agnes-3.0-pro",
        ],
        vision_detail_level="auto",
        support_image=True,
        image_base_url="https://vyceai.com/v1",
        image_model="agnes-image-2.0-flash",
        image_model_options=["agnes-image-2.0-flash"],
        aliases=["Vyceai", "agnes-ai", "Agnes"],
    )
