"""LM Studio local server provider definition."""

from .base import BaseProvider, ProviderPreset, register_provider


@register_provider
class LMStudioProvider(BaseProvider):
    """Local LM Studio server provider."""

    preset = ProviderPreset(
        id="lmstudio",
        name="LM Studio",
        protocol="openai_completion",
        base_url="http://localhost:1234/v1",
        require_api_key=False,
        json_schema_response_format=True,
        model="local-model",
        model_options=["local-model"],
        support_vision=False,
        support_image=False,
    )
