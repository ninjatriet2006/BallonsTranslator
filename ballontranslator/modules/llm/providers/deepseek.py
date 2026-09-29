"""DeepSeek provider definition."""

from .base import BaseProvider, ProviderPreset, register_provider


@register_provider
class DeepSeekProvider(BaseProvider):
    """DeepSeek chat and reasoning models provider."""

    preset = ProviderPreset(
        id="deepseek",
        name="DeepSeek",
        protocol="openai_completion",
        base_url="https://api.deepseek.com",
        require_api_key=True,
        model="deepseek-v4-flash",
        model_options=["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"],
        support_text=True,
        support_vision=False,
        support_image=False,
    )
