"""Anthropic Claude provider definition."""

from .base import BaseProvider, ProviderPreset, register_provider


@register_provider
class AnthropicProvider(BaseProvider):
    """Anthropic Claude models provider using Messages API."""

    preset = ProviderPreset(
        id="anthropic",
        name="Anthropic",
        protocol="anthropic_messages",
        base_url="https://api.anthropic.com/v1",
        require_api_key=True,
        model="claude-3-7-sonnet-latest",
        model_options=[
            "claude-3-7-sonnet-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
        ],
        support_vision=True,
        vision_model="claude-3-7-sonnet-latest",
        vision_model_options=[
            "claude-3-7-sonnet-latest",
            "claude-3-5-sonnet-latest",
        ],
        vision_detail_level="auto",
        support_image=False,
        aliases=["Claude"],
    )
