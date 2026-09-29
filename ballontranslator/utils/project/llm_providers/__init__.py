"""LLM Providers package exposing all provider presets and registry."""

from .base import (
    BaseProvider,
    ProviderPreset,
    get_all_provider_defaults,
    get_registered_providers,
    register_provider,
)
from .openai import OpenAIProvider
from .deepseek import DeepSeekProvider
from .gemini import GeminiProvider
from .anthropic import AnthropicProvider
from .agnes import AgnesProvider
from .openrouter import OpenRouterProvider
from .ollama import OllamaProvider
from .lmstudio import LMStudioProvider

PROVIDER_DEFAULTS = get_all_provider_defaults()

__all__ = [
    "BaseProvider",
    "ProviderPreset",
    "register_provider",
    "get_registered_providers",
    "get_all_provider_defaults",
    "PROVIDER_DEFAULTS",
    "OpenAIProvider",
    "DeepSeekProvider",
    "GeminiProvider",
    "AnthropicProvider",
    "AgnesProvider",
    "OpenRouterProvider",
    "OllamaProvider",
    "LMStudioProvider",
]
