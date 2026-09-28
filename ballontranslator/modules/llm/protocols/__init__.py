"""LLM Protocols Package."""

from typing import Any
from .base import BaseLLMProtocol, EncodedChatImage, LLMChatResult
from .openai_completion import OpenAICompletionProtocol, build_chat_completion_args
from .anthropic_messages import AnthropicMessagesProtocol
from .openai_response import OpenAIResponseProtocol

__all__ = [
    "BaseLLMProtocol",
    "EncodedChatImage",
    "LLMChatResult",
    "OpenAICompletionProtocol",
    "build_chat_completion_args",
    "AnthropicMessagesProtocol",
    "OpenAIResponseProtocol",
    "create_llm_protocol",
]


def create_llm_protocol(profile: Any, protocol_name: str = "openai_completion") -> BaseLLMProtocol:
    """Factory to instantiate the appropriate protocol adapter for an LLMProfile."""
    protocol_lower = str(protocol_name or "openai_completion").lower()
    if protocol_lower == "anthropic_messages":
        return AnthropicMessagesProtocol(profile)
    elif protocol_lower == "openai_response":
        return OpenAIResponseProtocol(profile)
    else:
        return OpenAICompletionProtocol(profile)
