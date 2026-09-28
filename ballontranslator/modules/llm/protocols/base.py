"""Abstract Base Protocol interface for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class LLMChatResult:
    """Standardized result returned by all LLM protocols."""
    content: str
    raw_response: Any = None
    token_usage: Dict[str, Any] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    reasoning_content: Optional[str] = None


@dataclass
class EncodedChatImage:
    """Standard container for an encoded image payload."""
    base64_data: str
    media_type: str = "image/jpeg"
    width: Optional[int] = None
    height: Optional[int] = None


class BaseLLMProtocol(ABC):
    """Abstract protocol for communicating with any LLM provider."""

    protocol_name: str = "base"

    def __init__(self, profile: Any, stop_event: Optional[Any] = None) -> None:
        self.profile = profile
        self.stop_event = stop_event

    @abstractmethod
    def send_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        thinking_level: str = "auto",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMChatResult:
        """Send chat/multimodal completion request and return normalized LLMChatResult."""
        pass

    @abstractmethod
    def format_vision_content(
        self,
        text_prompt: str,
        encoded_images: List[EncodedChatImage],
    ) -> Any:
        """Format multimodal images and text according to provider specifications."""
        pass

    def supports_thinking(self) -> bool:
        """Whether this protocol natively supports thinking / reasoning budgets."""
        return False

    def supports_native_schema(self) -> bool:
        """Whether this protocol supports strict JSON schema enforcement."""
        return False
