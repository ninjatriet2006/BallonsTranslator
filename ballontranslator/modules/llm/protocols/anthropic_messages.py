"""Anthropic Messages API Protocol Adapter (/v1/messages)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseLLMProtocol, EncodedChatImage, LLMChatResult


class AnthropicMessagesProtocol(BaseLLMProtocol):
    """Protocol Adapter for Anthropic Claude Messages API (/v1/messages)."""

    protocol_name: str = "anthropic_messages"

    def format_vision_content(
        self,
        text_prompt: str,
        encoded_images: List[EncodedChatImage],
    ) -> List[Dict[str, Any]]:
        """Format text and images using Anthropic base64 source blocks."""
        content: List[Dict[str, Any]] = []
        for img in encoded_images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.media_type,
                    "data": img.base64_data,
                }
            })
        content.append({"type": "text", "text": text_prompt})
        return content

    def build_headers(self, api_key: str, enable_prompt_caching: bool = True) -> Dict[str, str]:
        """Build standard headers for Anthropic API."""
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if enable_prompt_caching:
            headers["anthropic-beta"] = "prompt-caching-2024-07-31"
        return headers

    def build_payload(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        thinking_budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Construct Anthropic payload with top-level system prompt and thinking config."""
        payload: Dict[str, Any] = {
            "model": getattr(self.profile, 'model', 'claude-3-5-sonnet-20241022'),
            "messages": messages,
            "max_tokens": int(getattr(self.profile, 'max_tokens', 4096)),
        }
        if system_prompt:
            payload["system"] = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        if thinking_budget and thinking_budget > 0:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            # Max tokens must exceed thinking budget
            payload["max_tokens"] = max(payload["max_tokens"], thinking_budget + 1024)
        else:
            payload["temperature"] = float(getattr(self.profile, 'temperature', 0.7))

        return payload

    def supports_thinking(self) -> bool:
        return True

    def supports_native_schema(self) -> bool:
        return False

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
        raise NotImplementedError("Anthropic direct HTTP execution to be hooked in Phase 5.")
