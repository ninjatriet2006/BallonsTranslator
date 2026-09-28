"""OpenAI Responses API Protocol Adapter (/v1/responses)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseLLMProtocol, EncodedChatImage, LLMChatResult


class OpenAIResponseProtocol(BaseLLMProtocol):
    """Protocol Adapter for OpenAI Responses API (/v1/responses)."""

    protocol_name: str = "openai_response"

    def format_vision_content(
        self,
        text_prompt: str,
        encoded_images: List[EncodedChatImage],
    ) -> List[Dict[str, Any]]:
        """Format input modalities according to OpenAI Responses specifications."""
        items: List[Dict[str, Any]] = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text_prompt}]}
        ]
        for img in encoded_images:
            items[0]["content"].append({
                "type": "input_image",
                "image_url": f"data:{img.media_type};base64,{img.base64_data}",
            })
        return items

    def build_payload(
        self,
        input_items: List[Dict[str, Any]],
        instructions: Optional[str] = None,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Construct Responses API payload with instructions and schema."""
        payload: Dict[str, Any] = {
            "model": getattr(self.profile, 'model', 'gpt-4o'),
            "input": input_items,
        }
        if instructions:
            payload["instructions"] = instructions
        if response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": response_schema,
            }
        return payload

    def supports_thinking(self) -> bool:
        return True

    def supports_native_schema(self) -> bool:
        return True

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
        raise NotImplementedError("OpenAI Responses direct execution to be hooked in Phase 5.")
