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

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Convert OpenAI-style or raw messages into Anthropic system + messages list."""
        system_text: Optional[str] = None
        anthropic_messages: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    system_text = (f"{system_text}\n{content}" if system_text else content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            txt = part.get("text", "")
                            system_text = (f"{system_text}\n{txt}" if system_text else txt)
                continue

            # Convert multimodal content
            if isinstance(content, list):
                converted_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            converted_parts.append({"type": "text", "text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            url_obj = part.get("image_url", {})
                            url = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj)
                            if url.startswith("data:") and ";base64," in url:
                                header, data = url.split(";base64,", 1)
                                media_type = header[5:] or "image/jpeg"
                                converted_parts.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": data,
                                    },
                                })
                anthropic_messages.append({"role": role, "content": converted_parts})
            else:
                anthropic_messages.append({"role": role, "content": str(content)})

        return system_text, anthropic_messages

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
        import httpx
        from ballontranslator.modules.exceptions import (
            LLMApiKeyRequiredError,
            LLMRequestStopped,
        )
        from ballontranslator.utils.project.llm_profiles import resolve_api_key

        if self.stop_event is not None and self.stop_event.is_set():
            raise LLMRequestStopped()

        api_key = str(kwargs.get("api_key") or resolve_api_key(self.profile) or "").strip()
        if getattr(self.profile, "require_api_key", True) and not api_key:
            raise LLMApiKeyRequiredError(
                getattr(self.profile, "id", ""),
                getattr(self.profile, "name", ""),
            )

        base_url = str(getattr(self.profile, "base_url", "") or "https://api.anthropic.com/v1").strip().rstrip("/")
        endpoint = base_url if base_url.endswith("/messages") else f"{base_url}/messages"

        sys_from_msgs, converted_messages = self._convert_messages(messages)
        final_system = (
            f"{system_prompt}\n{sys_from_msgs}"
            if (system_prompt and sys_from_msgs)
            else (system_prompt or sys_from_msgs)
        )

        payload = self.build_payload(converted_messages, system_prompt=final_system)
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        if kwargs.get("model"):
            payload["model"] = str(kwargs["model"])

        headers = self.build_headers(api_key)
        timeout = float(kwargs.get("timeout") or getattr(self.profile, "request_timeout", 180.0) or 180.0)
        proxy = str(kwargs.get("proxy") or "")

        client_kwargs: Dict[str, Any] = {"timeout": timeout}
        if proxy:
            client_kwargs["mounts"] = {
                "http://": httpx.HTTPTransport(proxy=proxy),
                "https://": httpx.HTTPTransport(proxy=proxy),
            }

        with httpx.Client(**client_kwargs) as client:
            resp = client.post(endpoint, headers=headers, json=payload)

        if resp.status_code == 401:
            raise LLMApiKeyRequiredError(
                getattr(self.profile, "id", ""),
                getattr(self.profile, "name", ""),
            )
        resp.raise_for_status()
        data = resp.json()

        text_content = ""
        reasoning_content = ""
        for block in data.get("content", []):
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_content += block.get("text", "")
                elif block.get("type") == "thinking":
                    reasoning_content += block.get("thinking", "")

        usage_raw = data.get("usage", {})
        usage = {
            "prompt_tokens": usage_raw.get("input_tokens", 0),
            "completion_tokens": usage_raw.get("output_tokens", 0),
            "total_tokens": usage_raw.get("input_tokens", 0) + usage_raw.get("output_tokens", 0),
        }

        return LLMChatResult(
            content=text_content,
            raw_response=data,
            token_usage=usage,
            finish_reason=data.get("stop_reason"),
            reasoning_content=reasoning_content or None,
        )
