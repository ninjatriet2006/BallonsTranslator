"""OpenAI Chat Completions Protocol Adapter (/v1/chat/completions)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ballontranslator.utils.project.llm_profiles import (
    LLMProfile,
    PROVIDER_DEFAULTS,
    THINKING_AUTO,
    THINKING_DISABLED,
    normalize_thinking_level,
)
from .base import BaseLLMProtocol, EncodedChatImage, LLMChatResult

OPENAI_MAX_TOKENS_MODELS = frozenset({
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
})


def _normalized_base_url(url: str) -> str:
    return str(url or '').strip().rstrip('/')


def _openai_sdk_base_url(url: str) -> str:
    base_url = str(url or '').strip()
    endpoint_url = base_url.rstrip('/')
    suffix = '/chat/completions'
    if endpoint_url.endswith(suffix):
        parent_url = endpoint_url[:-len(suffix)]
        if parent_url:
            return parent_url
    return base_url


def _uses_provider_base_url(base_url: str, provider: str) -> bool:
    provider_url = _normalized_base_url(
        PROVIDER_DEFAULTS[provider]['base_url']
    ).lower()
    base_url = _normalized_base_url(base_url).lower()
    return base_url == provider_url or base_url.startswith(
        f'{provider_url}/'
    )


def build_chat_completion_args(
    profile: LLMProfile,
    model: str,
) -> Dict[str, Any]:
    """Map profile values to standard OpenAI /v1/chat/completions arguments."""
    base_url = _normalized_base_url(_openai_sdk_base_url(profile.base_url))
    openai_base_url = _normalized_base_url(
        PROVIDER_DEFAULTS['OpenAI']['base_url']
    )
    model_name = str(model or '').rsplit('/', 1)[-1].lower()
    is_native_openai = not base_url or base_url == openai_base_url
    args: Dict[str, Any] = {'top_p': float(profile.top_p)}

    version_match = re.match(r'^gpt-(\d+)(?:\.(\d+))?(?:-|$)', model_name)
    gpt_version = (
        (int(version_match.group(1)), int(version_match.group(2) or 0))
        if version_match
        else None
    )
    if not is_native_openai or gpt_version is None or gpt_version < (5, 5):
        args['temperature'] = float(profile.temperature)

    token_limit_key = (
        'max_completion_tokens'
        if is_native_openai and model_name not in OPENAI_MAX_TOKENS_MODELS
        else 'max_tokens'
    )
    args[token_limit_key] = int(profile.max_tokens)

    thinking_level = normalize_thinking_level(profile.thinking_level)
    if thinking_level == THINKING_AUTO:
        return args
    if thinking_level != THINKING_DISABLED:
        args['reasoning_effort'] = thinking_level
        return args
    if _uses_provider_base_url(base_url, 'DeepSeek'):
        args['extra_body'] = {'thinking': {'type': 'disabled'}}
    elif _uses_provider_base_url(base_url, 'OpenRouter'):
        args['extra_body'] = {'reasoning': {'effort': 'none'}}
    else:
        args['reasoning_effort'] = 'none'
    return args


class OpenAICompletionProtocol(BaseLLMProtocol):
    """Protocol Adapter for OpenAI-compatible Chat Completions (/v1/chat/completions)."""

    protocol_name: str = "openai_completion"

    def format_vision_content(
        self,
        text_prompt: str,
        encoded_images: List[EncodedChatImage],
    ) -> List[Dict[str, Any]]:
        """Format text and images using OpenAI image_url standard."""
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": text_prompt}
        ]
        for img in encoded_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img.media_type};base64,{img.base64_data}"
                }
            })
        return content

    def supports_thinking(self) -> bool:
        return True

    def supports_native_schema(self) -> bool:
        return bool(getattr(self.profile, 'json_schema_response_format', True))

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
        """Executed through requester delegator or direct client call."""
        raise NotImplementedError("Direct invocation delegated through requester client.")
