"""Shared OpenAI-compatible Chat Completions request transport."""

from __future__ import annotations

import email.utils
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .context.errors import provider_error_message
from .context.token_usage import format_completion_token_usage
from .exceptions import (
    LLMApiKeyRequiredError,
    LLMOutputLimitError,
    LLMRequestStopped,
)
from ballontranslator.utils.project.llm_profiles import (
    LLMProfile,
    PROVIDER_DEFAULTS,
    THINKING_AUTO,
    THINKING_DISABLED,
    normalize_thinking_level,
    resolve_api_key,
)
from ballontranslator.utils.project.config import pcfg

TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _parse_retry_after(response: Any, default: float) -> float:
    """Parse HTTP Retry-After header as seconds or date, capped at 120s."""
    if response is None:
        return default
    headers = getattr(response, 'headers', None)
    if not headers:
        return default
    retry_header = headers.get('retry-after') or headers.get('Retry-After')
    if not retry_header:
        return default
    try:
        val = float(retry_header)
        return max(0.5, min(val, 120.0))
    except (ValueError, TypeError):
        try:
            target_time = email.utils.parsedate_to_datetime(retry_header).timestamp()
            diff = target_time - time.time()
            return max(0.5, min(diff, 120.0))
        except Exception:
            return default


OPENAI_MAX_TOKENS_MODELS = frozenset({
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
})
_CHAT_COMPLETIONS_SUFFIX = '/chat/completions'


def _normalized_base_url(url: str) -> str:
    return str(url or '').strip().rstrip('/')


def _openai_sdk_base_url(url: str) -> str:
    """Return the API base expected by the OpenAI SDK chat resource.

    Provider documentation often presents the complete HTTP endpoint, while
    ``client.chat.completions.create`` adds that endpoint path itself.

    >>> _openai_sdk_base_url('https://example.test/v1/chat/completions/')
    'https://example.test/v1'
    >>> _openai_sdk_base_url('https://example.test/v1')
    'https://example.test/v1'
    """
    base_url = str(url or '').strip()
    endpoint_url = base_url.rstrip('/')
    if endpoint_url.endswith(_CHAT_COMPLETIONS_SUFFIX):
        parent_url = endpoint_url[:-len(_CHAT_COMPLETIONS_SUFFIX)]
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


from .llm.protocols.openai_completion import build_chat_completion_args as openai_chat_completion_args


def openai_json_response_format(
    profile: LLMProfile,
    name: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the profile-compatible JSON response format.

    >>> profile = LLMProfile.from_provider('LM Studio')
    >>> openai_json_response_format(profile, 'demo', {'type': 'object'})['type']
    'json_schema'
    >>> profile.json_schema_response_format = False
    >>> openai_json_response_format(profile, 'demo', {})
    {'type': 'json_object'}
    """
    if not profile.json_schema_response_format:
        return {'type': 'json_object'}
    return {
        'type': 'json_schema',
        'json_schema': {
            'name': name,
            'strict': True,
            'schema': schema,
        },
    }


@dataclass(frozen=True)
class LLMChatResult:
    content: str
    usage: Any = None
    finish_reason: str = ''


class LLMChatRequestError(RuntimeError):
    """A normalized provider status error from Chat Completions."""

    def __init__(self, provider_error: Exception) -> None:
        self.provider_error = provider_error
        super().__init__(provider_error_message(provider_error))


class LLMChatRequester:
    """Issue one profile-backed OpenAI-compatible chat request.

    Prompt construction and retries stay with the owning Translator or OCR
    module; this boundary owns only transport and provider normalization.

    >>> LLMChatRequester().client is None
    True
    """

    dummy_api_key = 'dummy-key'

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client: Any = None
        self.client_cache_key: Optional[
            Tuple[str, Optional[str], str]
        ] = None
        self._clients_cache: Dict[Tuple[str, Optional[str], str], Any] = {}
        self._clients_lock = threading.Lock()
        self.last_request_time = 0.0
        self.request_count_minute = 0
        self.minute_start_time = time.time()
        self._rate_lock = threading.Lock()
        self.stop_event: Optional[threading.Event] = None

    def set_stop_event(
        self,
        stop_event: Optional[threading.Event],
    ) -> None:
        self.stop_event = stop_event

    def _wait(self, seconds: float) -> None:
        if seconds <= 0:
            return
        if self.stop_event is not None:
            if self.stop_event.wait(seconds):
                raise LLMRequestStopped()
            return
        time.sleep(seconds)

    @staticmethod
    def _openai_module() -> Any:
        import openai  # type: ignore

        return openai

    def _http_client(self, proxy: str) -> Any:
        import httpx  # type: ignore

        limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=50,
            keepalive_expiry=30.0,
        )
        if not proxy:
            return httpx.Client(limits=limits)
        try:
            mounts = {
                'http://': httpx.HTTPTransport(proxy=proxy, limits=limits),
                'https://': httpx.HTTPTransport(proxy=proxy, limits=limits),
            }
            return httpx.Client(mounts=mounts, limits=limits)
        except Exception as error:
            self.logger.error(
                f"Failed to initialize proxy '{proxy}': {error}. "
                'Proceeding without proxy.'
            )
            return httpx.Client(limits=limits)

    @staticmethod
    def _api_key_for_profile(profile: LLMProfile) -> str:
        from ballontranslator.utils.project.key_pool import resolve_active_api_key
        api_key = resolve_active_api_key(profile).strip()
        if profile.require_api_key and not api_key:
            raise LLMApiKeyRequiredError(profile.id, profile.name)
        return api_key

    def _client_api_key_for_profile(self, profile: LLMProfile) -> str:
        api_key = self._api_key_for_profile(profile)
        if not api_key:
            self.logger.debug(
                f'LLM profile "{profile.name or profile.id}" does not require '
                'an API key; using a dummy API key for OpenAI-compatible '
                'client initialization.'
            )
            return self.dummy_api_key
        return api_key

    def _initialize_client(self, profile: LLMProfile, api_key: Optional[str] = None) -> Any:
        if api_key is None:
            api_key = self._client_api_key_for_profile(profile)
        if not api_key:
            api_key = self.dummy_api_key
        configured_base_url = str(profile.base_url or '').strip()
        base_url = _openai_sdk_base_url(configured_base_url) or None
        proxy = str(self.get_param_value('proxy') or '') if hasattr(self, 'get_param_value') else ''
        cache_key = (api_key, base_url, proxy)

        if not hasattr(self, '_clients_cache'):
            self._clients_cache = {}
        if not hasattr(self, '_clients_lock'):
            self._clients_lock = threading.Lock()

        with self._clients_lock:
            if cache_key in self._clients_cache:
                self.client = self._clients_cache[cache_key]
                self.client_cache_key = cache_key
                return self.client

            if configured_base_url and base_url != configured_base_url:
                self.logger.warning(
                    f'LLM profile "{profile.name or profile.id}" Base URL ends '
                    f'with {_CHAT_COMPLETIONS_SUFFIX}; using its parent URL as '
                    'the OpenAI-compatible API base.'
                )

            openai = self._openai_module()
            client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=self._http_client(proxy),
            )
            self._clients_cache[cache_key] = client
            self.client = client
            self.client_cache_key = cache_key
            return client

    def _respect_delay(self, profile: Optional[LLMProfile] = None) -> None:
        with self._rate_lock:
            current_time = time.time()
            profile_rpm = getattr(profile, 'rpm_limit', 0) if profile else 0
            if profile_rpm > 0:
                rpm = profile_rpm
            elif hasattr(self, 'get_param_value'):
                rpm = self.get_param_value('max requests per minute')
            else:
                rpm = 0

            profile_delay = getattr(profile, 'request_delay', 0.0) if profile else 0.0
            global_delay = getattr(pcfg.module, 'llm_request_delay', 0.0)
            if profile_delay > 0:
                delay = profile_delay
            elif global_delay > 0:
                delay = global_delay
            elif hasattr(self, 'get_param_value'):
                delay = self.get_param_value('delay')
            else:
                delay = 0.0

            if rpm > 0:
                if current_time - self.minute_start_time >= 60:
                    self.request_count_minute = 0
                    self.minute_start_time = current_time
                if self.request_count_minute >= rpm:
                    wait_time = 60.1 - (current_time - self.minute_start_time)
                    if wait_time > 0:
                        self.logger.warning(
                            f'RPM limit ({rpm}) reached. Waiting '
                            f'{wait_time:.2f} seconds.'
                        )
                        self._wait(wait_time)
                    self.request_count_minute = 0
                    self.minute_start_time = time.time()

            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < delay:
                self._wait(delay - time_since_last_request)

            self.last_request_time = time.time()
            self.request_count_minute += 1

    def request_chat_completion(
        self,
        profile: LLMProfile,
        api_args: Dict[str, Any],
    ) -> LLMChatResult:
        """Perform one request; feature owners decide whether to retry it."""
        if self.stop_event is not None and self.stop_event.is_set():
            raise LLMRequestStopped()
        if getattr(profile, 'protocol', 'openai_completion') == 'anthropic_messages':
            from ballontranslator.modules.llm.protocols.anthropic_messages import AnthropicMessagesProtocol
            protocol_handler = AnthropicMessagesProtocol(profile, stop_event=self.stop_event)
            self._respect_delay(profile)
            proxy = str(self.get_param_value('proxy') or '') if hasattr(self, 'get_param_value') else ''
            configured_timeout = None
            if hasattr(self, 'get_param_value'):
                try:
                    val = self.get_param_value('request timeout')
                    if val is not None:
                        configured_timeout = float(val)
                except Exception:
                    pass
            timeout = configured_timeout if (configured_timeout and configured_timeout > 0) else 180.0

            res = protocol_handler.send_chat_completion(
                messages=api_args.get('messages', []),
                max_tokens=api_args.get('max_tokens') or api_args.get('max_completion_tokens'),
                model=api_args.get('model'),
                proxy=proxy,
                timeout=timeout,
                api_key=self._client_api_key_for_profile(profile),
            )
            return LLMChatResult(
                content=res.content,
                usage=res.token_usage,
                finish_reason=res.finish_reason or 'stop',
            )

        openai = self._openai_module()
        active_key = self._client_api_key_for_profile(profile)
        client = self._initialize_client(profile, api_key=active_key)
        call_args = dict(api_args)
        if 'timeout' not in call_args:
            configured_timeout = None
            if hasattr(self, 'get_param_value'):
                try:
                    val = self.get_param_value('request timeout')
                    if val is not None:
                        configured_timeout = float(val)
                except Exception:
                    pass
            call_args['timeout'] = configured_timeout if (configured_timeout and configured_timeout > 0) else 180.0

        max_transient_retries = 3
        base_backoff = 2.0
        completion = None

        for attempt in range(max_transient_retries + 1):
            if self.stop_event is not None and self.stop_event.is_set():
                raise LLMRequestStopped()
            self._respect_delay(profile)
            if self.stop_event is not None and self.stop_event.is_set():
                raise LLMRequestStopped()

            try:
                completion = client.chat.completions.create(**call_args)
                break
            except getattr(openai, 'AuthenticationError') as error:
                from ballontranslator.utils.project.key_pool import is_key_file, get_key_pool
                raw_configured = resolve_api_key(profile).strip()
                if is_key_file(raw_configured):
                    pool = get_key_pool(raw_configured)
                    if pool.count() > 1:
                        pool.mark_rate_limited(active_key, cooldown_seconds=3600)
                        next_key = pool.get_next_key()
                        if next_key and next_key != active_key:
                            self.logger.warning(
                                f"Authentication failed for key in Keys File ({profile.name}). "
                                f"Switching to next key in pool..."
                            )
                            active_key = next_key
                            client = self._initialize_client(profile, api_key=active_key)
                            continue
                raise LLMApiKeyRequiredError(
                    profile.id, profile.name
                ) from error
            except Exception as error:
                status_error_type = getattr(openai, 'APIStatusError', ())
                connection_error_type = getattr(openai, 'APIConnectionError', ())
                timeout_error_type = getattr(openai, 'APITimeoutError', ())

                is_status_error = isinstance(error, status_error_type)
                status_code = getattr(error, 'status_code', None) if is_status_error else None
                is_connection_error = isinstance(error, connection_error_type)
                is_timeout_error = isinstance(error, timeout_error_type)

                is_transient = (
                    (is_status_error and status_code in TRANSIENT_STATUS_CODES)
                    or is_connection_error
                    or is_timeout_error
                )

                from ballontranslator.utils.project.key_pool import is_key_file, get_key_pool
                raw_configured = resolve_api_key(profile).strip()
                is_rate_limit = (
                    status_code == 429
                    or '429' in str(error)
                    or 'ResourceExhausted' in str(error)
                    or 'rate limit' in str(error).lower()
                )
                if is_rate_limit and is_key_file(raw_configured):
                    pool = get_key_pool(raw_configured)
                    if pool.count() > 1:
                        pool.mark_rate_limited(active_key, cooldown_seconds=60)
                        next_key = pool.get_next_key()
                        if next_key and next_key != active_key:
                            self.logger.warning(
                                f"Rate limit (HTTP 429) on key in Keys File ({profile.name}). "
                                f"Switching to next available key in pool..."
                            )
                            active_key = next_key
                            client = self._initialize_client(profile, api_key=active_key)
                            continue

                if is_transient and attempt < max_transient_retries:
                    response = getattr(error, 'response', None)
                    jitter = random.uniform(0.2, 1.0)
                    default_wait = min(60.0, (base_backoff ** attempt) * 2.0 + jitter)
                    wait_time = _parse_retry_after(response, default_wait)
                    status_desc = f"HTTP {status_code}" if status_code else type(error).__name__
                    self.logger.warning(
                        f"Transient error ({status_desc}) from LLM provider. "
                        f"Retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_transient_retries})..."
                    )
                    self._wait(wait_time)
                    continue

                if is_status_error:
                    raise LLMChatRequestError(error) from error
                raise

        if self.stop_event is not None and self.stop_event.is_set():
            raise LLMRequestStopped()

        choice = next(iter(getattr(completion, 'choices', ())), None)
        message = getattr(choice, 'message', None)
        content = getattr(message, 'content', None)
        if content is None:
            content = getattr(choice, 'text', '')
        result = LLMChatResult(
            content=str(content or ''),
            usage=getattr(completion, 'usage', None),
            finish_reason=str(
                getattr(choice, 'finish_reason', '') or ''
            ),
        )
        if result.finish_reason.strip().lower() == 'length':
            usage = format_completion_token_usage(result)
            model = str(api_args.get('model', '')).replace(
                '\r', ' '
            ).replace('\n', ' ')
            details = ', '.join(
                part for part in (
                    f'profile_id={profile.id!r}',
                    f'model={model!r}',
                    usage,
                    'finish_reason=length',
                    f'chars={len(result.content)}',
                    f'content={result.content!r}',
                )
                if part
            )
            self.logger.debug(f'LLM output-limited response: {details}')
            raise LLMOutputLimitError(
                profile.id,
                profile.name,
                profile.max_tokens,
                str(profile.thinking_level or THINKING_AUTO),
            )
        return result


def _repair_json_text(text: str) -> str:
    """Auto-repair broken JSON by balancing unclosed braces/brackets, stripping trailing commas, and closing unclosed strings."""
    if not text:
        return text

    # Remove trailing commas before closing brackets
    repaired = re.sub(r',\s*([\]}])', r'\1', text)
    repaired = re.sub(r',\s*$', '', repaired.rstrip())

    stack = []
    in_string = False
    escape = False

    for ch in repaired:
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch in '{[':
                stack.append('}' if ch == '{' else ']')
            elif ch in '}]':
                if stack and stack[-1] == ch:
                    stack.pop()

    if in_string:
        repaired += '"'

    repaired = re.sub(r',\s*$', '', repaired.rstrip())

    if stack:
        repaired += ''.join(reversed(stack))

    repaired = re.sub(r',\s*([\]}])', r'\1', repaired)
    return repaired


def _clean_json_text(raw_text: str) -> str:
    """Extract and clean valid JSON substring from LLM response."""
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*([{\[].*?[}\]])\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        if text.startswith('```'):
            lines = text.splitlines()
            if lines and lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].startswith('```'):
                lines = lines[:-1]
            text = '\n'.join(lines).strip()

        first_brace = text.find('{')
        first_bracket = text.find('[')
        start_idx = -1
        if first_brace != -1 and first_bracket != -1:
            start_idx = min(first_brace, first_bracket)
        elif first_brace != -1:
            start_idx = first_brace
        elif first_bracket != -1:
            start_idx = first_bracket

        if start_idx != -1:
            text = text[start_idx:]

        last_brace = text.rfind('}')
        last_bracket = text.rfind(']')
        end_idx = -1
        if last_brace != -1 and last_bracket != -1:
            end_idx = max(last_brace, last_bracket)
        elif last_brace != -1:
            end_idx = last_brace
        elif last_bracket != -1:
            end_idx = last_bracket

        if end_idx != -1:
            text = text[:end_idx + 1]

    return text

