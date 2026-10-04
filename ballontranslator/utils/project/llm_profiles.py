from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    get_args,
    get_origin,
    get_type_hints,
)

from ballontranslator.utils.system.logger import logger as LOGGER
from ballontranslator.utils.system.secret_store import SecretStore
from ballontranslator.utils.project.structures import Config, field, nested_dataclass


LLM_TRANSLATOR_KEY = "LLMTranslator"
LLM_OCR_KEY = "LLMOCR"
LLM_INPAINT_KEY = "LLMInpaint"
LLM_DETECTOR_KEY = "llm_detector"
OLD_LLM_TRANSLATORS = ("ChatGPT", "ChatGPT_exp", "LLM_API_Translator")

THINKING_AUTO = "Auto"
THINKING_DISABLED = "Disabled"
THINKING_LEVEL_OPTIONS = [
    THINKING_AUTO,
    THINKING_DISABLED,
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
]
VISION_DETAIL_LEVEL_OPTIONS = ["None", "auto", "low", "high"]

PROTOCOL_OPENAI_COMPLETION = "openai_completion"
PROTOCOL_OPENAI_RESPONSE = "openai_response"
PROTOCOL_ANTHROPIC_MESSAGES = "anthropic_messages"

PROTOCOL_OPTIONS = [
    PROTOCOL_OPENAI_COMPLETION,
    PROTOCOL_OPENAI_RESPONSE,
    PROTOCOL_ANTHROPIC_MESSAGES,
]

PROTOCOL_DISPLAY_NAMES = {
    PROTOCOL_OPENAI_COMPLETION: "OpenAI Chat Completions",
    PROTOCOL_OPENAI_RESPONSE: "OpenAI Responses API",
    PROTOCOL_ANTHROPIC_MESSAGES: "Anthropic Messages API",
}

PROVIDER_ALIASES = {
    "Google": "Gemini",
    "Vyceai": "Agnes AI",
    "Agnes": "Agnes AI",
    "agnes-ai": "Agnes AI",
    "Claude": "Anthropic",
}

from ballontranslator.utils.project.llm_providers import (
    PROVIDER_DEFAULTS,
    get_registered_providers,
)

DEFAULT_TRANSLATION_PROMPT = (
    "Translate faithfully and fluently into authentic, natural comic dialogue suited for speech bubbles. "
    "Preserve the original meaning, character voices, emotional nuance, and slang. "
    "Keep sentence length concise and readable. Maintain all formatting, punctuation (ellipses, exclamation marks), "
    "casing, and line breaks where appropriate. Keep names, honorifics, and terminology strictly consistent. "
    "Do not add translator notes, explanations, or quotes."
)

DEFAULT_OCR_PROMPT = (
    "Extract all visible text verbatim from this comic or manga image. "
    "Correctly infer reading orientation (vertical right-to-left, or horizontal left-to-right). "
    "Transcribe exact characters faithfully without translating, summarizing, or omitting text. "
    "If furigana (ruby text) is present, transcribe only the main characters without duplicating ruby. "
    "Return only the recognized text. If no text is visible, return an empty response."
)

DEFAULT_INPAINT_PROMPT = (
    "Remove all text elements from this image, including dialogue, narration, captions, credits, titles, and author notes.\n\n"
    "Strict Inpainting Rules:\n"
    "1. Flat Background Priority: If the text is situated on a solid, empty, or plain background (such as pure white #FFFFFF), replace the erased text ONLY with the identical flat solid background color. Strictly DO NOT hallucinate, invent, or render any new artwork, character faces, line art, textures, or screentone patterns in these empty spaces.\n"
    "2. Artwork Reconstruction: Only reconstruct underlying linework, screentones, and textures when text directly overlaps existing illustrations or panel borders. Match the original line weight and tone seamlessly.\n"
    "3. Preserve Existing Art: Strictly keep all original character illustrations, panel borders, ink lines, and existing white margins 100% intact. Do not erase, blur, or alter any non-text elements.\n"
    "4. Output: A clean scanlation-ready page with sharp linework and pristine, empty speech/text areas."
)

DEFAULT_DETECTOR_BOX_PROMPT = (
    "Detect all text clusters including speech balloons, narration boxes, thought bubbles, "
    "sound effects (SFX/onomatopoeia), titles, and side text. Bounding boxes must enclose the "
    "full text and balloon area with a comfortable margin; strictly do not clip or truncate the "
    "outer edges of speech bubbles or text strokes."
)

DEFAULT_DETECTOR_OCR_PROMPT = (
    "Transcribe all visible text inside the box verbatim without translating, summarizing, "
    "or omitting. If furigana (ruby text) is present, transcribe only the base characters. "
    "Preserve original line breaks with \\n."
)

def _normal_url(url: str) -> str:
    url = (url or "").strip()
    if url == "http://localhost:11434/v1/":
        return url
    return url.rstrip("/")


def _strip_model_prefix(model: str) -> str:
    model = (model or "").strip()
    if ": " in model:
        model = model.split(": ", 1)[1]
    if model == "(override model field)":
        model = ""
    return model


@nested_dataclass
class LLMProfile(Config):
    """Typed persistent LLM profile config.

    Example:
        >>> LLMProfile.from_provider('DeepSeek').base_url
        'https://api.deepseek.com'
    """

    id: str = ""
    profile_type = "llm"
    name: str = ""
    built_in: bool = False
    base_url: str = ""
    api_key: Any = ""
    require_api_key: bool = True
    model: str = ""
    model_options: List[str] = field(default_factory=list)
    support_text: bool = True
    support_vision: bool = False
    vision_model: str = ""
    vision_model_options: List[str] = field(default_factory=list)
    vision_detail_level: str = "None"
    vision_detail_level_options: List[str] = field(default_factory=lambda: list(VISION_DETAIL_LEVEL_OPTIONS))
    support_image: bool = False
    image_base_url: str = ""
    image_model: str = ""
    image_model_options: List[str] = field(default_factory=list)
    protocol: str = PROTOCOL_OPENAI_COMPLETION
    protocol_options: List[str] = field(default_factory=lambda: list(PROTOCOL_OPTIONS))
    thinking_level: str = THINKING_AUTO
    thinking_level_options: List[str] = field(default_factory=lambda: list(THINKING_LEVEL_OPTIONS))
    prompt: str = DEFAULT_TRANSLATION_PROMPT
    vision_prompt: str = DEFAULT_OCR_PROMPT
    detector_box_prompt: str = DEFAULT_DETECTOR_BOX_PROMPT
    detector_ocr_prompt: str = DEFAULT_DETECTOR_OCR_PROMPT
    image_prompt: str = DEFAULT_INPAINT_PROMPT
    max_tokens: int = 8192
    temperature: float = 0.1
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    json_schema_response_format: bool = False
    low_vram_mode: bool = False
    concurrency: int = 0
    request_delay: float = 0.0
    rpm_limit: int = 0
    key_cooldown: float = 0.0
    max_retry_wait: float = 0.0

    @classmethod
    def from_provider(cls, provider: str) -> "LLMProfile":
        provider = canonical_provider(provider)
        info = copy.deepcopy(PROVIDER_DEFAULTS[provider])
        return cls(**info, name=provider, built_in=True)

    def to_dict(self) -> Dict:
        return copy.deepcopy(self.__dict__)


def normalize_thinking_level(value: Any) -> str:
    """Return the canonical persisted reasoning-control value.

    >>> normalize_thinking_level('None')
    'Auto'
    >>> normalize_thinking_level('Disabled')
    'Disabled'
    """
    if not isinstance(value, str):
        return THINKING_AUTO
    level = value.strip()
    if level.lower() in {'', 'none', 'auto'}:
        return THINKING_AUTO
    if level.lower() == 'disabled':
        return THINKING_DISABLED
    return level


def _normalize_profile_thinking(profile: LLMProfile) -> LLMProfile:
    """Migrate legacy reasoning settings without dropping custom options.

    >>> profile = LLMProfile(
    ...     thinking_level='None',
    ...     thinking_level_options=['None', 'high'],
    ... )
    >>> migrated = _normalize_profile_thinking(profile)
    >>> migrated.thinking_level, migrated.thinking_level_options
    ('Auto', ['Auto', 'Disabled', 'high'])
    """
    raw_level = profile.thinking_level
    if not isinstance(raw_level, str):
        LOGGER.warning(
            'Discard invalid LLM profile thinking_level for %s.',
            profile.id or profile.name or '<unnamed>',
        )
    profile.thinking_level = normalize_thinking_level(raw_level)

    raw_options = profile.thinking_level_options
    if not isinstance(raw_options, list):
        LOGGER.warning(
            'Discard invalid LLM profile thinking_level_options for %s.',
            profile.id or profile.name or '<unnamed>',
        )
        raw_options = THINKING_LEVEL_OPTIONS
    elif any(not isinstance(option, str) for option in raw_options):
        LOGGER.warning(
            'Discard invalid entries from LLM profile '
            'thinking_level_options for %s.',
            profile.id or profile.name or '<unnamed>',
        )

    normalized_options = []
    for option in [THINKING_AUTO, THINKING_DISABLED, *raw_options]:
        if not isinstance(option, str):
            continue
        option = normalize_thinking_level(option)
        if option not in normalized_options:
            normalized_options.append(option)
    if profile.thinking_level not in normalized_options:
        normalized_options.append(profile.thinking_level)
    profile.thinking_level_options = normalized_options

    try:
        profile.concurrency = max(0, int(getattr(profile, 'concurrency', 0) or 0))
    except Exception:
        profile.concurrency = 0
    try:
        profile.request_delay = max(0.0, min(60.0, float(getattr(profile, 'request_delay', 0.0) or 0.0)))
    except Exception:
        profile.request_delay = 0.0
    try:
        profile.rpm_limit = max(0, int(getattr(profile, 'rpm_limit', 0) or 0))
    except Exception:
        profile.rpm_limit = 0

    raw_protocol = getattr(profile, 'protocol', None)
    if not isinstance(raw_protocol, str) or raw_protocol not in PROTOCOL_OPTIONS:
        profile.protocol = PROTOCOL_OPENAI_COMPLETION
    raw_proto_options = getattr(profile, 'protocol_options', None)
    if not isinstance(raw_proto_options, list) or not raw_proto_options:
        profile.protocol_options = list(PROTOCOL_OPTIONS)
    else:
        norm_proto_options = [opt for opt in PROTOCOL_OPTIONS if opt in raw_proto_options]
        for opt in PROTOCOL_OPTIONS:
            if opt not in norm_proto_options:
                norm_proto_options.append(opt)
        profile.protocol_options = norm_proto_options
    return profile


def profile_from_config(profile: Any) -> LLMProfile:
    if isinstance(profile, LLMProfile):
        loaded = copy.deepcopy(profile)
    elif isinstance(profile, Mapping):
        loaded = LLMProfile(**copy.deepcopy(dict(profile)))
    else:
        raise TypeError(f"Unsupported LLM profile config: {type(profile)!r}")
    return _normalize_profile_thinking(loaded)


def profile_to_dict(profile: Any) -> Dict:
    return profile_from_config(profile).to_dict()


def profile_to_export_dict(profile: Any) -> Dict:
    """Return the JSON mapping used by profile clipboard export.

    Clipboard export resolves the API key to plaintext; normal config saving
    continues to apply portable obfuscation independently.

    Example:
        >>> profile_to_export_dict(LLMProfile())['profile_type']
        'llm'
    """

    exported = profile_to_dict(profile)
    exported['profile_type'] = LLMProfile.profile_type
    exported['api_key'] = resolve_api_key(profile)
    return exported


def _normalize_import_profile_data(data: Mapping) -> Dict[str, Any]:
    """Normalize imported values while letting omitted fields use profile defaults."""

    type_hints = get_type_hints(LLMProfile)
    normalized = {}
    for key, expected in type_hints.items():
        origin = get_origin(expected)
        args = get_args(expected)
        if origin is list:
            value = data.get(key)
            item_type = args[0] if args else Any
            normalized[key] = (
                value if isinstance(value, list)
                and (item_type is Any or all(isinstance(item, item_type) for item in value))
                else []
            )
            continue
        if key not in data:
            continue
        value = data[key]
        expected = type_hints.get(key)
        if key == 'api_key':
            if isinstance(value, str):
                normalized[key] = value
            continue
        if expected is float:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                normalized[key] = float(value)
        elif expected is int:
            if isinstance(value, int) and not isinstance(value, bool):
                normalized[key] = value
        elif isinstance(value, expected):
            normalized[key] = value
    return normalized


def profiles_from_json(value: str) -> List[LLMProfile]:
    """Parse one or more valid LLM profiles from clipboard JSON.

    Example:
        >>> len(profiles_from_json(json.dumps(profile_to_export_dict(LLMProfile()))))
        1
    """

    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return []
    candidates = [decoded] if isinstance(decoded, Mapping) else decoded if isinstance(decoded, list) else []
    profiles = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or candidate.get('profile_type') != LLMProfile.profile_type:
            continue
        data = copy.deepcopy(dict(candidate))
        data.pop('profile_type', None)
        data = _normalize_import_profile_data(data)
        try:
            profiles.append(profile_from_config(data))
        except (TypeError, ValueError):
            continue
    return profiles


def default_profile(provider: str) -> LLMProfile:
    """Create a built-in profile for a provider.

    Example:
        >>> default_profile('Ollama').require_api_key
        False
    """

    return LLMProfile.from_provider(provider)


def default_profiles() -> List[LLMProfile]:
    return [default_profile(provider) for provider in PROVIDER_DEFAULTS]


def canonical_provider(provider: str) -> str:
    return PROVIDER_ALIASES.get(provider, provider)


def _provider_from_profile_id(profile_id: str) -> str:
    for provider, defaults in PROVIDER_DEFAULTS.items():
        if profile_id == defaults["id"]:
            return provider
    return ""


def _profile_value(profile: Any, attr: str, default=None) -> Any:
    if isinstance(profile, LLMProfile):
        return getattr(profile, attr)
    if isinstance(profile, Mapping):
        if attr in profile:
            return profile[attr]
    return default


def _builtin_profile_id(profile: Any) -> str:
    profile_id = str(_profile_value(profile, "id", "") or "")
    if _profile_value(profile, "built_in") and _provider_from_profile_id(profile_id):
        return profile_id
    return ""


def profile_by_id(
    profiles: Sequence[Any],
    profile_id: str,
) -> Optional[Any]:
    for profile in profiles:
        if _profile_value(profile, "id") == profile_id:
            return profile
    return None


def runtime_profile(
    profiles: Sequence[Any],
    selected_profile_id: str,
) -> LLMProfile:
    """Copy the selected runtime profile, falling back to the first entry.

    >>> runtime_profile([LLMProfile(id='first')], 'missing').id
    'first'
    """
    selected = profile_by_id(profiles, selected_profile_id)
    if selected is None and profiles:
        selected = profiles[0]
    if selected is None:
        raise RuntimeError('No LLM profile is configured.')
    return profile_from_config(selected)


def _merge_profile_options(default_options: Any, saved_options: Any, selected: Any) -> List[str]:
    merged = []
    for option in (default_options if isinstance(default_options, list) else []):
        if isinstance(option, str) and option not in merged:
            merged.append(option)
    for option in (saved_options if isinstance(saved_options, list) else []):
        if isinstance(option, str) and option not in merged:
            merged.append(option)
    if isinstance(selected, str) and selected and selected not in merged:
        merged.append(selected)
    return merged


def _merge_builtin_profile_options(profile: LLMProfile) -> LLMProfile:
    provider = _provider_from_profile_id(_builtin_profile_id(profile))
    if not provider:
        return profile
    defaults = PROVIDER_DEFAULTS[provider]
    profile.model_options = _merge_profile_options(
        defaults.get('model_options'), profile.model_options, profile.model,
    )
    profile.vision_model_options = _merge_profile_options(
        defaults.get('vision_model_options'), profile.vision_model_options, profile.vision_model,
    )
    profile.image_model_options = _merge_profile_options(
        defaults.get('image_model_options'), profile.image_model_options, profile.image_model,
    )
    if 'image_base_url' in defaults and not profile.image_base_url:
        profile.image_base_url = defaults['image_base_url']
    if 'protocol' in defaults and (not getattr(profile, 'protocol', None) or profile.protocol == PROTOCOL_OPENAI_COMPLETION):
        profile.protocol = defaults['protocol']
    return profile


def load_profiles(profiles: List[Any]) -> List[LLMProfile]:
    loaded = []
    for profile in profiles or []:
        if not isinstance(profile, (Mapping, LLMProfile)):
            continue
        loaded.append(_merge_builtin_profile_options(profile_from_config(profile)))
    return loaded


def _dedupe_profile_entries(entries: List[Tuple[Any, bool]], selected_profile_id: str = "") -> List[LLMProfile]:
    deduped = []
    by_key = {}
    for raw_profile, selected_old_translator in entries:
        profile = profile_from_config(raw_profile)
        selected_profile = profile.id == selected_profile_id
        builtin_id = _builtin_profile_id(profile)
        key = ("builtin", builtin_id) if builtin_id else ("custom", profile.id)
        provider = _provider_from_profile_id(builtin_id)
        has_key = bool(profile.api_key)
        builtin_model = bool(provider and profile.model in PROVIDER_DEFAULTS[provider]["model_options"][:2])
        score = (bool(selected_old_translator), bool(selected_profile), has_key, builtin_model)
        existing_idx = by_key.get(key)
        if existing_idx is None:
            by_key[key] = len(deduped)
            deduped.append((profile, score))
        elif score > deduped[existing_idx][1]:
            deduped[existing_idx] = (profile, score)

    profiles = []
    for profile, _score in deduped:
        builtin_id = _builtin_profile_id(profile)
        provider = _provider_from_profile_id(builtin_id)
        if provider:
            defaults = PROVIDER_DEFAULTS[provider]
            profile.id = defaults["id"]
            profile.name = provider
            profile.built_in = True
        profiles.append(profile)
    return profiles


def restore_builtin_profiles(existing_profiles: List[Any]) -> List[LLMProfile]:
    """Replace all built-in profiles while keeping user profiles.

    Example:
        >>> custom = copy_profile(default_profile('OpenAI'))
        >>> custom.id = 'custom'
        >>> restore_builtin_profiles([default_profile('OpenAI'), custom])[0].id
        'custom'
    """

    existing = load_profiles(existing_profiles)
    user_profiles = [p for p in existing if not p.built_in]
    preserved_keys = {}
    for profile in existing:
        if not profile.built_in or not profile.api_key:
            continue
        builtin_id = _builtin_profile_id(profile)
        if builtin_id:
            preserved_keys[builtin_id] = copy.deepcopy(profile.api_key)

    builtins = default_profiles()
    for profile in builtins:
        api_key = preserved_keys.get(profile.id)
        if api_key:
            profile.api_key = api_key
    return user_profiles + builtins


def copy_profile(profile: Any) -> LLMProfile:
    copied = profile_from_config(profile)
    copied.id = _stable_profile_id(copied.id, copied.base_url, copied.model, copied.api_key, suffix="copy")
    copied.name = copied.name + " Copy"
    copied.built_in = False
    return copied


def resolve_api_key(profile: Any, secret_store: SecretStore = None) -> str:
    secret_store = secret_store or SecretStore()
    return secret_store.resolve(_profile_value(profile, "api_key", "")).value


def store_api_key(profile: LLMProfile, api_key: str, secret_store: SecretStore = None) -> None:
    secret_store = secret_store or SecretStore()
    profile.api_key = secret_store.store(profile.id, api_key or "")


def _stable_profile_id(provider: str, base_url: str, model: str, api_key: Any, suffix: str = "") -> str:
    seed = "|".join([provider or "", _normal_url(base_url or ""), model or "", str(api_key or ""), suffix])
    digest = hashlib.sha1(seed.encode("utf8")).hexdigest()[:10]
    provider_slug = (provider or "llm").lower().replace(" ", "-")
    return f"{provider_slug}-{digest}"


def _infer_provider(provider: str, base_url: str, model: str) -> str:
    provider = canonical_provider(provider)
    url = _normal_url(base_url).lower()
    model = (model or "").lower()
    if "api.deepseek.com" in url or model.startswith("deepseek"):
        return "DeepSeek"
    if "generativelanguage.googleapis.com" in url or model.startswith("gemini"):
        return "Gemini"
    if "api.x.ai" in url or model.startswith("grok"):
        return "Grok"
    if "openrouter.ai" in url:
        return "OpenRouter"
    if "localhost:1234" in url:
        return "LM Studio"
    if "localhost:11434" in url or "127.0.0.1:11434" in url:
        return "Ollama"
    if provider in PROVIDER_DEFAULTS:
        return provider
    return "OpenAI"


def normalize_model(provider: str, model: str) -> Tuple[str, str]:
    provider = canonical_provider(provider)
    model = _strip_model_prefix(model)
    thinking = THINKING_AUTO
    if provider == "OpenAI":
        aliases = {
            "gpt3": "gpt-5.5",
            "text-davinci-003": "gpt-5.5",
            "gpt35-turbo": "gpt-5.5",
            "gpt-3.5-turbo": "gpt-5.5",
            "gpt4": "gpt-5.5",
            "gpt-4": "gpt-5.5",
        }
        model = aliases.get(model, model)
    elif provider == "DeepSeek":
        if model == "deepseek-reasoner":
            model = "deepseek-v4-flash"
            thinking = "high"
        elif model == "deepseek-chat":
            model = "deepseek-v4-flash"
            thinking = THINKING_AUTO
    if not model:
        model = PROVIDER_DEFAULTS[provider]["model"]
    return model, thinking


def _old_base_url(old_key: str, params: Dict, provider: str) -> str:
    if old_key == "LLM_API_Translator":
        base_url = params.get("endpoint") or ""
    else:
        base_url = params.get("3rd party api url") or ""
    if base_url:
        return str(base_url).strip()
    return PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["OpenAI"])["base_url"]


def _old_api_key(old_key: str, params: Dict) -> str:
    if old_key == "LLM_API_Translator":
        return str(params.get("apikey") or "").strip()
    return str(params.get("api key") or "").strip()


def profile_from_old_settings(old_key: str, params: Dict, secret_store: SecretStore = None) -> Optional[LLMProfile]:
    """Convert one old LLM translator config to a new profile.

    Example:
        >>> profile_from_old_settings('LLM_API_Translator', {}, secret_store=SecretStore()) is None
        True
    """

    params = params or {}
    override_model = _strip_model_prefix(str(params.get("override model") or ""))
    raw_model = _strip_model_prefix(str(override_model or params.get("model") or ""))
    using_override = bool(override_model)
    raw_provider = str(params.get("provider") or "")
    provider = _infer_provider(raw_provider, _old_base_url(old_key, params, "OpenAI"), raw_model)
    base_url = _old_base_url(old_key, params, provider)
    model, thinking = normalize_model(provider, raw_model)
    if provider in {"OpenAI", "DeepSeek"} and not using_override:
        if model not in PROVIDER_DEFAULTS[provider]["model_options"]:
            model = PROVIDER_DEFAULTS[provider]["model"]
            thinking = THINKING_AUTO
    api_key = _old_api_key(old_key, params)
    require_key = PROVIDER_DEFAULTS[provider]["require_api_key"]
    if require_key and not api_key:
        return None

    profile = default_profile(provider)
    matched_builtin = (
        _normal_url(base_url) == _normal_url(PROVIDER_DEFAULTS[provider]["base_url"])
        and model in PROVIDER_DEFAULTS[provider]["model_options"]
    )
    profile.id = PROVIDER_DEFAULTS[provider]["id"] if matched_builtin else _stable_profile_id(provider, base_url, model, api_key)
    profile.name = provider if matched_builtin else f"{provider} {model}"
    profile.built_in = matched_builtin
    profile.base_url = base_url
    profile.require_api_key = require_key
    profile.model = model
    profile.thinking_level = thinking
    if "max tokens" in params:
        profile.max_tokens = params["max tokens"]
    if "temperature" in params:
        profile.temperature = params["temperature"]
    if "top p" in params:
        profile.top_p = params["top p"]
    if "frequency penalty" in params:
        profile.frequency_penalty = params["frequency penalty"]
    if "presence penalty" in params:
        profile.presence_penalty = params["presence penalty"]
    if "low vram mode" in params:
        profile.low_vram_mode = params["low vram mode"]
    if api_key:
        store_api_key(profile, api_key, secret_store=secret_store)
    if profile.model not in profile.model_options:
        profile.model_options.insert(0, profile.model)
    return profile


def migrate_module_llm_profiles(module_cfg: Dict, secret_store: SecretStore = None) -> Dict:
    """Migrate old LLM translator settings in a raw module config dict.

    Example:
        >>> migrated = migrate_module_llm_profiles(
        ...     {}, secret_store=SecretStore())
        >>> bool(migrated['llm_profiles'])
        True
    """

    if not isinstance(module_cfg, dict):
        return module_cfg
    secret_store = secret_store or SecretStore()
    raw_profiles = module_cfg.get("llm_profiles") or []
    profiles_were_missing = "llm_profiles" not in module_cfg
    if isinstance(raw_profiles, list):
        profiles = load_profiles(raw_profiles) if raw_profiles else default_profiles()
    else:
        profiles = default_profiles()

    trans_params = module_cfg.get("translator_params")
    if not isinstance(trans_params, dict):
        trans_params = {}
        module_cfg["translator_params"] = trans_params
    current_translator = module_cfg.get("translator")
    migrated_entries = []
    selected_profile = None
    for old_key in OLD_LLM_TRANSLATORS:
        if old_key not in trans_params:
            continue
        old_params = trans_params.get(old_key) or {}
        profile = profile_from_old_settings(
            old_key,
            old_params,
            secret_store=secret_store,
        )
        trans_params.pop(old_key, None)
        if profile is None:
            continue
        migrated_entries.append((profile, bool(current_translator == old_key)))
        if current_translator == old_key:
            selected_profile = profile.id

    if migrated_entries:
        profiles = _dedupe_profile_entries(
            migrated_entries + [(profile, False) for profile in profiles],
            selected_profile or module_cfg.get("translator_llm_id", ""),
        )
        if selected_profile:
            module_cfg["translator"] = LLM_TRANSLATOR_KEY
            module_cfg["translator_llm_id"] = selected_profile
    elif profiles_were_missing:
        profiles = default_profiles()

    module_cfg["llm_profiles"] = profiles
    return module_cfg


def get_candidate_model_endpoints(base_url: str, api_key: str = "") -> List[str]:
    """Return prioritized candidate URLs to query available models."""
    url = str(base_url or "").strip().rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[:-len("/chat/completions")].rstrip("/")
    if url.endswith("/images/edits"):
        url = url[:-len("/images/edits")].rstrip("/")
    if url.endswith("/images"):
        url = url[:-len("/images")].rstrip("/")

    candidates: List[str] = []
    if not url:
        return candidates

    if url.endswith("/models"):
        return [url]

    candidates.append(f"{url}/models")

    if not url.endswith(("/v1", "/v1beta")) and "/v1/" not in url and "/v1beta/" not in url:
        candidates.append(f"{url}/v1/models")

    if "generativelanguage.googleapis.com" in url:
        if api_key:
            candidates.append(f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}")
        else:
            candidates.append("https://generativelanguage.googleapis.com/v1beta/models")

    if "11434" in url or "ollama" in url.lower():
        ollama_root = url.split("/v1")[0].rstrip("/")
        candidates.append(f"{ollama_root}/api/tags")

    seen = set()
    deduped = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def _extract_model_ids(data: Any) -> List[str]:
    """Extract and normalize model names from provider JSON responses."""
    models: List[str] = []
    if isinstance(data, dict):
        raw_list = data.get("data") or data.get("models") or []
        if isinstance(raw_list, list):
            for item in raw_list:
                if isinstance(item, dict):
                    if "supportedGenerationMethods" in item:
                        methods = item.get("supportedGenerationMethods") or []
                        if "generateContent" not in methods:
                            continue
                    name = item.get("id") or item.get("name") or ""
                elif isinstance(item, str):
                    name = item
                else:
                    name = ""
                name = str(name).strip()
                if name.startswith("models/"):
                    name = name[len("models/"):]
                if name:
                    models.append(name)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                name = item.get("id") or item.get("name") or ""
            elif isinstance(item, str):
                name = item
            else:
                name = ""
            name = str(name).strip()
            if name.startswith("models/"):
                name = name[len("models/"):]
            if name:
                models.append(name)

    seen = set()
    deduped = []
    for m in models:
        m = m.strip()
        if m and m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped


def fetch_remote_models(
    base_url: str,
    api_key: str = "",
    timeout: float = 15.0,
    proxy: Optional[str] = None,
) -> List[str]:
    """Fetch available model IDs from provider endpoint."""
    import requests

    base_url = str(base_url or "").strip()
    if not base_url:
        raise ValueError("Base URL cannot be empty")

    from ballontranslator.utils.project.key_pool import is_key_file, get_key_pool
    if is_key_file(api_key):
        pool = get_key_pool(api_key)
        first_key = pool.get_next_key()
        if first_key:
            api_key = first_key

    candidates = get_candidate_model_endpoints(base_url, api_key)
    if not candidates:
        raise ValueError(f"Invalid Base URL: {base_url}")

    proxies = None
    if proxy:
        proxies = {"http": proxy, "https": proxy}

    last_error: Optional[Exception] = None
    auth_failed: bool = False

    for endpoint in candidates:
        headers = {
            "Accept": "application/json",
            "User-Agent": "BallonsTranslator/1.0",
        }
        params = {}
        is_native_gemini = "generativelanguage.googleapis.com" in endpoint and "/openai/" not in endpoint

        if is_native_gemini:
            if api_key:
                headers["x-goog-api-key"] = api_key
                params["key"] = api_key
        else:
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        try:
            resp = requests.get(
                endpoint,
                headers=headers,
                params=params if params else None,
                timeout=timeout,
                proxies=proxies,
            )
            if resp.status_code == 200:
                data = resp.json()
                models = _extract_model_ids(data)
                if models:
                    return models
            elif resp.status_code in (401, 403):
                auth_failed = True
                msg = resp.text[:200]
                try:
                    err_json = resp.json()
                    if isinstance(err_json, dict) and "error" in err_json:
                        err_val = err_json["error"]
                        if isinstance(err_val, dict):
                            msg = err_val.get("message", msg)
                        elif isinstance(err_val, str):
                            msg = err_val
                except Exception:
                    pass
                last_error = RuntimeError(f"Authentication failed (HTTP {resp.status_code}): {msg}")
                break
            else:
                last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as exc:
            last_error = exc

    if auth_failed and last_error:
        raise last_error

    if last_error:
        raise RuntimeError(f"{last_error}")

    return []

