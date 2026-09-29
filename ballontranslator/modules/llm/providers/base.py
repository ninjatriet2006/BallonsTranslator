"""Base provider abstractions and registry for LLM providers."""

from __future__ import annotations

from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type


@dataclass
class ProviderPreset:
    """Standard definition of a built-in provider preset."""

    id: str
    name: str
    protocol: str = "openai_completion"
    base_url: str = ""
    require_api_key: bool = True
    model: str = ""
    model_options: List[str] = field(default_factory=list)
    support_text: bool = True
    support_vision: bool = False
    vision_model: str = ""
    vision_model_options: List[str] = field(default_factory=list)
    vision_detail_level: str = "auto"
    support_image: bool = False
    image_base_url: str = ""
    image_model: str = ""
    image_model_options: List[str] = field(default_factory=list)
    thinking_level: str = "Auto"
    json_schema_response_format: bool = False
    aliases: List[str] = field(default_factory=list)

    def to_defaults_dict(self) -> Dict[str, Any]:
        """Convert preset into dictionary format matching PROVIDER_DEFAULTS."""
        data: Dict[str, Any] = {
            "id": self.id,
            "base_url": self.base_url,
            "require_api_key": self.require_api_key,
            "model": self.model,
            "model_options": list(self.model_options),
            "protocol": self.protocol,
        }
        if self.support_vision:
            data["support_vision"] = True
            data["vision_model"] = self.vision_model or self.model
            data["vision_model_options"] = list(self.vision_model_options or self.model_options)
            data["vision_detail_level"] = self.vision_detail_level
        if self.support_image:
            data["support_image"] = True
            if self.image_base_url:
                data["image_base_url"] = self.image_base_url
            if self.image_model:
                data["image_model"] = self.image_model
            data["image_model_options"] = list(self.image_model_options)
        if self.thinking_level and self.thinking_level != "Auto":
            data["thinking_level"] = self.thinking_level
        if self.json_schema_response_format:
            data["json_schema_response_format"] = True
        return data


class BaseProvider(ABC):
    """Abstract Base Class for LLM Provider presets."""

    preset: ProviderPreset


_PROVIDER_REGISTRY: Dict[str, Type[BaseProvider]] = {}


def register_provider(cls: Type[BaseProvider]) -> Type[BaseProvider]:
    """Register a provider preset class."""
    if hasattr(cls, 'preset') and isinstance(cls.preset, ProviderPreset):
        _PROVIDER_REGISTRY[cls.preset.name] = cls
        _PROVIDER_REGISTRY[cls.preset.id] = cls
        for alias in cls.preset.aliases:
            _PROVIDER_REGISTRY[alias] = cls
    return cls


def get_registered_providers() -> Dict[str, ProviderPreset]:
    """Return map of canonical provider name to its ProviderPreset."""
    seen = {}
    for key, cls in _PROVIDER_REGISTRY.items():
        preset = cls.preset
        if preset.name not in seen:
            seen[preset.name] = preset
    return seen


def get_all_provider_defaults() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of all provider defaults keyed by provider name."""
    return {
        name: preset.to_defaults_dict()
        for name, preset in get_registered_providers().items()
    }
