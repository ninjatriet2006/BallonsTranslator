"""Base abstractions for LLM Inpaint and Image generation backends."""

from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image

from ballontranslator.modules.exceptions import (
    LLMApiKeyRequiredError,
    LLMBaseURLRequiredError,
    LLMModelRequiredError,
)
from ballontranslator.utils.system.logger import logger as LOGGER


class BaseImageBackend(ABC):
    """Abstract Base Class for image generation / inpainting provider backends."""

    backend_id: str = "base"
    name: str = "Base Backend"
    default_base_url: str = ""
    default_model: str = ""
    supported_models: List[str] = []

    @classmethod
    @abstractmethod
    def matches(cls, profile: Any, base_url: str, model: Optional[str] = None) -> bool:
        """Return True if this backend handles the given profile, base_url, or model."""
        pass

    @classmethod
    @abstractmethod
    def get_endpoint(cls, base_url: str) -> str:
        """Resolve the final HTTP endpoint URL for this provider."""
        pass

    @abstractmethod
    def request_image(
        self,
        requester: Any,
        client: Any,
        profile: Any,
        image_file: Optional[io.BytesIO],
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> np.ndarray:
        """Perform the HTTP request and return the resulting RGB(A) numpy image."""
        pass

    @staticmethod
    def decode_response_image(response_data: Any) -> np.ndarray:
        """Decode common Base64 or URL response into a uint8 BGR(A) numpy array."""
        if not isinstance(response_data, dict):
            raise RuntimeError(f"Unexpected image API response format: {response_data!r}")

        candidates = response_data.get("data") or response_data.get("images") or []
        if isinstance(candidates, list) and candidates:
            first = candidates[0]
            b64_str = ""
            if isinstance(first, dict):
                b64_str = first.get("b64_json") or first.get("base64") or ""
            elif isinstance(first, str):
                b64_str = first

            if b64_str:
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                raw_bytes = base64.b64decode(b64_str)
                nparr = np.frombuffer(raw_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    return img

        raise RuntimeError(f"Could not extract decoded image from API response: {response_data}")


_IMAGE_BACKEND_REGISTRY: List[Type[BaseImageBackend]] = []


def register_image_backend(cls: Type[BaseImageBackend]) -> Type[BaseImageBackend]:
    """Register an image provider backend."""
    if cls not in _IMAGE_BACKEND_REGISTRY:
        _IMAGE_BACKEND_REGISTRY.append(cls)
    return cls


def get_image_backend(profile: Any, model: Optional[str] = None, base_url: Optional[str] = None) -> BaseImageBackend:
    """Find and instantiate the matching image backend for the given profile."""
    resolved_base_url = str(base_url or getattr(profile, "image_base_url", "") or getattr(profile, "base_url", "") or "").strip()
    resolved_model = str(model or getattr(profile, "image_model", "") or "").strip()

    # Traverse registered backends in reverse order (newer/custom registered first)
    for backend_cls in reversed(_IMAGE_BACKEND_REGISTRY):
        if backend_cls.matches(profile, resolved_base_url, resolved_model):
            return backend_cls()

    # Fallback to default OpenAI compatible backend if none specifically matched
    for backend_cls in _IMAGE_BACKEND_REGISTRY:
        if backend_cls.backend_id == "openai":
            return backend_cls()

    from .openai import OpenAIImageBackend
    return OpenAIImageBackend()
