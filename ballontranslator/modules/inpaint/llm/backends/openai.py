"""OpenAI-compatible Image Inpainting Backend."""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np

from .base import BaseImageBackend, register_image_backend


@register_image_backend
class OpenAIImageBackend(BaseImageBackend):
    """Backend for standard OpenAI images/edits and images/generations endpoints."""

    backend_id = "openai"
    name = "OpenAI"
    default_base_url = "https://api.openai.com/v1"
    default_model = "gpt-image-2"
    supported_models = ["gpt-image-2"]

    @classmethod
    def matches(cls, profile: Any, base_url: str, model: Optional[str] = None) -> bool:
        host = urlparse(base_url).netloc.lower()
        if host in {"api.openai.com"} or host.endswith(".openai.com"):
            return True
        provider_name = str(getattr(profile, "name", "") or "").lower()
        return "openai" in provider_name

    @classmethod
    def get_endpoint(cls, base_url: str) -> str:
        base = base_url.rstrip("/")
        if base.endswith(("images", "images/edits", "images/generations")):
            return base
        return f"{base}/images/edits"

    @classmethod
    def get_generation_url(cls, base_url: str) -> str:
        base = base_url.rstrip("/")
        if base.endswith("/images/edits"):
            return f"{base[:-len('/images/edits')]}/images/generations"
        if base.endswith("/images"):
            return f"{base}/generations"
        if base.endswith("/images/generations"):
            return base
        return f"{base}/images/generations"

    def request_image(
        self,
        requester: Any,
        client: Any,
        profile: Any,
        image_file: Optional[io.BytesIO],
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> np.ndarray:
        raw_base = getattr(requester, "_image_base_url", lambda p: p.image_base_url)(profile)
        base_url = self.get_endpoint(raw_base)
        api_key = requester._api_key_for_profile(profile)
        headers = requester._headers(api_key)

        selected_model = str(model or getattr(profile, "image_model", "") or self.default_model).strip()
        selected_prompt = prompt if prompt is not None else getattr(profile, "image_prompt", "")

        if image_file is None:
            gen_url = self.get_generation_url(base_url)
            json_headers = requester._headers(api_key, json_request=True)
            response = client.post(
                gen_url,
                headers=json_headers,
                json={"model": selected_model, "prompt": selected_prompt},
            )
        else:
            response = client.post(
                base_url,
                headers=headers,
                data={"model": selected_model, "prompt": selected_prompt},
                files={
                    "image": (
                        "image.png",
                        image_file.getvalue(),
                        "image/png",
                    ),
                },
            )
        requester._raise_for_response(profile, response)
        return self.decode_response_image(response.json())
