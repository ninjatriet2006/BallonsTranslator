"""OpenRouter image generation backend."""

from __future__ import annotations

import io
from typing import Any, List, Optional
from urllib.parse import urlparse

import numpy as np

from .base import BaseImageBackend, register_image_backend


@register_image_backend
class OpenRouterImageBackend(BaseImageBackend):
    """Backend for OpenRouter images endpoint."""

    backend_id = "openrouter"
    name = "OpenRouter"
    default_base_url = "https://openrouter.ai/api/v1"
    default_model = "black-forest-labs/flux.2-klein-4b"
    supported_models = ["black-forest-labs/flux.2-klein-4b"]

    @classmethod
    def matches(cls, profile: Any, base_url: str, model: Optional[str] = None) -> bool:
        host = urlparse(base_url).netloc.lower()
        if host == "openrouter.ai" or host.endswith(".openrouter.ai"):
            return True
        provider_name = str(getattr(profile, "name", "") or "").lower()
        return "openrouter" in provider_name

    @classmethod
    def get_endpoint(cls, base_url: str) -> str:
        base = base_url.rstrip("/")
        if base.endswith("/images"):
            return base
        return f"{base}/images"

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
        endpoint = self.get_endpoint(raw_base)
        api_key = requester._api_key_for_profile(profile)
        headers = requester._headers(api_key, json_request=True)

        selected_model = str(model or getattr(profile, "image_model", "") or self.default_model).strip()
        selected_prompt = prompt if prompt is not None else getattr(profile, "image_prompt", "")
        payload = {"model": selected_model, "prompt": selected_prompt}
        if image_file is not None:
            import base64
            encoded = base64.b64encode(image_file.getvalue()).decode("ascii")
            payload["image"] = f"data:image/png;base64,{encoded}"

        response = client.post(endpoint, headers=headers, json=payload)
        requester._raise_for_response(profile, response)
        return self.decode_response_image(response.json())
