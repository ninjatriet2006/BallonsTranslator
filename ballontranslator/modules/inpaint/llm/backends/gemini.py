"""Google Gemini Imagen backend."""

from __future__ import annotations

import io
from typing import Any, List, Optional
from urllib.parse import urlparse, urlunparse

import numpy as np

from .base import BaseImageBackend, register_image_backend


@register_image_backend
class GeminiImageBackend(BaseImageBackend):
    """Backend for Google Gemini image generation (:generateContent)."""

    backend_id = "gemini"
    name = "Gemini Imagen"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    default_model = "gemini-3.1-flash-lite-image"
    supported_models = ["gemini-3.1-flash-lite-image"]

    @classmethod
    def matches(cls, profile: Any, base_url: str, model: Optional[str] = None) -> bool:
        host = urlparse(base_url).netloc.lower()
        if host == "generativelanguage.googleapis.com":
            return True
        model_str = str(model or getattr(profile, "image_model", "") or "").lower()
        if "gemini" in model_str or "imagen" in model_str:
            return True
        provider_name = str(getattr(profile, "name", "") or "").lower()
        return "gemini" in provider_name or "google" in provider_name

    @classmethod
    def get_endpoint(cls, base_url: str, model: str = "") -> str:
        base = base_url.rstrip("/")
        parsed = urlparse(base)
        path = parsed.path.rstrip("/")
        if path.endswith(":generateContent"):
            return base
        if path.endswith("/openai"):
            path = path[:-len("/openai")]
            base = urlunparse(
                parsed._replace(path=path, params="", query="", fragment="")
            ).rstrip("/")
        model_name = model or "gemini-3.1-flash-lite-image"
        model_path = model_name if model_name.startswith("models/") else f"models/{model_name}"
        return f"{base}/{model_path}:generateContent"

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
        selected_model = str(model or getattr(profile, "image_model", "") or self.default_model).strip()
        endpoint = self.get_endpoint(raw_base, selected_model)
        api_key = requester._api_key_for_profile(profile)
        headers = requester._headers(api_key, json_request=True)

        selected_prompt = prompt if prompt is not None else getattr(profile, "image_prompt", "")
        parts = []
        if selected_prompt:
            parts.append({"text": selected_prompt})
        if image_file is not None:
            import base64
            parts.append({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": base64.b64encode(image_file.getvalue()).decode("ascii"),
                }
            })
        payload = {"contents": [{"role": "user", "parts": parts}]}
        response = client.post(endpoint, headers=headers, json=payload)
        requester._raise_for_response(profile, response)

        # Decode Gemini response format
        data = response.json()
        candidates = data.get("candidates") or []
        for candidate in candidates:
            parts = candidate.get("content", {}).get("parts") or []
            for part in parts:
                inline = part.get("inlineData")
                if inline and inline.get("data"):
                    import base64
                    import cv2
                    raw = base64.b64decode(inline["data"])
                    nparr = np.frombuffer(raw, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        return img
        raise RuntimeError(f"Could not parse image from Gemini response: {data}")
