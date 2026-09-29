"""Agnes AI / Vyceai Image Inpainting Backend."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from .base import BaseImageBackend, register_image_backend


@register_image_backend
class AgnesImageBackend(BaseImageBackend):
    """Backend for Agnes AI / Vyceai image generation and inpainting."""

    backend_id = "agnes"
    name = "Agnes AI"
    default_base_url = "https://vyceai.com/v1"
    default_model = "agnes-image-2.0-flash"
    supported_models = ["agnes-image-2.0-flash"]

    @classmethod
    def matches(cls, profile: Any, base_url: str, model: Optional[str] = None) -> bool:
        host = urlparse(base_url).netloc.lower()
        if host in {"agnes-ai.com", "vyceai.com"} or host.endswith(".agnes-ai.com") or host.endswith(".vyceai.com"):
            return True
        model_str = str(model or getattr(profile, "image_model", "") or "").lower()
        if model_str.startswith("agnes-image"):
            return True
        provider_name = str(getattr(profile, "name", "") or "").lower()
        return "agnes" in provider_name or "vyceai" in provider_name

    @classmethod
    def get_endpoint(cls, base_url: str) -> str:
        base = base_url.rstrip("/")
        if base.endswith("/images/generations"):
            return base
        if "/v1" in base:
            return f"{base}/images/generations"
        return f"{base}/v1/images/generations"

    @classmethod
    def build_api_args(
        cls,
        profile: Any,
        image_file: Optional[io.BytesIO],
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        size = "1024x1024"
        if image_file is not None:
            try:
                with Image.open(io.BytesIO(image_file.getvalue())) as pil_img:
                    w, h = pil_img.size
                    if w > h * 1.2:
                        size = "1024x768"
                    elif h > w * 1.2:
                        size = "768x1024"
                    else:
                        size = "1024x1024"
            except Exception:
                size = "1024x1024"

        selected_model = str(model or getattr(profile, "image_model", "") or cls.default_model).strip()
        result: Dict[str, Any] = {
            "model": selected_model,
            "prompt": prompt if prompt is not None else getattr(profile, "image_prompt", ""),
            "size": size,
        }
        if image_file is None:
            result["return_base64"] = True
        else:
            encoded = base64.b64encode(image_file.getvalue()).decode("ascii")
            result["extra_body"] = {
                "image": [f"data:image/png;base64,{encoded}"],
                "response_format": "b64_json",
            }
        return result

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

        payload = self.build_api_args(profile, image_file, prompt=prompt, model=model)
        response = client.post(endpoint, headers=headers, json=payload)
        requester._raise_for_response(profile, response)
        return self.decode_response_image(response.json())
