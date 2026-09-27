from typing import Dict, List

import cv2
import numpy as np

from .base import (
    InpainterBase,
    register_inpainter,
    filter_mask_by_bboxes,
    inpaint_handle_alpha_channel,
)
from ballontranslator.utils.imgproc_utils import enlarge_window
from ballontranslator.utils.textblock_mask import extract_ballon_mask
from ..llm_image import LLMImageRequester
from ..textdetector import TextBlock
from ballontranslator.modules.exceptions import (
    LLMRequestStopped,
    LLMUserActionRequiredError,
)
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import (
    LLMProfile,
    runtime_profile,
)


LLM_IMAGE_REQUEST_PARAMS: Dict = {
    "max requests per minute": {
        "value": 5,
        "display_name": "Max Requests Per Minute",
        "description": "Global request limit for LLM image requests.",
    },
    "delay": {
        "value": 0.5,
        "display_name": "Delay",
        "description": "Delay between LLM image requests in seconds.",
    },
    "retry attempts": {
        "value": 3,
        "display_name": "Retry Attempts",
        "description": "Retries for API failures.",
    },
    "retry timeout": {
        "value": 7.0,
        "display_name": "Retry Timeout",
        "description": "Delay between retries in seconds.",
    },
    "request timeout": {
        "value": 180.0,
        "display_name": "Request Timeout",
        "description": (
            "HTTP timeout for image requests in seconds. Set to 0 to disable."
        ),
    },
    "max resolution": {
        "type": "selector",
        "options": [0, 256, 768, 1280],
        "value": 1280,
        "display_name": "Max Resolution",
        "description": (
            "Scale images down before sending them to the LLM. "
            "Set to 0 to keep the original size."
        ),
    },
    "proxy": {
        "value": "",
        "display_name": "Proxy",
        "description": "Proxy address used for the image request.",
    },
}


@register_inpainter("LLMInpaint")
class LLMInpaint(LLMImageRequester, InpainterBase):
    """Profile-backed image cleanup using shared LLM image transport.

    Example:
        >>> LLMInpaint._image_model_required('demo', ['demo'])
        'demo'
    """

    dependencies = ['httpx[socks,brotli]']

    params: Dict = {
        **LLM_IMAGE_REQUEST_PARAMS,
        "inpaint by block": {
            "type": "checkbox",
            "value": True,
            "display_name": "Inpaint By Block",
            "description": "Send each text block crop separately instead of sending the whole image.",
        },
        "description": "Inpaint using the selected image-capable LLM profile.",
    }

    inpaint_by_block = True

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._sync_inpaint_by_block()

    @property
    def profile(self) -> LLMProfile:
        profile = runtime_profile(
            pcfg.module.llm_profiles, pcfg.module.inpaint_llm_id
        )
        if not profile.support_image:
            raise RuntimeError(
                f'LLM profile "{profile.name}" does not have image cleanup enabled.'
            )
        self._image_model(profile)
        self._image_base_url(profile)
        return profile

    def _sync_inpaint_by_block(self) -> None:
        value = self.get_param_value('inpaint by block')
        if isinstance(value, str):
            value = value.lower().strip() == 'true'
        self.inpaint_by_block = bool(value)

    def updateParam(self, param_key: str, param_content) -> None:
        super().updateParam(param_key, param_content)
        if param_key == 'inpaint by block':
            self._sync_inpaint_by_block()

    def _inpaint(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        textblock_list: List[TextBlock] = None,
    ) -> np.ndarray:
        del textblock_list
        profile = self.profile
        retry_attempt = 0
        mask_original = (mask > 127)[..., None].astype(np.uint8)
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                raise LLMRequestStopped()
            try:
                result = self._request_inpaint(profile, img)
                if result.shape[:2] != img.shape[:2]:
                    result = cv2.resize(
                        result,
                        (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                result = result.astype(np.uint8, copy=False)
                return result * mask_original + img * (1 - mask_original)
            except (LLMUserActionRequiredError, LLMRequestStopped):
                raise
            except Exception as error:
                retry_attempt += 1
                if retry_attempt >= self.get_param_value('retry attempts'):
                    raise RuntimeError(
                        f'LLM image cleanup failed: {error}'
                    ) from error
                self.logger.warning(
                    'LLM image cleanup failed due to %s. Attempt: %s',
                    error,
                    retry_attempt,
                )
                self._wait(self.get_param_value('retry timeout'))

    def inpaint(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        textblock_list: List[TextBlock] = None,
        check_need_inpaint: bool = False,
    ) -> np.ndarray:
        profile = self.profile
        profile_concurrency = getattr(profile, 'concurrency', 0) if profile else 0
        workers = profile_concurrency if profile_concurrency > 0 else getattr(pcfg.module, 'llm_inpaint_threads', 2)
        workers = max(1, int(workers))

        if not self.inpaint_by_block or not textblock_list or len(textblock_list) <= 1 or workers <= 1:
            return super().inpaint(img, mask, textblock_list, check_need_inpaint)

        if not self.all_model_loaded():
            self.load_model()

        original_alpha = None
        if len(img.shape) == 3 and img.shape[2] == 4:
            original_alpha = img[:, :, 3:4]
            img_rgb = img[:, :, :3]
        else:
            img_rgb = img

        if pcfg.module.filter_mask_by_bboxes:
            mask = filter_mask_by_bboxes(mask, textblock_list)

        im_h, im_w = img_rgb.shape[:2]
        inpainted = np.copy(img_rgb)
        original_mask = mask.copy()

        # Identify which blocks need LLM inpaint vs simple background fill
        items_to_inpaint = []
        for idx, blk in enumerate(textblock_list):
            xyxy = blk.xyxy
            xyxy_e = enlarge_window(xyxy, im_w, im_h, ratio=1.7)
            im_crop = inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]]
            msk_crop = mask[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]]
            need_inp = True
            if pcfg.module.check_need_inpaint or check_need_inpaint:
                ballon_msk, non_text_msk = extract_ballon_mask(im_crop, msk_crop)
                if ballon_msk is not None:
                    non_text_region = np.where(non_text_msk > 0)
                    non_text_px = im_crop[non_text_region]
                    average_bg_color = np.median(non_text_px, axis=0)
                    std_rgb = np.std(non_text_px - average_bg_color, axis=0)
                    std_max = np.max(std_rgb)
                    inpaint_thresh = 7 if np.std(std_rgb) > 1 else 10
                    if std_max < inpaint_thresh:
                        need_inp = False
                        im_crop[np.where(ballon_msk > 0)] = average_bg_color
            if need_inp:
                items_to_inpaint.append((idx, xyxy, xyxy_e, im_crop.copy(), msk_crop.copy()))
            mask[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = 0

        if items_to_inpaint:
            import concurrent.futures

            def _worker(item):
                if self.stop_event is not None and self.stop_event.is_set():
                    raise LLMRequestStopped()
                idx, xyxy, xyxy_e, im_c, msk_c = item
                cleaned = self.memory_safe_inpaint(im_c, msk_c)
                return idx, xyxy_e, cleaned

            max_w = min(workers, len(items_to_inpaint))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as executor:
                futures = [executor.submit(_worker, it) for it in items_to_inpaint]
                for fut in concurrent.futures.as_completed(futures):
                    idx, xyxy_e, cleaned = fut.result()
                    inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]] = cleaned

        if original_alpha is not None:
            result_alpha = inpaint_handle_alpha_channel(original_alpha, original_mask)
            return np.concatenate([inpainted, result_alpha], axis=2)
        return inpainted
