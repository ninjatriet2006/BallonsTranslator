import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .base import register_textdetectors, TextDetectorBase, TextBlock, ProjImgTrans
from ..llm_chat import (
    LLMChatRequester,
    openai_chat_completion_args,
    openai_json_response_format,
)
from ..llm_vision import encode_chat_image
from ballontranslator.modules.exceptions import (
    LLMModelRequiredError,
    LLMRequestStopped,
    LLMUserActionRequiredError,
)
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import (
    LLMProfile,
    runtime_profile,
)
from ballontranslator.utils.textblock import sort_regions, examine_textblk, sort_pnts


DETECTOR_SYSTEM_PROMPT = (
    "You are an expert manga and comic OCR and text detection engine. "
    "Your task is to detect all text blocks (speech balloons, narration boxes, "
    "thought bubbles, sound effects, and side text) in the comic image and transcribe their exact original text.\n"
    "Respond with a JSON object containing a 'blocks' array. "
    "Each block must have:\n"
    "- 'box_2d': [ymin, xmin, ymax, xmax] normalized between 0 and 1000.\n"
    "- 'text': verbatim transcription of the text.\n"
    "- 'vertical': true if text is vertical (top-to-bottom), false if horizontal.\n"
    "Do not include any explanation or markdown; output pure JSON only."
)

DETECTOR_USER_PROMPT = (
    "Detect all text boxes and transcribe their text in this manga/comic page. "
    "Return the 2D bounding boxes [ymin, xmin, ymax, xmax] normalized from 0 to 1000."
)

DETECTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "[ymin, xmin, ymax, xmax] coordinates normalized from 0 to 1000",
                    },
                    "text": {
                        "type": "string",
                        "description": "Recognized text inside this bounding box",
                    },
                    "vertical": {
                        "type": "boolean",
                        "description": "True if vertical Japanese/Chinese text, False if horizontal",
                    },
                },
                "required": ["box_2d", "text", "vertical"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["blocks"],
    "additionalProperties": False,
}


def _clean_json_text(raw_text: str) -> str:
    """Extract and clean valid JSON substring from LLM response."""
    text = raw_text.strip()
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
        last_brace = text.rfind('}')
        last_bracket = text.rfind(']')
        end_idx = max(last_brace, last_bracket)
        if end_idx > start_idx:
            text = text[start_idx:end_idx + 1]
    return text


def _parse_box_coordinates(
    box_raw: Any,
    img_w: int,
    img_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Parse bounding box into (x1, y1, x2, y2) in absolute pixel coordinates.
    Expects [ymin, xmin, ymax, xmax] normalized to 0-1000 or 0-1, or dict.
    """
    coords = None
    if isinstance(box_raw, (list, tuple)) and len(box_raw) == 4:
        try:
            coords = [float(v) for v in box_raw]
        except (ValueError, TypeError):
            return None
    elif isinstance(box_raw, dict):
        if all(k in box_raw for k in ('ymin', 'xmin', 'ymax', 'xmax')):
            coords = [float(box_raw['ymin']), float(box_raw['xmin']), float(box_raw['ymax']), float(box_raw['xmax'])]
        elif all(k in box_raw for k in ('y1', 'x1', 'y2', 'x2')):
            coords = [float(box_raw['y1']), float(box_raw['x1']), float(box_raw['y2']), float(box_raw['x2'])]
        elif all(k in box_raw for k in ('top', 'left', 'bottom', 'right')):
            coords = [float(box_raw['top']), float(box_raw['left']), float(box_raw['bottom']), float(box_raw['right'])]

    if not coords:
        return None

    ymin, xmin, ymax, xmax = coords

    max_val = max(abs(ymin), abs(xmin), abs(ymax), abs(xmax))
    if max_val <= 1.0:
        y1 = int(round(ymin * img_h))
        x1 = int(round(xmin * img_w))
        y2 = int(round(ymax * img_h))
        x2 = int(round(xmax * img_w))
    elif max_val <= 1000.0:
        y1 = int(round(ymin * img_h / 1000.0))
        x1 = int(round(xmin * img_w / 1000.0))
        y2 = int(round(ymax * img_h / 1000.0))
        x2 = int(round(xmax * img_w / 1000.0))
    else:
        y1 = int(round(ymin))
        x1 = int(round(xmin))
        y2 = int(round(ymax))
        x2 = int(round(xmax))

    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    x1 = max(0, min(img_w - 1, x1))
    x2 = max(0, min(img_w, x2))
    y1 = max(0, min(img_h - 1, y1))
    y2 = max(0, min(img_h, y2))

    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def extract_stroke_mask(roi: np.ndarray) -> np.ndarray:
    """Extract fine text stroke mask inside a bounding box to avoid wiping background artwork."""
    if roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 2:
        return np.full((max(1, roi.shape[0]), max(1, roi.shape[1])), 255, dtype=np.uint8)

    if roi.ndim == 3:
        if roi.shape[2] == 4:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi

    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg_median = float(np.median(border))
    bg_is_light = bg_median > 127.0

    thresh_type = cv2.THRESH_BINARY_INV if bg_is_light else cv2.THRESH_BINARY
    _, otsu_mask = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)

    total_pixels = gray.shape[0] * gray.shape[1]
    fg_ratio = np.count_nonzero(otsu_mask) / max(1, total_pixels)

    if 0.03 <= fg_ratio <= 0.65:
        return otsu_mask

    ad_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresh_type, 15, 8
    )
    ad_ratio = np.count_nonzero(ad_mask) / max(1, total_pixels)
    if 0.03 <= ad_ratio <= 0.70:
        return ad_mask

    return np.full_like(gray, 255)


@register_textdetectors('llm_detector')
class LLMDetector(LLMChatRequester, TextDetectorBase):
    """Profile-backed All-in-One Text Detector & OCR using vision chat models."""

    dependencies = ['openai>=2.8.1', 'httpx[socks,brotli]']

    params = {
        'max requests per minute': {
            'value': 20,
            'display_name': 'Max Requests Per Minute',
            'description': 'Global request limit for LLM Detector.',
        },
        'delay': {
            'value': 0.3,
            'display_name': 'Delay',
            'description': 'Delay between LLM Detector requests in seconds.',
        },
        'retry attempts': {
            'value': 3,
            'display_name': 'Retry Attempts',
            'description': 'Retries for API failures.',
        },
        'retry timeout': {
            'value': 7.0,
            'display_name': 'Retry Timeout',
            'description': 'Delay between retries in seconds.',
        },
        'request timeout': {
            'value': 180.0,
            'display_name': 'Request Timeout',
            'description': 'HTTP request timeout in seconds.',
        },
        'mask mode': {
            'type': 'selector',
            'options': ['adaptive_stroke', 'bounding_box'],
            'value': 'adaptive_stroke',
            'display_name': 'Mask Mode',
            'description': 'adaptive_stroke extracts letter strokes; bounding_box fills the whole box.',
        },
        'mask dilate size': {
            'value': 3,
            'display_name': 'Mask Dilate Size',
            'description': 'Pixel dilation applied to the inpainting mask.',
        },
        'proxy': {
            'value': '',
            'display_name': 'Proxy',
            'description': 'Proxy address used for the OpenAI-compatible client.',
        },
        'description': 'All-in-One LLM Vision (Det & OCR)',
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.token_count = 0
        self.token_count_last = 0

    @property
    def profile(self) -> LLMProfile:
        detector_id = getattr(pcfg.module, 'detector_llm_id', '')
        profile_id = detector_id or pcfg.module.ocr_llm_id
        profile = runtime_profile(
            pcfg.module.llm_profiles,
            profile_id,
        )
        if not profile.support_vision:
            raise RuntimeError(f'LLM profile "{profile.name}" does not have vision enabled.')
        self._vision_model(profile)
        return profile

    @staticmethod
    def _vision_model(profile: LLMProfile) -> str:
        model = str(profile.vision_model or '').strip()
        model_options = [str(option).strip() for option in profile.vision_model_options if str(option).strip()]
        if not model or not model_options:
            raise LLMModelRequiredError(profile.id, profile.name, target='vision_model')
        return model

    def _image_content_part(self, img: np.ndarray, profile: LLMProfile) -> Dict:
        return encode_chat_image(
            img,
            detail=str(profile.vision_detail_level or 'None'),
            failure_message='Failed to encode detection image.',
        ).image_part()

    def _messages(
        self,
        img: np.ndarray,
        profile: LLMProfile,
        prompt: Optional[str] = None,
    ) -> List[Dict]:
        return [
            {"role": "system", "content": DETECTOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt if prompt is not None else DETECTOR_USER_PROMPT},
                    self._image_content_part(img, profile),
                ],
            },
        ]

    def _api_args(
        self,
        profile: LLMProfile,
        messages: List[Dict],
        response_schema: Optional[Dict],
    ) -> Dict:
        model = self._vision_model(profile)
        api_args = openai_chat_completion_args(
            profile,
            model,
        )
        api_args['messages'] = messages
        if response_schema is not None:
            api_args['response_format'] = openai_json_response_format(
                profile,
                'detection_and_ocr',
                response_schema,
            )
        return api_args

    def _request_with_retries(
        self,
        profile: LLMProfile,
        messages: List[Dict],
        response_schema: Optional[Dict],
        failure_label: str = 'LLM Detection+OCR',
    ) -> str:
        retry_attempt = 0
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                raise LLMRequestStopped()
            try:
                completion = self.request_chat_completion(
                    profile,
                    self._api_args(profile, messages, response_schema),
                )
                if completion.usage is not None:
                    self.token_count += getattr(completion.usage, 'total_tokens', 0)
                    self.token_count_last = getattr(completion.usage, 'total_tokens', 0)
                else:
                    self.token_count_last = 0
                if self.token_count_last:
                    self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')
                return completion.content
            except (LLMUserActionRequiredError, LLMRequestStopped):
                raise
            except Exception as e:
                if profile.json_schema_response_format and response_schema is not None:
                    self.logger.warning(
                        f"{failure_label} failed with json_schema ({e}). Falling back to json_object format."
                    )
                    profile.json_schema_response_format = False
                    continue
                retry_attempt += 1
                if retry_attempt >= self.get_param_value('retry attempts'):
                    raise RuntimeError(f'{failure_label} failed: {e}') from e
                self.logger.warning(
                    f'{failure_label} failed due to {e}. Attempt: {retry_attempt}'
                )
                self._wait(self.get_param_value('retry timeout'))

    def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:
        if self.stop_event is not None and self.stop_event.is_set():
            raise LLMRequestStopped()

        im_h, im_w = img.shape[:2]
        profile = self.profile

        # Bound max dimension to 2048 for API transmission to keep request fast and reliable
        max_dim = 2048
        if max(im_h, im_w) > max_dim:
            scale = max_dim / float(max(im_h, im_w))
            send_w, send_h = int(round(im_w * scale)), int(round(im_h * scale))
            send_img = cv2.resize(img, (send_w, send_h), interpolation=cv2.INTER_AREA)
        else:
            send_img = img

        messages = self._messages(send_img, profile)
        raw_response = self._request_with_retries(
            profile,
            messages,
            DETECTION_RESPONSE_SCHEMA,
        )

        cleaned_json = _clean_json_text(raw_response)
        try:
            data = json.loads(cleaned_json)
        except Exception as e:
            self.logger.error(f"Failed to parse LLM detector JSON: {e}\nRaw response:\n{raw_response}")
            raise ValueError(f"Invalid JSON returned by Vision LLM: {e}") from e

        if isinstance(data, list):
            raw_blocks = data
        elif isinstance(data, dict):
            raw_blocks = (
                data.get('blocks')
                or data.get('text_blocks')
                or data.get('boxes')
                or data.get('detections')
                or []
            )
        else:
            raw_blocks = []

        blk_list: List[TextBlock] = []
        for item in raw_blocks:
            if not isinstance(item, dict):
                continue
            box_raw = item.get('box_2d') or item.get('bbox') or item.get('box') or item.get('coordinates')
            coords = _parse_box_coordinates(box_raw, im_w, im_h)
            if coords is None:
                continue

            x1, y1, x2, y2 = coords
            raw_text = str(item.get('text', '')).strip()
            lines_text = [l for l in raw_text.splitlines() if l.strip()]
            if not lines_text and raw_text:
                lines_text = [raw_text]

            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            pts_sorted, detected_vertical = sort_pnts(pts)
            is_vertical = bool(item.get('vertical', detected_vertical))

            blk = TextBlock(
                xyxy=[x1, y1, x2, y2],
                lines=[pts_sorted],
                src_is_vertical=is_vertical,
                text=lines_text,
            )
            blk.vertical = is_vertical
            blk.adjust_bbox()
            examine_textblk(blk, im_w, im_h)
            blk_list.append(blk)

        blk_list = sort_regions(blk_list)

        mask = np.zeros((im_h, im_w), dtype=np.uint8)
        mask_mode = self.get_param_value('mask mode')

        for blk in blk_list:
            bx1, by1, bx2, by2 = blk.xyxy
            if mask_mode == 'bounding_box':
                mask[by1:by2, bx1:bx2] = 255
            else:
                roi = img[by1:by2, bx1:bx2]
                roi_mask = extract_stroke_mask(roi)
                mask[by1:by2, bx1:bx2] = np.bitwise_or(mask[by1:by2, bx1:bx2], roi_mask)

        dilate_size = int(self.get_param_value('mask dilate size'))
        if dilate_size > 0:
            element = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * dilate_size + 1, 2 * dilate_size + 1),
                (dilate_size, dilate_size),
            )
            mask = cv2.dilate(mask, element)

        return mask, blk_list
