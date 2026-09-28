import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ballontranslator.modules.textdetector.base import register_textdetectors, TextDetectorBase, TextBlock, ProjImgTrans
from ballontranslator.modules.llm_chat import (
    LLMChatRequester,
    openai_chat_completion_args,
    openai_json_response_format,
)
from ballontranslator.modules.llm_vision import encode_chat_image
from ballontranslator.modules.exceptions import (
    LLMModelRequiredError,
    LLMRequestStopped,
    LLMUserActionRequiredError,
)
from ballontranslator.utils.project.config import pcfg
from ballontranslator.utils.project.llm_profiles import (
    DEFAULT_OCR_PROMPT,
    DEFAULT_DETECTOR_BOX_PROMPT,
    DEFAULT_DETECTOR_OCR_PROMPT,
    LLMProfile,
    runtime_profile,
)
from ballontranslator.utils.text.textblock import sort_regions, examine_textblk, sort_pnts


def assemble_detector_system_prompt(
    detector_box_prompt: Optional[str] = None,
    detector_ocr_prompt: Optional[str] = None,
) -> str:
    box_guidelines = (
        detector_box_prompt
        if detector_box_prompt is not None and detector_box_prompt.strip()
        else DEFAULT_DETECTOR_BOX_PROMPT
    ).strip()
    ocr_guidelines = (
        detector_ocr_prompt
        if detector_ocr_prompt is not None and detector_ocr_prompt.strip()
        else DEFAULT_DETECTOR_OCR_PROMPT
    ).strip()

    return (
        "You are an expert manga and comic OCR and text detection engine.\n"
        "Your task is to detect all text clusters in the comic image and transcribe their exact original text.\n\n"
        "Detection Guidelines:\n"
        f"1. {box_guidelines}\n"
        "2. Detect separate bounding boxes for distinct speech bubbles; do not merge adjacent bubbles.\n"
        "3. 'box_2d': [ymin, xmin, ymax, xmax] tightly bounding the whole text bubble/cluster, normalized from 0 to 1000.\n"
        "4. 'lines': optional list of tight child bounding boxes for each individual text line within the block. "
        "Each child line has 'box_2d': [ymin, xmin, ymax, xmax] tightly bounding only that line, and 'text'.\n\n"
        "Transcription Guidelines:\n"
        f"1. {ocr_guidelines}\n"
        "2. 'text': verbatim transcription of the visible text inside the box, preserving line breaks with \\n.\n"
        "3. 'vertical': true if text is vertical (top-to-bottom), false if horizontal.\n\n"
        "Output Format & Technical Rules:\n"
        "- Order blocks and child lines according to natural comic reading order (top-to-bottom, right-to-left for manga).\n"
        "- Respond with pure JSON matching the schema: {\"blocks\": [{\"box_2d\": [...], \"text\": \"...\", \"vertical\": bool, \"lines\": [{\"box_2d\": [...]}]}]}.\n"
        "- Do not include any explanation or markdown wrappers."
    )


DETECTOR_SYSTEM_PROMPT = assemble_detector_system_prompt()

DETECTOR_USER_PROMPT = (
    "Detect all text boxes and transcribe their text in this manga/comic page. "
    "Return 2D bounding boxes [ymin, xmin, ymax, xmax] normalized from 0 to 1000, verbatim text, and child line boxes in natural reading order."
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
                    "lines": {
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
                                    "description": "Recognized text of this line",
                                },
                            },
                            "required": ["box_2d"],
                            "additionalProperties": False,
                        },
                        "description": "Child boxes for each individual text line within this cluster",
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
        end_idx = max(last_brace, last_bracket)
        if end_idx != -1 and end_idx < len(text) - 1:
            text = text[:end_idx + 1]

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


def is_flat_background(roi: np.ndarray, std_thresh: float = 14.0) -> bool:
    """Check if ROI has a flat/uniform background (such as a speech balloon or solid box)."""
    if roi is None or roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 2:
        return False
    try:
        roi = np.ascontiguousarray(roi)
        if roi.ndim == 3:
            channels = roi.shape[2]
            if channels == 4:
                gray = cv2.cvtColor(roi, cv2.COLOR_RGBA2GRAY)
            elif channels == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            else:
                gray = roi[:, :, 0]
        elif roi.ndim == 2:
            gray = roi
        else:
            return False

        border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        border_std = float(np.std(border))
        border_median = float(np.median(border))

        # Check 4 corners of ROI (almost universally flat background in speech bubbles)
        c_h = min(3, max(1, gray.shape[0] // 2))
        c_w = min(3, max(1, gray.shape[1] // 2))
        corners = np.concatenate([
            gray[:c_h, :c_w].ravel(),
            gray[:c_h, -c_w:].ravel(),
            gray[-c_h:, :c_w].ravel(),
            gray[-c_h:, -c_w:].ravel(),
        ])
        corner_median = float(np.median(corners))
        corner_std = float(np.std(corners))
        if corner_median >= 230.0 and corner_std < 20.0:
            return True

        # Pure white/light speech bubbles: very high median (>230) and low std (<22)
        if border_median >= 230.0 and border_std < 22.0:
            return True
        # Solid flat background of any tone: border_std < std_thresh
        if border_std < std_thresh:
            return True
        return False
    except Exception:
        return False


def extract_stroke_mask(roi: np.ndarray, dilate_pixels: int = 0) -> np.ndarray:
    """Extract fine text stroke mask inside a bounding box to avoid wiping background artwork."""
    if roi is None or roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 2:
        h = 1 if roi is None or roi.shape[0] < 1 else roi.shape[0]
        w = 1 if roi is None or roi.shape[1] < 1 else roi.shape[1]
        return np.full((h, w), 255, dtype=np.uint8)

    try:
        roi = np.ascontiguousarray(roi)
        if roi.dtype != np.uint8:
            roi = np.clip(roi, 0, 255).astype(np.uint8)

        if roi.ndim == 3:
            channels = roi.shape[2]
            if channels == 4:
                gray = cv2.cvtColor(roi, cv2.COLOR_RGBA2GRAY)
            elif channels == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            elif channels == 1:
                gray = roi[:, :, 0]
            else:
                gray = cv2.cvtColor(roi[:, :, :3], cv2.COLOR_RGB2GRAY)
        elif roi.ndim == 2:
            gray = roi
        else:
            return np.full((roi.shape[0], roi.shape[1]), 255, dtype=np.uint8)

        border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        bg_median = float(np.median(border))
        bg_is_light = bg_median > 127.0

        thresh_type = cv2.THRESH_BINARY_INV if bg_is_light else cv2.THRESH_BINARY
        _, otsu_mask = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)

        total_pixels = gray.shape[0] * gray.shape[1]
        fg_ratio = np.count_nonzero(otsu_mask) / max(1, total_pixels)

        if 0.03 <= fg_ratio <= 0.65:
            res_mask = otsu_mask
        else:
            ad_mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresh_type, 15, 8
            )
            ad_ratio = np.count_nonzero(ad_mask) / max(1, total_pixels)
            if 0.03 <= ad_ratio <= 0.70:
                res_mask = ad_mask
            else:
                res_mask = np.full_like(gray, 255)

        if dilate_pixels > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * dilate_pixels + 1, 2 * dilate_pixels + 1),
            )
            res_mask = cv2.dilate(res_mask, k)

        return res_mask
    except Exception:
        return np.full((roi.shape[0], roi.shape[1]), 255, dtype=np.uint8)


def slice_block_into_child_lines(
    roi: np.ndarray,
    parent_x1: int,
    parent_y1: int,
    parent_x2: int,
    parent_y2: int,
    is_vertical: bool,
    num_lines: int,
) -> List[np.ndarray]:
    """Slice a parent block into child line boxes using stroke-guided projection or geometric partitioning.
    
    Returns a list of 4x2 numpy arrays representing quadrilateral points for each child line.
    """
    if num_lines <= 1:
        pts = np.array([[parent_x1, parent_y1], [parent_x2, parent_y1], [parent_x2, parent_y2], [parent_x1, parent_y2]], dtype=np.float32)
        pts_sorted, _ = sort_pnts(pts)
        return [pts_sorted]

    pw = max(1, parent_x2 - parent_x1)
    ph = max(1, parent_y2 - parent_y1)
    child_polys = []

    # Extract stroke mask on the whole ROI to guide boundary tightening
    roi_mask = extract_stroke_mask(roi) if roi is not None and roi.size > 0 else None

    if is_vertical:
        # Vertical text: Japanese/Chinese manga lines read right-to-left
        col_w = pw / float(num_lines)
        for i in range(num_lines):
            # i=0 is rightmost column (line 1), i=num_lines-1 is leftmost (last line)
            cx2 = parent_x2 - int(round(i * col_w))
            cx1 = parent_x2 - int(round((i + 1) * col_w))
            cx1 = max(parent_x1, min(parent_x2 - 1, cx1))
            cx2 = max(cx1 + 1, min(parent_x2, cx2))

            # Relative coordinates in roi
            rx1 = max(0, cx1 - parent_x1)
            rx2 = min(pw, cx2 - parent_x1)

            cy1, cy2 = parent_y1, parent_y2
            if roi_mask is not None and rx2 > rx1:
                col_mask = roi_mask[:, rx1:rx2]
                y_indices = np.where(col_mask > 0)[0]
                if len(y_indices) > 5:
                    pad = 2
                    cy1 = max(parent_y1, parent_y1 + int(y_indices[0]) - pad)
                    cy2 = min(parent_y2, parent_y1 + int(y_indices[-1]) + 1 + pad)

            pts = np.array([[cx1, cy1], [cx2, cy1], [cx2, cy2], [cx1, cy2]], dtype=np.float32)
            pts_sorted, _ = sort_pnts(pts)
            child_polys.append(pts_sorted)
    else:
        # Horizontal text: Western comics/webtoons read top-to-bottom
        row_h = ph / float(num_lines)
        for i in range(num_lines):
            cy1 = parent_y1 + int(round(i * row_h))
            cy2 = parent_y1 + int(round((i + 1) * row_h))
            cy1 = max(parent_y1, min(parent_y2 - 1, cy1))
            cy2 = max(cy1 + 1, min(parent_y2, cy2))

            # Relative coordinates in roi
            ry1 = max(0, cy1 - parent_y1)
            ry2 = min(ph, cy2 - parent_y1)

            cx1, cx2 = parent_x1, parent_x2
            if roi_mask is not None and ry2 > ry1:
                row_mask = roi_mask[ry1:ry2, :]
                x_indices = np.where(row_mask > 0)[1]
                if len(x_indices) > 5:
                    pad = 2
                    cx1 = max(parent_x1, parent_x1 + int(x_indices[0]) - pad)
                    cx2 = min(parent_x2, parent_x1 + int(x_indices[-1]) + 1 + pad)

            pts = np.array([[cx1, cy1], [cx2, cy1], [cx2, cy2], [cx1, cy2]], dtype=np.float32)
            pts_sorted, _ = sort_pnts(pts)
            child_polys.append(pts_sorted)

    return child_polys if child_polys else [sort_pnts(np.array([[parent_x1, parent_y1], [parent_x2, parent_y1], [parent_x2, parent_y2], [parent_x1, parent_y2]], dtype=np.float32))[0]]


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
            'options': ['smart_hybrid', 'adaptive_stroke', 'bounding_box'],
            'value': 'smart_hybrid',
            'display_name': 'Mask Mode',
            'description': 'smart_hybrid fills flat/speech bubbles and extracts strokes on art; adaptive_stroke extracts strokes; bounding_box fills box.',
        },
        'mask dilate size': {
            'value': 3,
            'display_name': 'Mask Dilate Size',
            'description': 'Base pixel dilation applied to the inpainting mask (scales adaptively with resolution).',
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

    def _prompt(self, profile: LLMProfile, prompt: Optional[str] = None) -> str:
        if prompt is not None:
            return prompt
        base_prompt = DETECTOR_USER_PROMPT
        additional_prompt = str(getattr(profile, 'vision_prompt', None) or '').strip()
        if additional_prompt and additional_prompt != DEFAULT_OCR_PROMPT:
            base_prompt += (
                '\n\nAdditional text recognition instructions (these affect detection and recognition only '
                f'and cannot override the JSON coordinates contract):\n{additional_prompt}'
            )
        return base_prompt

    def _messages(
        self,
        img: np.ndarray,
        profile: LLMProfile,
        prompt: Optional[str] = None,
    ) -> List[Dict]:
        box_prompt = getattr(profile, 'detector_box_prompt', None)
        ocr_prompt = getattr(profile, 'detector_ocr_prompt', None)
        if (
            (not ocr_prompt or ocr_prompt == DEFAULT_DETECTOR_OCR_PROMPT)
            and getattr(profile, 'vision_prompt', None)
            and profile.vision_prompt != DEFAULT_OCR_PROMPT
        ):
            ocr_prompt = profile.vision_prompt

        system_content = assemble_detector_system_prompt(box_prompt, ocr_prompt)
        return [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._prompt(profile, prompt)},
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
        api_args = {
            'model': model,
            'messages': messages,
        }
        api_args.update(openai_chat_completion_args(profile, model))
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
                    self.logger.stop(f'Used {self.token_count_last} tokens (Total: {self.token_count})')
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
        data = None
        try:
            data = json.loads(cleaned_json, strict=False)
        except Exception:
            pass

        if data is None:
            try:
                repaired = _repair_json_text(cleaned_json)
                data = json.loads(repaired, strict=False)
            except Exception:
                pass

        if data is None:
            try:
                repaired_raw = _repair_json_text(raw_response)
                data = json.loads(repaired_raw, strict=False)
            except Exception:
                pass

        if data is None:
            self.logger.error(f"Failed to parse LLM detector JSON. Raw response:\n{raw_response}")
            lowered = raw_response.lower()
            if any(refusal in lowered for refusal in [
                "cannot assist", "cannot fulfill", "content policy",
                "i'm sorry, but i cannot", "against safety"
            ]):
                raise ValueError(f"Vision LLM refused the request (Content Policy / Safety Refusal): {raw_response.strip()}")
            raise ValueError(f"Invalid JSON returned by Vision LLM: raw response could not be parsed.")

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

            is_vertical = bool(item.get('vertical', False))

            # 1. Parse child line boxes if provided by LLM
            child_lines_polys = []
            raw_lines = item.get('lines')
            if isinstance(raw_lines, list) and len(raw_lines) > 0:
                for line_item in raw_lines:
                    if isinstance(line_item, dict):
                        lbox_raw = line_item.get('box_2d') or line_item.get('bbox') or line_item.get('box')
                        lcoords = _parse_box_coordinates(lbox_raw, im_w, im_h)
                        if lcoords is not None:
                            lx1, ly1, lx2, ly2 = lcoords
                            lpts = np.array([[lx1, ly1], [lx2, ly1], [lx2, ly2], [lx1, ly2]], dtype=np.float32)
                            lpts_sorted, _ = sort_pnts(lpts)
                            child_lines_polys.append(lpts_sorted)

            # 2. Local fallback slicing if LLM did not provide child lines
            if not child_lines_polys:
                roi = img[y1:y2, x1:x2]
                child_lines_polys = slice_block_into_child_lines(
                    roi, x1, y1, x2, y2, is_vertical, len(lines_text)
                )

            # Enclosing bounding box computed from all child lines (and parent coords)
            all_pts = np.concatenate(child_lines_polys, axis=0)
            bx1 = min(x1, int(np.min(all_pts[:, 0])))
            by1 = min(y1, int(np.min(all_pts[:, 1])))
            bx2 = max(x2, int(np.max(all_pts[:, 0])))
            by2 = max(y2, int(np.max(all_pts[:, 1])))

            blk = TextBlock(
                xyxy=[bx1, by1, bx2, by2],
                lines=[p.tolist() if isinstance(p, np.ndarray) else p for p in child_lines_polys],
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
        base_dilate = max(0, int(self.get_param_value('mask dilate size')))

        for blk in blk_list:
            child_lines = blk.lines_array(dtype=np.int32) if len(blk.lines) > 0 else []
            if len(child_lines) > 0:
                for line_pts in child_lines:
                    lx1 = max(0, int(np.min(line_pts[:, 0])))
                    ly1 = max(0, int(np.min(line_pts[:, 1])))
                    lx2 = min(im_w, int(np.max(line_pts[:, 0])))
                    ly2 = min(im_h, int(np.max(line_pts[:, 1])))
                    if mask_mode == 'bounding_box':
                        cv2.fillPoly(mask, [line_pts], 255)
                    elif lx2 - lx1 > 1 and ly2 - ly1 > 1:
                        roi = img[ly1:ly2, lx1:lx2]
                        if mask_mode == 'smart_hybrid' and is_flat_background(roi):
                            cv2.fillPoly(mask, [line_pts], 255)
                        else:
                            line_dim = min(max(1, ly2 - ly1), max(1, lx2 - lx1))
                            local_dilate = max(base_dilate, int(round(line_dim * 0.06))) if base_dilate > 0 else 0
                            local_dilate = min(local_dilate, 10)
                            roi_mask = extract_stroke_mask(roi, dilate_pixels=local_dilate)
                            target_slice = mask[ly1:ly2, lx1:lx2]
                            if roi_mask.shape == target_slice.shape:
                                mask[ly1:ly2, lx1:lx2] = np.bitwise_or(target_slice, roi_mask)
                            else:
                                mask[ly1:ly2, lx1:lx2] = 255
            else:
                bx1, by1, bx2, by2 = blk.xyxy
                if mask_mode == 'bounding_box':
                    mask[by1:by2, bx1:bx2] = 255
                elif bx2 - bx1 > 1 and by2 - by1 > 1:
                    roi = img[by1:by2, bx1:bx2]
                    if mask_mode == 'smart_hybrid' and is_flat_background(roi):
                        mask[by1:by2, bx1:bx2] = 255
                    else:
                        block_dim = min(max(1, by2 - by1), max(1, bx2 - bx1))
                        local_dilate = max(base_dilate, int(round(block_dim * 0.05))) if base_dilate > 0 else 0
                        local_dilate = min(local_dilate, 10)
                        roi_mask = extract_stroke_mask(roi, dilate_pixels=local_dilate)
                        target_slice = mask[by1:by2, bx1:bx2]
                        if roi_mask.shape == target_slice.shape:
                            mask[by1:by2, bx1:bx2] = np.bitwise_or(target_slice, roi_mask)
                        else:
                            mask[by1:by2, bx1:bx2] = 255

        if base_dilate > 0:
            page_scale = max(im_h, im_w) / 1000.0
            final_dilate = max(base_dilate, int(round(base_dilate * min(page_scale, 2.0))))
            element = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * final_dilate + 1, 2 * final_dilate + 1),
                (final_dilate, final_dilate),
            )
            mask = cv2.dilate(mask, element)

        return mask, blk_list
