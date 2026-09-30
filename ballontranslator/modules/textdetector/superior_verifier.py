"""Superior Verifier for Vision (Text Detection & OCR).

Combines LLM Vision Detector (rich semantic understanding, reading order, OCR)
with specialized Offline Detectors (CTD / DBNet / YOLO for sub-pixel boundary precision)
and an optional Verifier LLM arbitrator to produce high-precision, leak-free text blocks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ballontranslator.modules.textdetector.base import TEXTDETECTORS, TextBlock
from ballontranslator.modules.textdetector.llm.detector_llm import (
    extract_stroke_mask,
    is_flat_background,
    slice_block_into_child_lines,
)
from ballontranslator.utils.project.config import pcfg
from ballontranslator.utils.project.llm_profiles import LLMProfile, runtime_profile
from ballontranslator.utils.text.textblock import examine_textblk, sort_pnts, sort_regions

LOGGER = logging.getLogger(__name__)


def compute_box_iou(
    box1: Sequence[float],
    box2: Sequence[float],
) -> float:
    """Compute Intersection over Union (IoU) of two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))

    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_box_iom(
    box1: Sequence[float],
    box2: Sequence[float],
) -> float:
    """Compute Intersection over Minimum area (IoM) of two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    min_area = min(area1, area2)

    if min_area <= 0.0:
        return 0.0
    return inter_area / min_area


def match_boxes(
    llm_boxes: Sequence[Sequence[float]],
    offline_boxes: Sequence[Sequence[float]],
    iou_thresh: float = 0.35,
    iom_thresh: float = 0.55,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match LLM bounding boxes with Offline detector bounding boxes.
    
    Returns:
        matched_pairs: List of (llm_idx, offline_idx)
        unmatched_llm: List of llm_indices with no matching offline box
        unmatched_offline: List of offline_indices with no matching llm box
    """
    n_llm = len(llm_boxes)
    n_off = len(offline_boxes)

    if n_llm == 0:
        return [], [], list(range(n_off))
    if n_off == 0:
        return [], list(range(n_llm)), []

    # Score matrix combining IoU and IoM
    scores = np.zeros((n_llm, n_off), dtype=np.float32)
    for i in range(n_llm):
        for j in range(n_off):
            iou = compute_box_iou(llm_boxes[i], offline_boxes[j])
            iom = compute_box_iom(llm_boxes[i], offline_boxes[j])
            if iou >= iou_thresh or iom >= iom_thresh:
                scores[i, j] = max(iou, iom * 0.8)

    matched_pairs: List[Tuple[int, int]] = []
    matched_llm = set()
    matched_off = set()

    # Greedy match highest score pairs
    while True:
        max_idx = np.unravel_index(np.argmax(scores), scores.shape)
        max_val = scores[max_idx]
        if max_val <= 0.0:
            break
        i, j = int(max_idx[0]), int(max_idx[1])
        matched_pairs.append((i, j))
        matched_llm.add(i)
        matched_off.add(j)
        scores[i, :] = -1.0
        scores[:, j] = -1.0

    unmatched_llm = [i for i in range(n_llm) if i not in matched_llm]
    unmatched_offline = [j for j in range(n_off) if j not in matched_off]

    return matched_pairs, unmatched_llm, unmatched_offline


def refine_block_geometry(
    llm_blk: TextBlock,
    offline_blk: TextBlock,
    img: Optional[np.ndarray] = None,
    im_w: int = 1000,
    im_h: int = 1000,
) -> TextBlock:
    """Refine LLM TextBlock using the sub-pixel crisp boundaries of an offline TextBlock."""
    ox1, oy1, ox2, oy2 = offline_blk.xyxy
    is_vertical = getattr(llm_blk, 'vertical', False) or getattr(llm_blk, 'src_is_vertical', False)
    lines_text = llm_blk.text if isinstance(llm_blk.text, list) else [str(llm_blk.text)]

    # Use offline line geometry if available and matching line count
    if len(offline_blk.lines) > 0 and len(offline_blk.lines) == len(lines_text):
        refined_lines = [p.tolist() if isinstance(p, np.ndarray) else p for p in offline_blk.lines]
    elif img is not None and (ox2 - ox1) > 4 and (oy2 - oy1) > 4:
        roi = img[oy1:oy2, ox1:ox2]
        child_polys = slice_block_into_child_lines(
            roi, ox1, oy1, ox2, oy2, is_vertical, len(lines_text)
        )
        refined_lines = [p.tolist() if isinstance(p, np.ndarray) else p for p in child_polys]
    else:
        # Scale LLM child lines to fit inside offline boundaries
        lx1, ly1, lx2, ly2 = llm_blk.xyxy
        lw = max(1, lx2 - lx1)
        lh = max(1, ly2 - ly1)
        ow = max(1, ox2 - ox1)
        oh = max(1, oy2 - oy1)

        refined_lines = []
        for line in llm_blk.lines:
            pts = np.array(line, dtype=np.float32)
            if pts.ndim == 2 and pts.shape[0] >= 4 and pts.shape[1] == 2:
                # Normalize relative to old LLM box then scale to new offline box
                rel_x = (pts[:, 0] - lx1) / float(lw)
                rel_y = (pts[:, 1] - ly1) / float(lh)
                new_x = np.clip(ox1 + rel_x * ow, ox1, ox2)
                new_y = np.clip(oy1 + rel_y * oh, oy1, oy2)
                pts_scaled = np.stack([new_x, new_y], axis=1)
                sorted_pts, _ = sort_pnts(pts_scaled)
                refined_lines.append(sorted_pts.tolist())
            else:
                refined_lines.append(line)

        if not refined_lines:
            default_poly = np.array([[ox1, oy1], [ox2, oy1], [ox2, oy2], [ox1, oy2]], dtype=np.float32)
            refined_lines = [default_poly.tolist()]

    llm_blk.xyxy = [int(ox1), int(oy1), int(ox2), int(oy2)]
    llm_blk.lines = refined_lines
    llm_blk.adjust_bbox()
    examine_textblk(llm_blk, im_w, im_h)
    return llm_blk


def arbitrate_unmatched_offline_blocks(
    img: np.ndarray,
    unmatched_offline_blks: List[TextBlock],
    verifier_profile: Optional[LLMProfile] = None,
    stop_event: Optional[Any] = None,
) -> List[TextBlock]:
    """Arbitrate offline blocks that were missed by the primary LLM Detector.
    
    If verifier LLM profile is provided, queries the verifier to validate if the
    missed region contains real dialogue text or background artwork.
    """
    if not unmatched_offline_blks or img is None or img.size == 0:
        return []

    # If no verifier profile, filter for high-confidence speech bubbles
    if verifier_profile is None or not getattr(verifier_profile, 'support_vision', False):
        accepted_blks = []
        for blk in unmatched_offline_blks:
            x1, y1, x2, y2 = blk.xyxy
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                roi = img[y1:y2, x1:x2]
                if is_flat_background(roi):
                    # Flat white/colored speech bubble detected by offline model
                    if not blk.get_text().strip():
                        blk.text = ['...']
                    accepted_blks.append(blk)
        return accepted_blks

    # With verifier LLM available, inspect disputed regions
    try:
        from ballontranslator.modules.llm_chat import LLMChatRequester, openai_chat_completion_args
        from ballontranslator.modules.llm_vision import encode_chat_image

        accepted_blks = []
        requester = LLMChatRequester()

        for blk in unmatched_offline_blks:
            if stop_event is not None and stop_event.is_set():
                break

            x1, y1, x2, y2 = blk.xyxy
            pad = 6
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(img.shape[1], x2 + pad)
            cy2 = min(img.shape[0], y2 + pad)

            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                continue

            crop = img[cy1:cy2, cx1:cx2]
            image_part = encode_chat_image(
                crop,
                detail="low",
                failure_message="Failed to encode crop for verification.",
            ).image_part()

            prompt = (
                "Look at this cropped region from a manga/comic page. "
                "Does it contain legible speech, dialogue, or sound effects? "
                "If YES, transcribe the exact visible text. "
                "If NO (just blank art, drawing line, screen tone, or noise), reply strictly with 'NO_TEXT'."
            )

            messages = [
                {"role": "system", "content": "You are a manga OCR verifier. Transcribe text or say NO_TEXT."},
                {"role": "user", "content": [{"type": "text", "text": prompt}, image_part]},
            ]

            model = getattr(verifier_profile, 'vision_model', '') or 'default'
            args = {'model': model, 'messages': messages}
            args.update(openai_chat_completion_args(verifier_profile, model))

            completion = requester.request_chat_completion(verifier_profile, args)
            content = str(completion.content or '').strip()

            if content and 'NO_TEXT' not in content.upper() and len(content) > 0:
                lines = [l for l in content.splitlines() if l.strip()]
                blk.text = lines if lines else [content]
                accepted_blks.append(blk)
                LOGGER.info(f"[Superior Verify] Recovered missed text block: '{content}' at {blk.xyxy}")

        return accepted_blks
    except Exception as e:
        LOGGER.warning(f"[Superior Verify] Arbitration failed ({e}). Proceeding without verifier additions.")
        return []


def run_superior_verify(
    img: np.ndarray,
    llm_blks: List[TextBlock],
    offline_detector: Optional[Any] = None,
    verifier_profile: Optional[LLMProfile] = None,
    stop_event: Optional[Any] = None,
) -> Tuple[np.ndarray, List[TextBlock]]:
    """Execute Superior Verification combining LLM Detector with an Offline Detector.
    
    1. Runs the offline detector (CTD) on the original image.
    2. Matches bounding boxes using spatial IoU / IoM.
    3. Replaces approximate LLM box bounds with pixel-precise offline bounds.
    4. Arbitrates missed bubbles or false positives.
    5. Builds an updated, tight inpaint mask.
    
    Returns:
        (refined_mask, refined_blks)
    """
    im_h, im_w = img.shape[:2]
    if not llm_blks and (img is None or img.size == 0):
        return np.zeros((im_h, im_w), dtype=np.uint8), llm_blks

    offline_blks: List[TextBlock] = []
    
    # Run or obtain offline detector
    try:
        det_instance = offline_detector
        should_unload = False
        if det_instance is None:
            if 'ctd' in TEXTDETECTORS.module_dict:
                ctd_cls = TEXTDETECTORS.resolve_module('ctd')
                if ctd_cls is not None:
                    det_instance = ctd_cls()
                    det_instance.load_model()
                    should_unload = True
            elif 'ctbd' in TEXTDETECTORS.module_dict:
                ctbd_cls = TEXTDETECTORS.resolve_module('ctbd')
                if ctbd_cls is not None:
                    det_instance = ctbd_cls()
                    det_instance.load_model()
                    should_unload = True

        if det_instance is not None:
            _, offline_blks = det_instance.detect(img)
            LOGGER.info(f"[Superior Verify] Offline detector found {len(offline_blks)} candidate blocks.")
            if should_unload:
                try:
                    det_instance.unload_model(empty_cache=True)
                except Exception:
                    pass
    except Exception as e:
        LOGGER.warning(f"[Superior Verify] Offline detector execution failed: {e}. Falling back to LLM boxes.")
        offline_blks = []

    if not offline_blks:
        # Generate default mask from llm_blks
        mask = np.zeros((im_h, im_w), dtype=np.uint8)
        for blk in llm_blks:
            bx1, by1, bx2, by2 = blk.xyxy
            cv2.rectangle(mask, (bx1, by1), (bx2, by2), 255, -1)
        return mask, llm_blks

    # Match LLM boxes to Offline boxes
    llm_coords = [b.xyxy for b in llm_blks]
    off_coords = [b.xyxy for b in offline_blks]
    matched_pairs, unmatched_llm, unmatched_off = match_boxes(llm_coords, off_coords)

    LOGGER.info(
        f"[Superior Verify] Matching results: {len(matched_pairs)} matched pairs, "
        f"{len(unmatched_llm)} LLM-only, {len(unmatched_off)} Offline-only."
    )

    refined_blks: List[TextBlock] = []

    # 1. Refine matched pairs (take offline pixel bounds + LLM text and reading order)
    for l_idx, o_idx in matched_pairs:
        lblk = llm_blks[l_idx]
        oblk = offline_blks[o_idx]
        refined_blk = refine_block_geometry(lblk, oblk, img=img, im_w=im_w, im_h=im_h)
        refined_blks.append(refined_blk)

    # 2. Retain LLM-only blocks (important for SFX and stylized text)
    for l_idx in unmatched_llm:
        refined_blks.append(llm_blks[l_idx])

    # 3. Arbitrate offline-only blocks (recover missed speech bubbles)
    if unmatched_off:
        candidate_offline_blks = [offline_blks[j] for j in unmatched_off]
        recovered = arbitrate_unmatched_offline_blocks(
            img,
            candidate_offline_blks,
            verifier_profile=verifier_profile,
            stop_event=stop_event,
        )
        refined_blks.extend(recovered)

    # Sort blocks in standard reading order
    refined_blks = sort_regions(refined_blks)

    # Generate high-quality inpainting mask for refined blocks
    mask = np.zeros((im_h, im_w), dtype=np.uint8)
    base_dilate = max(0, int(getattr(pcfg.module, 'mask_dilate_size', 3) if hasattr(pcfg.module, 'mask_dilate_size') else 3))

    for blk in refined_blks:
        child_lines = blk.lines_array(dtype=np.int32) if len(blk.lines) > 0 else []
        if len(child_lines) > 0:
            for line_pts in child_lines:
                lx1 = max(0, int(np.min(line_pts[:, 0])))
                ly1 = max(0, int(np.min(line_pts[:, 1])))
                lx2 = min(im_w, int(np.max(line_pts[:, 0])))
                ly2 = min(im_h, int(np.max(line_pts[:, 1])))
                if lx2 - lx1 > 1 and ly2 - ly1 > 1:
                    roi = img[ly1:ly2, lx1:lx2]
                    if is_flat_background(roi):
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
            if bx2 - bx1 > 1 and by2 - by1 > 1:
                roi = img[by1:by2, bx1:bx2]
                if is_flat_background(roi):
                    cv2.rectangle(mask, (bx1, by1), (bx2, by2), 255, -1)
                else:
                    roi_mask = extract_stroke_mask(roi, dilate_pixels=base_dilate)
                    target_slice = mask[by1:by2, bx1:bx2]
                    if roi_mask.shape == target_slice.shape:
                        mask[by1:by2, bx1:bx2] = np.bitwise_or(target_slice, roi_mask)
                    else:
                        mask[by1:by2, bx1:bx2] = 255

    return mask, refined_blks
