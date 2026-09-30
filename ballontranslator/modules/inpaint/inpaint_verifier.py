"""Inpaint Verifier & Residual QA.

Audits inpainted images against original images to detect:
- Lingering text strokes / Ghost text
- Incomplete inpaint artifacts
And triggers an automatic Pass 2 cleanup with dilated masks to guarantee clean backgrounds.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ballontranslator.modules.textdetector.base import TextBlock

LOGGER = logging.getLogger(__name__)


def detect_inpaint_residuals(
    original_img: np.ndarray,
    inpainted_img: np.ndarray,
    mask: np.ndarray,
    blk_list: List[TextBlock],
    edge_thresh: float = 0.15,
) -> Tuple[bool, np.ndarray, List[int]]:
    """Detect un-erased text strokes (ghost text) in the inpainted image.
    
    Returns:
        (has_residuals, dilated_residual_mask, residual_block_indices)
    """
    if (
        original_img is None
        or inpainted_img is None
        or mask is None
        or not blk_list
        or original_img.shape != inpainted_img.shape
    ):
        return False, np.zeros_like(mask if mask is not None else np.zeros((1, 1), dtype=np.uint8)), []

    im_h, im_w = original_img.shape[:2]
    residual_mask = np.zeros((im_h, im_w), dtype=np.uint8)
    residual_indices: List[int] = []

    for idx, blk in enumerate(blk_list):
        bx1, by1, bx2, by2 = blk.xyxy
        # Add 4px margin
        x1 = max(0, bx1 - 4)
        y1 = max(0, by1 - 4)
        x2 = min(im_w, bx2 + 4)
        y2 = min(im_h, by2 + 4)

        if (x2 - x1) < 6 or (y2 - y1) < 6:
            continue

        mask_roi = mask[y1:y2, x1:x2]
        if np.count_nonzero(mask_roi) < 10:
            continue

        orig_roi = original_img[y1:y2, x1:x2]
        inp_roi = inpainted_img[y1:y2, x1:x2]

        orig_gray = cv2.cvtColor(orig_roi, cv2.COLOR_BGR2GRAY) if orig_roi.ndim == 3 else orig_roi
        inp_gray = cv2.cvtColor(inp_roi, cv2.COLOR_BGR2GRAY) if inp_roi.ndim == 3 else inp_roi

        # Detect high-frequency edges in masked area
        orig_edges = cv2.Canny(orig_gray, 50, 150)
        inp_edges = cv2.Canny(inp_gray, 50, 150)

        masked_orig_edges = np.count_nonzero((orig_edges > 0) & (mask_roi > 0))
        masked_inp_edges = np.count_nonzero((inp_edges > 0) & (mask_roi > 0))

        if masked_orig_edges >= 25:
            residual_ratio = masked_inp_edges / float(masked_orig_edges)
            if residual_ratio > edge_thresh:
                residual_indices.append(idx)
                residual_mask[y1:y2, x1:x2] = np.bitwise_or(
                    residual_mask[y1:y2, x1:x2],
                    mask_roi,
                )

    if not residual_indices:
        return False, residual_mask, []

    # Dilate residual mask by 5px to give inpainter sufficient context to obliterate ghost text
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated_mask = cv2.dilate(residual_mask, kernel)

    return True, dilated_mask, residual_indices


def run_inpaint_verify_and_refine(
    inpainter: Any,
    original_img: np.ndarray,
    inpainted_img: np.ndarray,
    mask: np.ndarray,
    blk_list: List[TextBlock],
    stop_event: Optional[Any] = None,
) -> np.ndarray:
    """Verify inpainting quality and automatically run Pass 2 on any lingering ghost text."""
    if inpainter is None or inpainted_img is None or mask is None:
        return inpainted_img

    try:
        has_residuals, residual_mask, bad_indices = detect_inpaint_residuals(
            original_img, inpainted_img, mask, blk_list
        )

        if not has_residuals:
            return inpainted_img

        if stop_event is not None and stop_event.is_set():
            return inpainted_img

        LOGGER.info(
            f"[Inpaint Verifier] Detected ghost text / residual edges in {len(bad_indices)} "
            f"blocks (indices: {bad_indices}). Running Pass 2 inpainting..."
        )

        pass2_result = inpainter.inpaint(inpainted_img, residual_mask, blk_list)
        if pass2_result is not None and pass2_result.shape == inpainted_img.shape:
            # Seamlessly update areas covered by residual mask
            refined = inpainted_img.copy()
            update_pixels = residual_mask > 0
            refined[update_pixels] = pass2_result[update_pixels]
            LOGGER.info(f"[Inpaint Verifier] Pass 2 inpainting completed successfully.")
            return refined

        return inpainted_img
    except Exception as e:
        LOGGER.warning(f"[Inpaint Verifier] Verification pass failed: {e}. Keeping Pass 1 result.")
        return inpainted_img
