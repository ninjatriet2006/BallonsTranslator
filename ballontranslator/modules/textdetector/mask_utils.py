"""Computer vision mask and geometry utilities for text detectors and verifiers."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import cv2
import numpy as np

from ballontranslator.utils.text.textblock import sort_pnts


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
    if inter_area <= 0.0:
        return 0.0

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def compute_box_iom(
    box1: Sequence[float],
    box2: Sequence[float],
) -> float:
    """Compute Intersection over Minimum (IoM) of two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    min_area = min(area1, area2)
    if min_area <= 0.0:
        return 0.0
    return inter_area / min_area


def is_flat_background(roi: np.ndarray, std_thresh: float = 14.0) -> bool:
    """Check if the outer perimeter and corners of an ROI indicate a flat/solid background."""
    if roi is None or roi.size == 0 or roi.shape[0] < 4 or roi.shape[1] < 4:
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
            else:
                gray = roi[:, :, 0]
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
    pw = max(1, parent_x2 - parent_x1)
    ph = max(1, parent_y2 - parent_y1)
    child_polys = []

    # Extract stroke mask on the whole ROI to guide boundary tightening
    roi_mask = extract_stroke_mask(roi) if roi is not None and roi.size > 0 else None

    if num_lines <= 1:
        if roi_mask is not None and roi_mask.size > 0:
            stroke_y, stroke_x = np.where(roi_mask > 0)
            if len(stroke_y) > 10:
                pad = 3
                sy1 = max(parent_y1, parent_y1 + int(np.min(stroke_y)) - pad)
                sy2 = min(parent_y2, parent_y1 + int(np.max(stroke_y)) + 1 + pad)
                sx1 = max(parent_x1, parent_x1 + int(np.min(stroke_x)) - pad)
                sx2 = min(parent_x2, parent_x1 + int(np.max(stroke_x)) + 1 + pad)
                if (sy2 - sy1) >= 4 and (sx2 - sx1) >= 4:
                    pts = np.array([[sx1, sy1], [sx2, sy1], [sx2, sy2], [sx1, sy2]], dtype=np.float32)
                    pts_sorted, _ = sort_pnts(pts)
                    return [pts_sorted]
        pts = np.array([[parent_x1, parent_y1], [parent_x2, parent_y1], [parent_x2, parent_y2], [parent_x1, parent_y2]], dtype=np.float32)
        pts_sorted, _ = sort_pnts(pts)
        return [pts_sorted]

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
