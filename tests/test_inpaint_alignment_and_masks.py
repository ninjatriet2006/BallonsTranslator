import unittest
import numpy as np
import cv2
import json
from unittest.mock import patch

from ballontranslator.modules.inpaint.base import filter_mask_by_bboxes
from ballontranslator.modules.textdetector import TextBlock
from ballontranslator.modules.textdetector.llm.detector_llm import (
    is_flat_background,
    extract_stroke_mask,
    LLMDetector,
)
from ballontranslator.modules.llm_image import (
    letterbox_image,
    unletterbox_image,
)
from ballontranslator.utils.project.config import pcfg
from ballontranslator.utils.project.llm_profiles import LLMProfile


class InpaintAlignmentAndMaskTests(unittest.TestCase):
    def test_filter_mask_by_bboxes_preserves_margin(self):
        """Verify that filter_mask_by_bboxes does not clip descenders/ascenders within margin."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Bounding box is from x=20..60, y=20..60
        blk = TextBlock(xyxy=[20, 20, 60, 60])

        # A descender pixel at y=65, x=40 (5px outside bottom edge of bbox)
        mask[65, 40] = 255
        # A distant pixel at y=95, x=95 (far outside)
        mask[95, 95] = 255

        # With default margin=10, the descender at y=65 must be preserved
        filtered = filter_mask_by_bboxes(mask, [blk], margin=10)
        self.assertEqual(filtered[65, 40], 255)
        # Distant pixel must be filtered out
        self.assertEqual(filtered[95, 95], 0)

    def test_is_flat_background_detection(self):
        """Verify is_flat_background correctly detects flat balloons and avoids artwork."""
        # 1. White speech bubble with black text in center
        white_roi = np.full((50, 50, 3), 255, dtype=np.uint8)
        white_roi[15:35, 15:35] = 0
        self.assertTrue(is_flat_background(white_roi))

        # 2. Solid dark gray box
        gray_roi = np.full((40, 40, 3), 50, dtype=np.uint8)
        gray_roi[10:30, 10:30] = 240
        self.assertTrue(is_flat_background(gray_roi))

        # 3. High variance texture / artwork (random noise or strong gradient on border)
        textured_roi = np.zeros((60, 60, 3), dtype=np.uint8)
        # Create non-uniform border
        textured_roi[0, :, :] = np.linspace(0, 255, 60)[:, None]
        textured_roi[-1, :, :] = np.linspace(255, 0, 60)[:, None]
        textured_roi[:, 0, :] = 120
        textured_roi[:, -1, :] = 20
        self.assertFalse(is_flat_background(textured_roi))

    def test_extract_stroke_mask_with_dilation(self):
        """Verify extract_stroke_mask dilates detected strokes when dilate_pixels > 0."""
        # White background with a 5x20 text stroke at (18..23, 10..30)
        roi = np.full((40, 40, 3), 255, dtype=np.uint8)
        roi[18:23, 10:30] = 0

        mask_raw = extract_stroke_mask(roi, dilate_pixels=0)
        mask_dilated = extract_stroke_mask(roi, dilate_pixels=3)

        self.assertGreater(
            np.count_nonzero(mask_dilated),
            np.count_nonzero(mask_raw),
        )

    def test_smart_hybrid_fills_flat_balloon_and_strokes_textured(self):
        """Verify smart_hybrid mode fills entire line box on flat balloon, but extracts strokes on texture."""
        profile = LLMProfile(
            id='test_profile',
            name='Test Profile',
            support_vision=True,
            vision_model='test-model',
            vision_model_options=['test-model'],
        )
        pcfg.module.llm_profiles = [profile]
        pcfg.module.ocr_llm_id = 'test_profile'
        pcfg.module.detector_llm_id = ''

        detector = LLMDetector()
        detector.params['mask mode']['value'] = 'smart_hybrid'
        detector.params['mask dilate size']['value'] = 0

        # An image with a flat white bubble at top, and a textured area at bottom
        img = np.zeros((600, 600, 3), dtype=np.uint8)
        # Top half: pure white background
        img[0:300, :] = 255
        # Put black text at top: y=50..70, x=120..180 (centered inside y=40..80, x=100..220)
        img[50:70, 120:180] = 0

        # Bottom half: complex gradient texture
        for y in range(300, 600):
            img[y, :, 0] = (y * 3) % 256
            img[y, :, 1] = (y * 7) % 256
            img[y, :, 2] = (y * 11) % 256
        # Put white text at bottom: y=400..420, x=100..200
        img[400:420, 100:200] = 255

        mock_response = json.dumps({
            "blocks": [
                {
                    "box_2d": [67, 167, 133, 367],  # normalized 0-1000 for y=40..80, x=100..220
                    "text": "Top Bubble",
                    "vertical": False,
                    "lines": [{"box_2d": [67, 167, 133, 367]}]
                },
                {
                    "box_2d": [650, 167, 717, 367],  # normalized for y=390..430, x=100..220
                    "text": "Bottom Art",
                    "vertical": False,
                    "lines": [{"box_2d": [650, 167, 717, 367]}]
                }
            ]
        })

        with patch.object(detector, '_request_with_retries', return_value=mock_response):
            mask, _ = detector._detect(img)

        # On top flat balloon, entire line box (including white space near edge) must be 255
        # Check a pixel inside the box where there was white background (e.g. y=45, x=105)
        top_y1 = int(round(67 * 600 / 1000.0))
        top_x1 = int(round(167 * 600 / 1000.0))
        self.assertEqual(mask[top_y1 + 2, top_x1 + 2], 255)

        # On bottom textured background, background pixels inside the box must NOT be filled completely
        bot_y1 = int(round(650 * 600 / 1000.0))
        bot_y2 = int(round(717 * 600 / 1000.0))
        bot_x1 = int(round(167 * 600 / 1000.0))
        bot_x2 = int(round(367 * 600 / 1000.0))
        bot_roi_mask = mask[bot_y1:bot_y2, bot_x1:bot_x2]
        # Not completely filled: has 0s where texture was preserved
        self.assertTrue(np.any(bot_roi_mask == 0))

    def test_letterbox_and_unletterbox_preserves_aspect_ratio_and_coordinates(self):
        """Verify letterbox_image and unletterbox_image eliminate distortion and preserve exact alignment."""
        # Non-square image: height 120, width 300
        orig_h, orig_w = 120, 300
        orig_img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        # Draw a specific feature at (y=40..60, x=100..150)
        orig_img[40:60, 100:150] = 200

        padded, pad_info = letterbox_image(orig_img)
        pad_top, pad_bottom, pad_left, pad_right = pad_info

        # Padded must be perfectly square 300x300
        self.assertEqual(padded.shape[:2], (300, 300))
        self.assertEqual(pad_top + orig_h + pad_bottom, 300)
        self.assertEqual(pad_left + orig_w + pad_right, 300)
        self.assertEqual(pad_left, 0)
        self.assertEqual(pad_right, 0)
        self.assertEqual(pad_top, 90)
        self.assertEqual(pad_bottom, 90)

        # Simulate API generating a 1024x1024 square image preserving the feature
        simulated_api_square = cv2.resize(padded, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        # Unletterbox back to original shape
        unpadded = unletterbox_image(simulated_api_square, pad_info, (orig_h, orig_w))

        # Must match original dimensions exactly
        self.assertEqual(unpadded.shape[:2], (orig_h, orig_w))

        # The feature at (y=50, x=125) must be preserved in place
        self.assertEqual(unpadded[50, 125, 0], 200)
        # Background pixel outside the feature (y=10, x=10) must be 0
        self.assertEqual(unpadded[10, 10, 0], 0)

    def test_square_letterbox_pass_through(self):
        """Verify square image passes through without padding."""
        img = np.ones((256, 256, 3), dtype=np.uint8)
        padded, pad_info = letterbox_image(img)
        self.assertEqual(pad_info, (0, 0, 0, 0))
        self.assertEqual(padded.shape, img.shape)


if __name__ == '__main__':
    unittest.main()
