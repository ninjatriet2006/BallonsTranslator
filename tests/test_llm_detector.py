import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ballontranslator.modules.textdetector.base import TEXTDETECTORS
from ballontranslator.modules.textdetector.detector_llm import (
    LLMDetector,
    _clean_json_text,
    _parse_box_coordinates,
    extract_stroke_mask,
)
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import (
    LLM_DETECTOR_KEY,
    LLMProfile,
)
from ballontranslator.utils.textblock import TextBlock


class TestLLMDetector(unittest.TestCase):

    def test_clean_json_text(self):
        # Clean fenced markdown
        markdown = "```json\n{\"blocks\": []}\n```"
        self.assertEqual(_clean_json_text(markdown), "{\"blocks\": []}")

        # Text with preamble and postscript
        messy = "Here is the result:\n{\"blocks\": [{\"text\": \"hello\"}]}\nHope this helps!"
        self.assertEqual(_clean_json_text(messy), "{\"blocks\": [{\"text\": \"hello\"}]}")

        # Raw clean json
        clean = "{\"blocks\": []}"
        self.assertEqual(_clean_json_text(clean), clean)

    def test_parse_box_coordinates_normalized_1000(self):
        # [ymin, xmin, ymax, xmax] in 0-1000
        box = [100, 200, 500, 600]
        # Image size: w=1000, h=1000
        coords = _parse_box_coordinates(box, 1000, 1000)
        self.assertEqual(coords, (200, 100, 600, 500))

    def test_parse_box_coordinates_normalized_float(self):
        # [ymin, xmin, ymax, xmax] in 0.0-1.0
        box = [0.1, 0.2, 0.5, 0.6]
        coords = _parse_box_coordinates(box, 1000, 1000)
        self.assertEqual(coords, (200, 100, 600, 500))

    def test_parse_box_coordinates_dict(self):
        box_dict = {"ymin": 100, "xmin": 200, "ymax": 500, "xmax": 600}
        coords = _parse_box_coordinates(box_dict, 1000, 1000)
        self.assertEqual(coords, (200, 100, 600, 500))

    def test_parse_box_coordinates_invalid(self):
        # Degenerate boxes should return None
        self.assertIsNone(_parse_box_coordinates([100, 100, 100, 100], 1000, 1000))
        self.assertIsNone(_parse_box_coordinates("invalid", 1000, 1000))
        self.assertIsNone(_parse_box_coordinates([10, 10], 1000, 1000))

    def test_extract_stroke_mask(self):
        # White background with black text stroke in center
        roi = np.full((50, 50, 3), 255, dtype=np.uint8)
        roi[20:30, 20:30] = [0, 0, 0]

        mask = extract_stroke_mask(roi)
        self.assertEqual(mask.shape, (50, 50))
        self.assertEqual(mask.dtype, np.uint8)
        # Background perimeter should be 0, text region should be 255
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(mask[25, 25], 255)

    def test_extract_stroke_mask_dark_background(self):
        # Dark background with white text stroke in center
        roi = np.full((50, 50, 3), 0, dtype=np.uint8)
        roi[20:30, 20:30] = [255, 255, 255]

        mask = extract_stroke_mask(roi)
        self.assertEqual(mask.shape, (50, 50))
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(mask[25, 25], 255)

    def test_detector_registered(self):
        self.assertIn('llm_detector', TEXTDETECTORS.module_dict)
        self.assertEqual(TEXTDETECTORS.module_dict['llm_detector'], LLMDetector)

    def test_profile_resolution(self):
        profile = LLMProfile(
            id='test_vision_profile',
            name='Test Vision',
            support_vision=True,
            vision_model='gpt-4o',
            vision_model_options=['gpt-4o'],
        )
        pcfg.module.llm_profiles = [profile]
        pcfg.module.ocr_llm_id = 'test_vision_profile'
        pcfg.module.detector_llm_id = ''

        detector = LLMDetector()
        self.assertEqual(detector.profile.id, 'test_vision_profile')

        # Dedicated detector_llm_id takes priority if set
        profile2 = LLMProfile(
            id='test_detector_profile',
            name='Test Detector',
            support_vision=True,
            vision_model='gemini-2.0-flash',
            vision_model_options=['gemini-2.0-flash'],
        )
        pcfg.module.llm_profiles.append(profile2)
        pcfg.module.detector_llm_id = 'test_detector_profile'

        self.assertEqual(detector.profile.id, 'test_detector_profile')

    def test_detect_full_flow(self):
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

        mock_llm_response = json.dumps({
            "blocks": [
                {
                    "box_2d": [100, 200, 300, 500],
                    "text": "Hello world!\nSecond line",
                    "vertical": False,
                },
                {
                    "box_2d": [500, 600, 800, 900],
                    "text": "こんにちは",
                    "vertical": True,
                },
            ]
        })

        test_img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
        # Put some dark pixels in the text areas
        test_img[150:250, 250:450] = 0
        test_img[550:750, 650:850] = 0

        with patch.object(detector, '_request_with_retries', return_value=mock_llm_response):
            mask, blk_list = detector._detect(test_img)

        self.assertEqual(len(blk_list), 2)
        self.assertEqual(mask.shape, (1000, 1000))

        # Check block texts and flags
        texts = [b.get_text() for b in blk_list]
        self.assertTrue(any("Hello world!" in t for t in texts))
        self.assertTrue(any("こんにちは" in t for t in texts))

        # Check inpainting mask has non-zero pixels
        self.assertGreater(np.count_nonzero(mask), 0)


if __name__ == '__main__':
    unittest.main()
