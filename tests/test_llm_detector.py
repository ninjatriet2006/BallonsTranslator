import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ballontranslator.modules.textdetector.base import TEXTDETECTORS
from ballontranslator.modules.textdetector.detector_llm import (
    LLMDetector,
    _clean_json_text,
    _repair_json_text,
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

    def test_repair_json_text_unclosed_brackets_and_user_error(self):
        # Exact payload from user log missing the final closing brace '}'
        broken_user_payload = (
            '{"blocks": [{"box_2d": [10, 290, 210, 850], "text": "嗯~~（深呼吸）\\n嗯来了！啊..啊啊~~", "vertical": false, '
            '"lines": [{"box_2d": [20, 410, 70, 630], "text": "嗯~~（深呼吸）"}, {"box_2d": [90, 410, 140, 700], "text": "嗯来了！啊..啊啊~~"}]}, '
            '{"box_2d": [900, 290, 990, 450], "text": "无极八荒\\nPMY 116337839 twitter@TJBH91", "vertical": false, '
            '"lines": [{"box_2d": [900, 290, 950, 450], "text": "无极八荒"}, {"box_2d": [950, 290, 990, 450], "text": "PMY 116337839 twitter@TJBH91"}]}]'
        )
        repaired = _repair_json_text(broken_user_payload)
        parsed = json.loads(repaired, strict=False)
        self.assertEqual(len(parsed["blocks"]), 2)
        self.assertEqual(parsed["blocks"][0]["lines"][0]["text"], "嗯~~（深呼吸）")
        self.assertEqual(parsed["blocks"][1]["text"], "无极八荒\nPMY 116337839 twitter@TJBH91")

        # Trailing comma before closing brackets
        trailing_comma = '{"blocks": [{"box_2d": [1, 2, 3, 4], "text": "hi",},],}'
        repaired_comma = _repair_json_text(trailing_comma)
        parsed_comma = json.loads(repaired_comma, strict=False)
        self.assertEqual(len(parsed_comma["blocks"]), 1)

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

    def test_api_args_contains_model_and_messages(self):
        profile = LLMProfile(
            id='test_vision_profile',
            name='Test Vision',
            support_vision=True,
            vision_model='qwen2.5-vl-72b-instruct',
            vision_model_options=['qwen2.5-vl-72b-instruct'],
        )
        pcfg.module.llm_profiles = [profile]
        pcfg.module.ocr_llm_id = 'test_vision_profile'
        pcfg.module.detector_llm_id = ''

        detector = LLMDetector()
        messages = [{'role': 'user', 'content': 'test'}]
        args = detector._api_args(detector.profile, messages, None)

        self.assertIn('model', args)
        self.assertEqual(args['model'], 'qwen2.5-vl-72b-instruct')
        self.assertIn('messages', args)
        self.assertEqual(args['messages'], messages)

    def test_custom_vision_prompt_incorporated(self):
        from ballontranslator.modules.textdetector.detector_llm import DETECTOR_USER_PROMPT
        profile = LLMProfile(
            id='test_vision_profile',
            name='Test Vision',
            support_vision=True,
            vision_model='qwen2.5-vl-72b-instruct',
            vision_model_options=['qwen2.5-vl-72b-instruct'],
            vision_prompt='Ignore tiny sound effects and watermarks.',
        )
        pcfg.module.llm_profiles = [profile]
        pcfg.module.ocr_llm_id = 'test_vision_profile'
        pcfg.module.detector_llm_id = ''

        detector = LLMDetector()
        prompt = detector._prompt(profile)

        self.assertTrue(prompt.startswith(DETECTOR_USER_PROMPT))
        self.assertIn('Ignore tiny sound effects and watermarks.', prompt)
        self.assertIn('Additional text recognition instructions', prompt)

        # Explicit prompt overrides everything
        self.assertEqual(detector._prompt(profile, 'override'), 'override')

    def test_detector_prompts_assembled(self):
        from ballontranslator.modules.textdetector.detector_llm import (
            assemble_detector_system_prompt,
            DETECTOR_USER_PROMPT,
        )
        # Default prompt contains key guidelines and technical schema
        default_sys = assemble_detector_system_prompt()
        self.assertIn("Detection Guidelines:", default_sys)
        self.assertIn("Transcription Guidelines:", default_sys)
        self.assertIn("Output Format & Technical Rules:", default_sys)
        self.assertIn("box_2d", default_sys)
        self.assertIn("0 to 1000", default_sys)
        self.assertIn("lines", default_sys)

        # Custom prompts are cleanly injected into guidelines while keeping technical rules
        custom_box = "Detect only big speech bubbles and ignore all SFX."
        custom_ocr = "Transcribe with kanji and furigana separated by space."
        custom_sys = assemble_detector_system_prompt(custom_box, custom_ocr)
        self.assertIn(custom_box, custom_sys)
        self.assertIn(custom_ocr, custom_sys)
        self.assertIn("box_2d", custom_sys)
        self.assertIn("Output Format & Technical Rules:", custom_sys)

        # Verify LLMDetector._messages integrates custom profile prompts
        profile = LLMProfile(
            id='test_det_profile',
            name='Test Det Profile',
            support_vision=True,
            vision_model='test-model',
            vision_model_options=['test-model'],
            detector_box_prompt=custom_box,
            detector_ocr_prompt=custom_ocr,
        )
        detector = LLMDetector()
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        messages = detector._messages(dummy_img, profile)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn(custom_box, messages[0]["content"])
        self.assertIn(custom_ocr, messages[0]["content"])
        self.assertIn("0 to 1000", messages[0]["content"])

    def test_llm_child_lines_explicit(self):
        """When LLM provides child line boxes, blk.lines must use them."""
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
                    "text": "Line 1\nLine 2",
                    "vertical": False,
                    "lines": [
                        {"box_2d": [100, 200, 180, 500], "text": "Line 1"},
                        {"box_2d": [220, 200, 300, 500], "text": "Line 2"},
                    ]
                }
            ]
        })

        test_img = np.full((1000, 1000, 3), 255, dtype=np.uint8)

        with patch.object(detector, '_request_with_retries', return_value=mock_llm_response):
            mask, blk_list = detector._detect(test_img)

        self.assertEqual(len(blk_list), 1)
        blk = blk_list[0]
        # Must have exactly 2 child lines in blk.lines!
        self.assertEqual(len(blk.lines), 2)

    def test_local_slicing_fallback_multiline(self):
        """When LLM does not provide lines, multiline text must be sliced into child boxes automatically."""
        from ballontranslator.modules.textdetector.detector_llm import slice_block_into_child_lines

        roi = np.full((100, 100, 3), 255, dtype=np.uint8)
        # Horizontal slicing: 3 lines
        child_polys_h = slice_block_into_child_lines(roi, 10, 10, 110, 110, is_vertical=False, num_lines=3)
        self.assertEqual(len(child_polys_h), 3)

        # Vertical slicing: 2 lines
        child_polys_v = slice_block_into_child_lines(roi, 10, 10, 110, 110, is_vertical=True, num_lines=2)
        self.assertEqual(len(child_polys_v), 2)

    def test_child_box_inpaint_mask_preserves_gaps(self):
        """Verify that inpaint mask with child boxes leaves gaps between lines clean."""
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
        detector.params['mask mode']['value'] = 'bounding_box'
        detector.params['mask dilate size']['value'] = 0

        # Two separate child lines with a gap between y=180 and y=220
        mock_llm_response = json.dumps({
            "blocks": [
                {
                    "box_2d": [100, 200, 300, 500],
                    "text": "Line 1\nLine 2",
                    "vertical": False,
                    "lines": [
                        {"box_2d": [100, 200, 180, 500]},
                        {"box_2d": [220, 200, 300, 500]},
                    ]
                }
            ]
        })

        test_img = np.full((1000, 1000, 3), 255, dtype=np.uint8)

        with patch.object(detector, '_request_with_retries', return_value=mock_llm_response):
            mask, blk_list = detector._detect(test_img)

        # Pixels inside child line 1 (y=150, x=300) must be 255
        self.assertEqual(mask[150, 300], 255)
        # Pixels inside child line 2 (y=250, x=300) must be 255
        self.assertEqual(mask[250, 300], 255)
        # The GAP between the two lines (y=200, x=300) must be 0 (preserved background artwork)!
        self.assertEqual(mask[200, 300], 0)


if __name__ == '__main__':
    unittest.main()
