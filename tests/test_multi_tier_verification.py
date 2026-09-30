"""Unit tests for Multi-tier Verification Architecture.

Pillar 1: Superior Verify for Vision (Detection & OCR)
Pillar 2: Normal Verify for Translation (Audit & Self-Correction)
Pillar 3: Normal Verify for Inpainting (Residual QA & Pass 2)
"""

import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from ballontranslator.modules.textdetector.base import TextBlock
from ballontranslator.modules.textdetector.superior_verifier import (
    compute_box_iou,
    compute_box_iom,
    match_boxes,
    refine_block_geometry,
    run_superior_verify,
)
from ballontranslator.modules.translators.translation_verifier import (
    audit_translation,
    is_repetitive_loop,
    self_correct_translation,
)
from ballontranslator.modules.inpaint.inpaint_verifier import (
    detect_inpaint_residuals,
    run_inpaint_verify_and_refine,
)


class TestSuperiorVisionVerifier(unittest.TestCase):

    def test_box_iou_and_iom_metrics(self):
        box_a = [100, 100, 200, 200]  # 100x100 = 10000
        box_b = [100, 100, 200, 200]
        self.assertAlmostEqual(compute_box_iou(box_a, box_b), 1.0)
        self.assertAlmostEqual(compute_box_iom(box_a, box_b), 1.0)

        # Disjoint boxes
        box_c = [300, 300, 400, 400]
        self.assertEqual(compute_box_iou(box_a, box_c), 0.0)
        self.assertEqual(compute_box_iom(box_a, box_c), 0.0)

        # Partial overlap (50x100 overlap)
        box_d = [150, 100, 250, 200]
        # Inter = 50 * 100 = 5000. Area1 = 10000, Area2 = 10000. Union = 15000.
        self.assertAlmostEqual(compute_box_iou(box_a, box_d), 5000.0 / 15000.0)
        self.assertAlmostEqual(compute_box_iom(box_a, box_d), 5000.0 / 10000.0)

    def test_box_matching_spatial(self):
        llm_boxes = [
            [95, 95, 205, 205],    # rough box around bubble 1
            [500, 500, 600, 600],  # SFX detected by LLM
        ]
        offline_boxes = [
            [100, 100, 200, 200],  # crisp bubble 1
            [800, 800, 900, 900],  # small bubble missed by LLM
        ]

        matched, unmatched_llm, unmatched_off = match_boxes(llm_boxes, offline_boxes)
        self.assertEqual(matched, [(0, 0)])
        self.assertEqual(unmatched_llm, [1])
        self.assertEqual(unmatched_off, [1])

    def test_refine_block_geometry(self):
        llm_blk = TextBlock(
            xyxy=[95, 95, 205, 205],
            text=["Hello", "World"],
            src_is_vertical=False,
        )
        offline_blk = TextBlock(
            xyxy=[100, 100, 200, 200],
            text=[],
        )

        dummy_img = np.full((300, 300, 3), 255, dtype=np.uint8)
        refined = refine_block_geometry(llm_blk, offline_blk, img=dummy_img, im_w=300, im_h=300)

        self.assertEqual(refined.xyxy, [100, 100, 200, 200])
        self.assertEqual(refined.text, ["Hello", "World"])

    def test_run_superior_verify_mocked_detector(self):
        dummy_img = np.full((400, 400, 3), 255, dtype=np.uint8)
        # Add simulated text stroke
        dummy_img[120:180, 120:180] = 0

        llm_blk = TextBlock(
            xyxy=[90, 90, 210, 210],
            text=["Mock speech"],
            src_is_vertical=False,
        )

        mock_offline = MagicMock()
        mock_offline.detect.return_value = (
            np.zeros((400, 400), dtype=np.uint8),
            [TextBlock(xyxy=[100, 100, 200, 200], text=[])],
        )

        mask, refined_blks = run_superior_verify(
            dummy_img,
            [llm_blk],
            offline_detector=mock_offline,
        )

        self.assertEqual(len(refined_blks), 1)
        # Stroke-tightened coordinates inside [100, 100, 200, 200]
        self.assertEqual(refined_blks[0].xyxy, [117, 117, 183, 183])
        self.assertEqual(refined_blks[0].text, ["Mock speech"])
        self.assertEqual(mask.shape, (400, 400))


class TestTranslationVerifier(unittest.TestCase):

    def test_audit_valid(self):
        src = ["こんにちは", "さようなら"]
        trans = ["Xin chào", "Tạm biệt"]
        valid, bad, msg = audit_translation(src, trans)
        self.assertTrue(valid)
        self.assertEqual(bad, [])

    def test_audit_dropped_lines(self):
        src = ["こんにちは", "ありがとう", "さようなら"]
        trans = ["Xin chào", "", "Tạm biệt"]
        valid, bad, msg = audit_translation(src, trans)
        self.assertFalse(valid)
        self.assertIn(1, bad)

    def test_audit_count_mismatch(self):
        src = ["A", "B", "C"]
        trans = ["A", "B"]
        valid, bad, msg = audit_translation(src, trans)
        self.assertFalse(valid)
        self.assertEqual(bad, [2])

    def test_repetitive_loop_detection(self):
        normal_text = "Tôi hiểu rồi, chúng ta sẽ bắt đầu ngay bây giờ."
        self.assertFalse(is_repetitive_loop(normal_text))

        repetitive = "ngay bây giờ " * 20
        self.assertTrue(is_repetitive_loop(repetitive))

    def test_self_correct_translation(self):
        src = ["Hello", "Thank you", "Goodbye"]
        trans = ["Xin chào", "", "Tạm biệt"]

        mock_translator = MagicMock()
        mock_translator.translate.return_value = ["Cảm ơn"]

        repaired = self_correct_translation(
            mock_translator,
            src,
            trans,
            [1],
        )
        self.assertEqual(repaired, ["Xin chào", "Cảm ơn", "Tạm biệt"])


class TestInpaintVerifier(unittest.TestCase):

    def test_detect_clean_inpaint(self):
        orig = np.full((100, 100, 3), 255, dtype=np.uint8)
        orig[40:60, 40:60] = 0  # text stroke

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[35:65, 35:65] = 255

        # Perfectly clean inpaint (solid white)
        inpainted_clean = np.full((100, 100, 3), 255, dtype=np.uint8)

        blk = TextBlock(xyxy=[35, 35, 65, 65])
        has_residuals, _, bad_indices = detect_inpaint_residuals(
            orig, inpainted_clean, mask, [blk]
        )
        self.assertFalse(has_residuals)
        self.assertEqual(bad_indices, [])

    def test_detect_ghost_text_and_pass2(self):
        orig = np.full((100, 100, 3), 255, dtype=np.uint8)
        # Draw high-contrast text stroke
        cv2.putText(orig, "TEST", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:90, 10:90] = 255

        # Inpainted image still has ghost text
        inpainted_ghost = orig.copy()

        blk = TextBlock(xyxy=[10, 10, 90, 90])
        has_residuals, res_mask, bad_indices = detect_inpaint_residuals(
            orig, inpainted_ghost, mask, [blk], edge_thresh=0.05
        )
        self.assertTrue(has_residuals)
        self.assertIn(0, bad_indices)
        self.assertTrue(np.count_nonzero(res_mask) > 0)

        # Test pass 2 invocation
        mock_inpainter = MagicMock()
        mock_inpainter.inpaint.return_value = np.full((100, 100, 3), 255, dtype=np.uint8)

        final_img = run_inpaint_verify_and_refine(
            mock_inpainter,
            orig,
            inpainted_ghost,
            mask,
            [blk],
        )
        self.assertTrue(mock_inpainter.inpaint.called)
        # Check that areas under residual mask were replaced with clean inpaint
        self.assertEqual(final_img[50, 50, 0], 255)


if __name__ == "__main__":
    unittest.main()
