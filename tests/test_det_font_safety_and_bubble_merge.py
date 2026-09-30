"""Unit tests for font size safety, clamping, and diagonal speech bubble merge protection."""

import unittest
import numpy as np

from ballontranslator.utils.text.textblock import TextBlock, try_merge_textline, examine_textblk
from ballontranslator.utils.text.textlines_merge import quadrilateral_can_merge_region, Quadrilateral
from ballontranslator.modules.textdetector.llm.detector_llm import LLMDetector, slice_block_into_child_lines


class TestDetFontSafetyAndBubbleMerge(unittest.TestCase):
    """Test suite for DET font safety and speech bubble separation."""

    def test_try_merge_textline_prevents_diagonal_merge_horizontal(self):
        """Diagonal speech bubbles placed diagonally should not merge."""
        # Bubble 1: top-left [100, 100, 200, 140]
        blk1 = TextBlock([100, 100, 200, 140], lines=[[[100, 100], [200, 100], [200, 140], [100, 140]]])
        blk1.vertical = blk1.src_is_vertical = False
        examine_textblk(blk1, 1000, 1000)

        # Bubble 2: bottom-right diagonal [190, 145, 290, 185]
        # Only 10px horizontal overlap out of 100px width (10%), centers are shifted by 90px
        blk2 = TextBlock([190, 145, 290, 185], lines=[[[190, 145], [290, 145], [290, 185], [190, 185]]])
        blk2.vertical = blk2.src_is_vertical = False
        examine_textblk(blk2, 1000, 1000)

        merged = try_merge_textline(blk1, blk2)
        self.assertFalse(merged, "Should not merge horizontally diagonal speech bubbles.")

    def test_try_merge_textline_allows_legitimate_paragraph_lines_horizontal(self):
        """Lines within the same paragraph/speech bubble should merge."""
        # Line 1: [100, 100, 250, 130] (width 150)
        blk1 = TextBlock([100, 100, 250, 130], lines=[[[100, 100], [250, 100], [250, 130], [100, 130]]])
        blk1.vertical = blk1.src_is_vertical = False
        examine_textblk(blk1, 1000, 1000)

        # Line 2 directly beneath Line 1: [105, 138, 245, 168] (large overlap > 80%, centers aligned)
        blk2 = TextBlock([105, 138, 245, 168], lines=[[[105, 138], [245, 138], [245, 168], [105, 168]]])
        blk2.vertical = blk2.src_is_vertical = False
        examine_textblk(blk2, 1000, 1000)

        merged = try_merge_textline(blk1, blk2)
        self.assertTrue(merged, "Should merge stacked text lines within the same bubble.")

    def test_try_merge_textline_prevents_diagonal_merge_vertical(self):
        """Diagonal vertical speech bubbles should not merge."""
        # Bubble 1: top-right column [300, 100, 330, 250] (height 150)
        blk1 = TextBlock([300, 100, 330, 250], lines=[[[300, 100], [330, 100], [330, 250], [300, 250]]])
        blk1.vertical = blk1.src_is_vertical = True
        examine_textblk(blk1, 1000, 1000)

        # Bubble 2: bottom-left diagonal [260, 240, 290, 390] (only 10px vertical overlap, centers shifted by 140px)
        blk2 = TextBlock([260, 240, 290, 390], lines=[[[260, 240], [290, 240], [290, 390], [260, 390]]])
        blk2.vertical = blk2.src_is_vertical = True
        examine_textblk(blk2, 1000, 1000)

        merged = try_merge_textline(blk1, blk2)
        self.assertFalse(merged, "Should not merge vertically diagonal speech bubbles.")

    def test_slice_block_tightens_single_line_stroke_mask(self):
        """Single line box inside a huge white bubble should tighten to actual strokes."""
        # Create an ROI of size 120x100 with white background and dark text in the middle
        roi = np.full((120, 100, 3), 255, dtype=np.uint8)
        # Put black text strokes at rows 45 to 65, cols 20 to 80
        roi[45:65, 20:80] = 0

        polys = slice_block_into_child_lines(
            roi, parent_x1=10, parent_y1=10, parent_x2=110, parent_y2=130, is_vertical=False, num_lines=1
        )
        self.assertEqual(len(polys), 1)
        poly = polys[0]
        # Bounding box should be tightened around the text strokes (y roughly 10+45=55 to 10+65=75)
        sy1 = np.min(poly[:, 1])
        sy2 = np.max(poly[:, 1])
        self.assertLess(sy2 - sy1, 40, "Tightened line height should be much smaller than the 120px parent box.")

    def test_detector_llm_font_size_clamp_params(self):
        """LLMDetector should support font size constraints and auto clamp."""
        detector = LLMDetector()
        self.assertIn("font size multiplier", detector.params)
        self.assertIn("font size max", detector.params)
        self.assertIn("font size min", detector.params)
        self.assertIn("font size auto clamp", detector.params)


if __name__ == "__main__":
    unittest.main()
