"""Unit tests for content-driven text layout, boundary clamping, and collision avoidance."""

import unittest
from typing import List
import numpy as np

from ballontranslator.utils.text.textblock import TextBlock, TextAlignment
from ballontranslator.utils.text.text_layout import (
    layout_text_content_driven,
    resolve_box_collisions,
    layout_text,
)


class TestContentDrivenTypesetting(unittest.TestCase):
    """Test suite for content-driven layout and collision avoidance."""

    def test_short_sentence_stays_on_one_line(self):
        """Short 1-2 word sentences should remain on a single line without wrapping."""
        words = ["Xin", "chào"]
        wl_list = [25, 35]
        delimiter = " "
        delimiter_len = 6
        line_height = 24
        orig_rect = [500, 200, 30, 120]  # Vertical original box

        text, xywh = layout_text_content_driven(
            words=words,
            wl_list=wl_list,
            delimiter=delimiter,
            delimiter_len=delimiter_len,
            line_height=line_height,
            bounding_rect=orig_rect,
            img_w=1000,
            img_h=1000,
            src_is_vertical=True,
        )

        self.assertEqual(text, "Xin chào")
        self.assertNotIn("\n", text, "Short sentence must stay on a single line.")
        # Ensure box is centered around original centroid (515, 260)
        cx = orig_rect[0] + orig_rect[2] / 2.0
        cy = orig_rect[1] + orig_rect[3] / 2.0
        new_cx = xywh[0] + xywh[2] / 2.0
        new_cy = xywh[1] + xywh[3] / 2.0
        self.assertAlmostEqual(cx, new_cx, delta=1.5)
        self.assertAlmostEqual(cy, new_cy, delta=1.5)

    def test_never_cuts_longest_word(self):
        """The box width must always be at least as wide as the longest single word."""
        words = ["Đây", "là", "nghiêm-túc", "nhất"]
        wl_list = [25, 15, 120, 30]  # "nghiêm-túc" is 120px wide
        delimiter = " "
        delimiter_len = 6
        line_height = 20
        orig_rect = [300, 300, 40, 150]

        text, xywh = layout_text_content_driven(
            words=words,
            wl_list=wl_list,
            delimiter=delimiter,
            delimiter_len=delimiter_len,
            line_height=line_height,
            bounding_rect=orig_rect,
            img_w=1000,
            img_h=1000,
            src_is_vertical=True,
        )

        # Width of the box must be greater than 120px
        self.assertGreaterEqual(xywh[2], 120)
        # "nghiêm-túc" must remain a single unbroken word in the lines
        lines = text.split("\n")
        self.assertTrue(any("nghiêm-túc" in line for line in lines))

    def test_sequential_wrapping_no_top_bottom_fragmentation(self):
        """Sequential word wrap must flow line by line from top to bottom."""
        words = ["Tôi", "không", "thể", "tin", "được", "chuyện", "này"]
        wl_list = [25, 45, 25, 20, 35, 45, 30]
        delimiter = " "
        delimiter_len = 6
        line_height = 20
        orig_rect = [500, 200, 40, 150]

        text, xywh = layout_text_content_driven(
            words=words,
            wl_list=wl_list,
            delimiter=delimiter,
            delimiter_len=delimiter_len,
            line_height=line_height,
            bounding_rect=orig_rect,
            img_w=1000,
            img_h=1000,
            src_is_vertical=True,
        )

        lines = text.split("\n")
        # Text has 7 words; should wrap cleanly into 2-3 lines
        self.assertIn(len(lines), [2, 3])
        # Every word should be preserved in order
        all_words_output = " ".join(lines).split()
        self.assertEqual(all_words_output, words)
        # Height must match line count
        expected_min_h = len(lines) * line_height
        self.assertGreaterEqual(xywh[3], expected_min_h)

    def test_expansion_capped_at_envelope(self):
        """Width expansion should be capped within 50% - 100% of reference envelope."""
        words = ["A"] * 20  # Very long sentence of 20 words
        wl_list = [20] * 20
        delimiter = " "
        delimiter_len = 5
        line_height = 18
        orig_rect = [200, 200, 60, 60]  # Horizontal box of width 60

        text, xywh = layout_text_content_driven(
            words=words,
            wl_list=wl_list,
            delimiter=delimiter,
            delimiter_len=delimiter_len,
            line_height=line_height,
            bounding_rect=orig_rect,
            img_w=1000,
            img_h=1000,
            src_is_vertical=False,
            scale_factor_range=(1.5, 2.0),
        )

        # Max allowed width is 60 * 2.0 = 120 (plus small padding)
        self.assertLessEqual(xywh[2], 150)

    def test_boundary_clamping_prevents_going_off_image(self):
        """Boxes positioned near page margins must be clamped to image boundaries."""
        words = ["Một", "câu", "rất", "dài", "ở", "sát", "mép", "phải"]
        wl_list = [30, 25, 25, 25, 15, 25, 30, 35]
        delimiter = " "
        delimiter_len = 6
        line_height = 20

        # Right edge: centroid near x=980 on a 1000px wide image
        orig_rect_right = [970, 300, 20, 100]
        text, xywh_right = layout_text_content_driven(
            words=words,
            wl_list=wl_list,
            delimiter=delimiter,
            delimiter_len=delimiter_len,
            line_height=line_height,
            bounding_rect=orig_rect_right,
            img_w=1000,
            img_h=1000,
            src_is_vertical=True,
        )
        self.assertGreaterEqual(xywh_right[0], 0)
        self.assertLessEqual(xywh_right[0] + xywh_right[2], 1000, "Must not exceed right image border.")

        # Bottom edge: centroid near y=980 on a 1000px high image
        orig_rect_bottom = [400, 970, 60, 20]
        text, xywh_bottom = layout_text_content_driven(
            words=words,
            wl_list=wl_list,
            delimiter=delimiter,
            delimiter_len=delimiter_len,
            line_height=line_height,
            bounding_rect=orig_rect_bottom,
            img_w=1000,
            img_h=1000,
            src_is_vertical=False,
        )
        self.assertGreaterEqual(xywh_bottom[1], 0)
        self.assertLessEqual(xywh_bottom[1] + xywh_bottom[3], 1000, "Must not exceed bottom image border.")

    def test_resolve_box_collisions_two_overlapping(self):
        """Two overlapping boxes should be pushed apart to maintain min_gap."""
        b1 = [100, 100, 80, 40]
        b2 = [110, 115, 80, 40]  # Heavy vertical overlap
        resolved = resolve_box_collisions([b1, b2], min_gap=8, img_w=1000, img_h=1000)

        r1, r2 = resolved[0], resolved[1]
        overlap_x = min(r1[0] + r1[2], r2[0] + r2[2]) - max(r1[0], r2[0])
        overlap_y = min(r1[1] + r1[3], r2[1] + r2[3]) - max(r1[1], r2[1])

        # Either horizontally or vertically separated by >= 8px
        is_separated = (overlap_x <= -8) or (overlap_y <= -8)
        self.assertTrue(is_separated, f"Boxes must be separated by at least 8px: ox={overlap_x}, oy={overlap_y}")

    def test_resolve_box_collisions_three_clustered(self):
        """Three clustered boxes should all be separated without collision."""
        boxes = [
            [200, 200, 60, 30],
            [210, 210, 60, 30],
            [205, 220, 60, 30],
        ]
        resolved = resolve_box_collisions(boxes, min_gap=8, img_w=1000, img_h=1000)

        for i in range(len(resolved)):
            for j in range(i + 1, len(resolved)):
                r1, r2 = resolved[i], resolved[j]
                ox = min(r1[0] + r1[2], r2[0] + r2[2]) - max(r1[0], r2[0])
                oy = min(r1[1] + r1[3], r2[1] + r2[3]) - max(r1[1], r2[1])
                is_separated = (ox <= -8) or (oy <= -8)
                self.assertTrue(is_separated, f"Pair {i}-{j} must be separated: ox={ox}, oy={oy}")

    def test_resolve_box_collisions_non_overlapping_untouched(self):
        """Non-overlapping distant boxes should remain exactly at their positions."""
        boxes = [
            [100, 100, 50, 50],
            [300, 300, 50, 50],
        ]
        resolved = resolve_box_collisions(boxes, min_gap=8, img_w=1000, img_h=1000)
        self.assertEqual(boxes, resolved)

    def test_layout_text_integration_dispatches_content_driven(self):
        """Calling layout_text with bounding_rect for non-CJK text invokes content-driven layout."""
        blk = TextBlock([100, 100, 130, 250])
        blk.src_is_vertical = True
        blk.vertical = False

        dummy_mask = np.full((150, 150), 255, dtype=np.uint8)
        words = ["Xin", "chào", "các", "bạn"]
        wl_list = [25, 35, 20, 30]

        concated_text, xywh, start_top, adjust_xy = layout_text(
            blk=blk,
            mask=dummy_mask,
            mask_xyxy=[100, 100, 130, 250],
            centroid=[15, 75],
            words=words,
            wl_list=wl_list,
            delimiter=" ",
            delimiter_len=6,
            line_height=20,
            src_is_cjk=False,
            tgt_is_cjk=False,
            bounding_rect=[100, 100, 30, 150],
            img_w=1000,
            img_h=1000,
        )

        self.assertFalse(start_top)
        self.assertEqual(adjust_xy, 0)
        self.assertEqual(" ".join(concated_text.split()), "Xin chào các bạn")
        self.assertGreater(xywh[2], 0)
        self.assertGreater(xywh[3], 0)


if __name__ == "__main__":
    unittest.main()
