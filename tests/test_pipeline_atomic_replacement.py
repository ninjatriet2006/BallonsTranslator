import unittest
from unittest import mock
import numpy as np

from ballontranslator.utils.text.textblock import TextBlock
from ballontranslator.utils.fonts.fontformat import FontFormat
from ballontranslator.utils.project.config import pcfg
from ballontranslator.ui.mainwindow import MainWindow


class TestPipelineAtomicReplacement(unittest.TestCase):

    def setUp(self):
        self.blk_p0 = TextBlock(
            xyxy=[10, 10, 100, 50],
            lines=[[[10, 10], [100, 10], [100, 50], [10, 50]]],
            text=["Hello"],
            translation="Xin chào",
        )
        self.blk_p0.rich_text = "Xin chào"

        self.blk_p1 = TextBlock(
            xyxy=[20, 20, 120, 60],
            lines=[[[20, 20], [120, 20], [120, 60], [20, 60]]],
            text=["World"],
            translation="Thế giới",
        )
        self.blk_p1.rich_text = "Thế giới"

    def test_on_run_imgtrans_preserves_blocks_upfront_when_detect_enabled(self):
        """Starting pipeline with detect must NOT clear pages upfront across requested pages."""
        win = mock.MagicMock(spec=MainWindow)
        win.backup_blkstyles = []
        win.bottomBar = mock.MagicMock()
        win.bottomBar.textblockChecker.isChecked.return_value = False
        win.st_manager = mock.MagicMock()
        win.module_manager = mock.MagicMock()

        pages = {
            "page_0.png": [self.blk_p0],
            "page_1.png": [self.blk_p1],
        }
        win.imgtrans_proj = mock.MagicMock()
        win.imgtrans_proj.pages = pages
        win.imgtrans_proj.num_pages = 2
        win.imgtrans_proj.clear_page_progress = mock.MagicMock()

        orig_detect = pcfg.module.enable_detect
        pcfg.module.enable_detect = True
        try:
            MainWindow.on_run_imgtrans(win)

            # Pages must NOT be cleared upfront!
            self.assertEqual(len(win.imgtrans_proj.pages["page_0.png"]), 1)
            self.assertEqual(len(win.imgtrans_proj.pages["page_1.png"]), 1)
            self.assertEqual(win.imgtrans_proj.pages["page_0.png"][0].get_text(), "Hello")
            self.assertEqual(win.imgtrans_proj.pages["page_1.png"][0].get_text(), "World")
            self.assertEqual(win.imgtrans_proj.pages["page_0.png"][0].rich_text, "Xin chào")
            self.assertEqual(win.imgtrans_proj.pages["page_1.png"][0].rich_text, "Thế giới")
            # Pipeline should be triggered
            win.module_manager.runImgtransPipeline.assert_called_once()
        finally:
            pcfg.module.enable_detect = orig_detect

    def test_on_run_imgtrans_preserves_text_and_richtext_upfront_when_ocr_enabled(self):
        """Starting pipeline with OCR/translation must NOT wipe text or rich_text upfront."""
        win = mock.MagicMock(spec=MainWindow)
        win.backup_blkstyles = []
        win.bottomBar = mock.MagicMock()
        win.bottomBar.textblockChecker.isChecked.return_value = False
        win.st_manager = mock.MagicMock()
        win.module_manager = mock.MagicMock()

        pages = {
            "page_0.png": [self.blk_p0],
            "page_1.png": [self.blk_p1],
        }
        win.imgtrans_proj = mock.MagicMock()
        win.imgtrans_proj.pages = pages
        win.imgtrans_proj.num_pages = 2
        win.imgtrans_proj.clear_page_progress = mock.MagicMock()

        orig_detect = pcfg.module.enable_detect
        orig_ocr = pcfg.module.enable_ocr
        orig_trans = pcfg.module.enable_translate
        pcfg.module.enable_detect = False
        pcfg.module.enable_ocr = True
        pcfg.module.enable_translate = True
        try:
            MainWindow.on_run_imgtrans(win)

            # Text and rich_text must NOT be wiped upfront!
            self.assertEqual(win.imgtrans_proj.pages["page_0.png"][0].get_text(), "Hello")
            self.assertEqual(win.imgtrans_proj.pages["page_0.png"][0].rich_text, "Xin chào")
            self.assertEqual(win.imgtrans_proj.pages["page_1.png"][0].get_text(), "World")
            self.assertEqual(win.imgtrans_proj.pages["page_1.png"][0].rich_text, "Thế giới")
            # Styles should be safely backed up
            self.assertEqual(len(win.backup_blkstyles), 2)
            win.module_manager.runImgtransPipeline.assert_called_once()
        finally:
            pcfg.module.enable_detect = orig_detect
            pcfg.module.enable_ocr = orig_ocr
            pcfg.module.enable_translate = orig_trans


if __name__ == '__main__':
    unittest.main()
