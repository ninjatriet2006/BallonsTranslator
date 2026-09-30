import os
import sys
import unittest
from unittest import mock

# Ensure QApplication is initialized for QWidget tests
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from qtpy.QtWidgets import QApplication
from qtpy.QtGui import QPixmap

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

from ballontranslator.ui.custom_widget.message import (
    SegmentedProgressBar,
    TaskProgressBar,
    ImgtransProgressMessageBox,
)
from ballontranslator.ui.threads.pipeline_threads import ImgtransThread, TranslateThread
from ballontranslator.utils.project.config import pcfg


class TestProgressBox(unittest.TestCase):

    def test_segmented_progress_bar_values(self):
        bar = SegmentedProgressBar()
        self.assertEqual(bar.success_ratio, 0.0)
        self.assertEqual(bar.error_ratio, 0.0)

        # Backward compatibility setValue(int)
        bar.setValue(60)
        self.assertAlmostEqual(bar.success_ratio, 0.6)
        self.assertEqual(bar.error_ratio, 0.0)

        # Dual value setting
        bar.setValues(0.7, 0.2)
        self.assertAlmostEqual(bar.success_ratio, 0.7)
        self.assertAlmostEqual(bar.error_ratio, 0.2)

        # Clamping when exceeding 1.0
        bar.setValues(0.8, 0.5)
        self.assertAlmostEqual(bar.success_ratio, 0.8)
        self.assertAlmostEqual(bar.error_ratio, 0.2)

    def test_segmented_progress_bar_paint_rendering(self):
        """Ensure paintEvent executes without raising SIP or C++ exceptions across Qt backends."""
        bar = SegmentedProgressBar()
        bar.resize(200, 10)
        pm = QPixmap(200, 10)

        # 1. Zero state
        bar.setValues(0.0, 0.0)
        bar.render(pm)

        # 2. Only success
        bar.setValues(0.5, 0.0)
        bar.render(pm)

        # 3. Dual success and error
        bar.setValues(0.4, 0.3)
        bar.render(pm)

        # 4. Only error
        bar.setValues(0.0, 0.8)
        bar.render(pm)

    def test_task_progress_bar_error_rendering(self):
        tpb = TaskProgressBar(description="Detecting: ", verbose=True)
        # Normal update
        tpb.updateProgress(50)
        self.assertIn("Detecting: ", tpb.textlabel.text())
        self.assertNotIn("failed", tpb.textlabel.text())

        # Update with errors
        tpb.updateProgress(60, error_count=2, total_count=10)
        self.assertIn("failed", tpb.textlabel.text())
        self.assertIn("2", tpb.textlabel.text())
        # Check segmented values
        # 60% total (0.6), 2/10 error (0.2) -> success = 0.4, error = 0.2
        self.assertAlmostEqual(tpb.progressbar.success_ratio, 0.4)
        self.assertAlmostEqual(tpb.progressbar.error_ratio, 0.2)

        # 100% done with errors
        tpb.updateProgress(100, error_count=3, total_count=10)
        self.assertIn("3", tpb.verbose_label.text())
        self.assertIn("failed", tpb.verbose_label.text())

    def test_imgtrans_progress_message_box_five_bars_and_size(self):
        box = ImgtransProgressMessageBox()

        # Check all 5 bars exist and are in the expected pipeline order
        self.assertTrue(hasattr(box, 'upscale_bar'))
        self.assertTrue(hasattr(box, 'detect_bar'))
        self.assertTrue(hasattr(box, 'ocr_bar'))
        self.assertTrue(hasattr(box, 'translate_bar'))
        self.assertTrue(hasattr(box, 'inpaint_bar'))

        self.assertIsInstance(box.upscale_bar, TaskProgressBar)
        self.assertIsInstance(box.detect_bar, TaskProgressBar)
        self.assertIsInstance(box.ocr_bar, TaskProgressBar)
        self.assertIsInstance(box.translate_bar, TaskProgressBar)
        self.assertIsInstance(box.inpaint_bar, TaskProgressBar)

        # Check stable width (540px)
        self.assertEqual(box.width(), 540)
        box.fit_to_content()
        self.assertEqual(box.width(), 540)

        # Check updateUpscaleProgress
        box.updateUpscaleProgress(80, error_count=1, total_count=5)
        self.assertAlmostEqual(box.upscale_bar.progressbar.success_ratio, 0.6)
        self.assertAlmostEqual(box.upscale_bar.progressbar.error_ratio, 0.2)

        # Check zero_progress resets all 5 bars
        box.zero_progress()
        self.assertEqual(box.upscale_bar.progressbar.success_ratio, 0.0)
        self.assertEqual(box.detect_bar.progressbar.success_ratio, 0.0)
        self.assertEqual(box.ocr_bar.progressbar.success_ratio, 0.0)
        self.assertEqual(box.translate_bar.progressbar.success_ratio, 0.0)
        self.assertEqual(box.inpaint_bar.progressbar.success_ratio, 0.0)

        # Ensure box paints cleanly
        pm = QPixmap(540, 300)
        box.render(pm)

    def test_pipeline_thread_signals_and_upscale_finished(self):
        # Verify signal emissions and slot reception for progress reporting
        th = ImgtransThread(mock.MagicMock(), mock.MagicMock(), mock.MagicMock(), mock.MagicMock())
        th.imgtrans_proj = mock.MagicMock()
        th.num_pages = 5

        received = {}
        th.update_upscale_progress.connect(lambda p, e=0: received.update({'upscale': (p, e)}))
        th.update_detect_progress.connect(lambda p, e=0: received.update({'detect': (p, e)}))
        th.update_ocr_progress.connect(lambda p, e=0: received.update({'ocr': (p, e)}))
        th.update_translate_progress.connect(lambda p, e=0: received.update({'translate': (p, e)}))
        th.update_inpaint_progress.connect(lambda p, e=0: received.update({'inpaint': (p, e)}))

        th.update_upscale_progress.emit(1, 0)
        th.update_detect_progress.emit(2, 1)
        th.update_ocr_progress.emit(3, 0)
        th.update_translate_progress.emit(4, 2)
        th.update_inpaint_progress.emit(5, 0)

        self.assertEqual(received['upscale'], (1, 0))
        self.assertEqual(received['detect'], (2, 1))
        self.assertEqual(received['ocr'], (3, 0))
        self.assertEqual(received['translate'], (4, 2))
        self.assertEqual(received['inpaint'], (5, 0))

        # Verify TranslateThread progress_changed signal
        with mock.patch('ballontranslator.ui.threads.pipeline_threads.TRANSLATORS', {'test': mock.MagicMock()}):
            tr_thread = TranslateThread()
            tr_received = []
            tr_thread.progress_changed.connect(lambda p, e=0: tr_received.append((p, e)))
            tr_thread.progress_changed.emit(3, 1)
            self.assertEqual(tr_received, [(3, 1)])

        # Test upscale_finished
        pcfg.module.enable_upscale = True
        pcfg.module.upscaler = 'RealESRGAN'
        th.upscale_counter = 4
        self.assertFalse(th.upscale_finished())
        th.upscale_counter = 5
        self.assertTrue(th.upscale_finished())

        pcfg.module.enable_upscale = False
        self.assertTrue(th.upscale_finished())

        # Test recent_finished_index with upscale
        pcfg.module.enable_upscale = True
        pcfg.module.upscaler = 'RealESRGAN'
        pcfg.module.enable_detect = True
        pcfg.module.enable_ocr = False
        pcfg.module.enable_inpaint = False
        pcfg.module.enable_translate = False
        th.upscale_counter = 2
        th.detect_counter = 4
        # recent_finished_index should be min(2, 4) - 1 = 1
        self.assertEqual(th.recent_finished_index(10), 1)


if __name__ == '__main__':
    unittest.main()
