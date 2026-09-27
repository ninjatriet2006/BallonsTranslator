import os
import unittest
from unittest import mock
from qtpy.QtCore import QObject
from qtpy.QtWidgets import QApplication, QMessageBox, QCheckBox

from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.textblock import TextBlock
from ballontranslator.utils import shared

app = QApplication.instance()
if app is None:
    app = QApplication([])


class TestApplyAllPages(unittest.TestCase):

    def setUp(self):
        self.blk1 = TextBlock(
            xyxy=[10, 10, 100, 50],
            lines=[[[10, 10], [100, 10], [100, 50], [10, 50]]],
            translation="Page 1 Text 1",
        )
        self.blk1.font_size = 28
        self.blk1.font_family = "Arial"
        self.blk1.frgb = (0, 0, 0)

        self.blk2 = TextBlock(
            xyxy=[10, 60, 100, 90],
            lines=[[[10, 60], [100, 60], [100, 90], [10, 90]]],
            translation="Page 1 Text 2",
        )
        self.blk2.font_size = 14
        self.blk2.font_family = "Arial"
        self.blk2.frgb = (0, 0, 0)

        self.blk3 = TextBlock(
            xyxy=[20, 20, 120, 70],
            lines=[[[20, 20], [120, 20], [120, 70], [20, 70]]],
            translation="Page 2 Text 1",
        )
        self.blk3.font_size = 36
        self.blk3.font_family = "Times New Roman"
        self.blk3.frgb = (255, 255, 255)

        class DummyProj:
            def __init__(self, pages):
                self.pages = pages
                self.saved = False

            def save(self):
                self.saved = True

        self.proj = DummyProj({
            "p1.png": [self.blk1, self.blk2],
            "p2.png": [self.blk3],
        })

    def _create_manager(self, target_fmt, textblk_items=None):
        from ballontranslator.ui.text_engine.editing.manager import SceneTextManager

        manager = SceneTextManager.__new__(SceneTextManager)
        QObject.__init__(manager)
        manager.mainwindow = None
        manager.imgtrans_proj = self.proj
        manager.textblk_item_list = textblk_items or []
        manager.canvas = mock.MagicMock()
        manager.canvas.scene = mock.MagicMock()
        manager.formatpanel = mock.MagicMock()
        manager.formatpanel.global_format = target_fmt
        return manager

    def test_apply_to_all_pages_with_keep_size(self):
        target_fmt = FontFormat()
        target_fmt.font_family = "Comic Sans MS"
        target_fmt.frgb = (255, 0, 0)
        target_fmt.font_size = 40
        target_fmt.font_weight = 75

        manager = self._create_manager(target_fmt)

        with mock.patch('ballontranslator.ui.text_engine.editing.manager.QMessageBox') as mock_msgbox_cls:
            mock_dialog = mock.MagicMock()
            mock_dialog.exec.return_value = QMessageBox.StandardButton.Ok
            mock_msgbox_cls.return_value = mock_dialog
            mock_msgbox_cls.StandardButton = QMessageBox.StandardButton
            mock_msgbox_cls.Icon = QMessageBox.Icon

            with mock.patch('ballontranslator.ui.text_engine.editing.manager.QCheckBox') as mock_cb_cls:
                mock_cb = mock.MagicMock()
                mock_cb.isChecked.return_value = True
                mock_cb_cls.return_value = mock_cb

                manager.onFormatAllPages(target_fmt)

        # Check Page 1 Block 1
        self.assertEqual(self.blk1.font_family, "Comic Sans MS")
        self.assertEqual(self.blk1.fontformat.frgb, (255, 0, 0))
        self.assertEqual(self.blk1.font_size, 28)  # Preserved!

        # Check Page 1 Block 2
        self.assertEqual(self.blk2.font_family, "Comic Sans MS")
        self.assertEqual(self.blk2.fontformat.frgb, (255, 0, 0))
        self.assertEqual(self.blk2.font_size, 14)  # Preserved!

        # Check Page 2 Block 1
        self.assertEqual(self.blk3.font_family, "Comic Sans MS")
        self.assertEqual(self.blk3.fontformat.frgb, (255, 0, 0))
        self.assertEqual(self.blk3.font_size, 36)  # Preserved!

        # Verify proj saved
        self.assertTrue(self.proj.saved)

    def test_apply_to_all_pages_without_keep_size(self):
        target_fmt = FontFormat()
        target_fmt.font_family = "Courier New"
        target_fmt.frgb = (0, 0, 255)
        target_fmt.font_size = 50

        manager = self._create_manager(target_fmt)

        with mock.patch('ballontranslator.ui.text_engine.editing.manager.QMessageBox') as mock_msgbox_cls:
            mock_dialog = mock.MagicMock()
            mock_dialog.exec.return_value = QMessageBox.StandardButton.Ok
            mock_msgbox_cls.return_value = mock_dialog
            mock_msgbox_cls.StandardButton = QMessageBox.StandardButton
            mock_msgbox_cls.Icon = QMessageBox.Icon

            with mock.patch('ballontranslator.ui.text_engine.editing.manager.QCheckBox') as mock_cb_cls:
                mock_cb = mock.MagicMock()
                mock_cb.isChecked.return_value = False
                mock_cb_cls.return_value = mock_cb

                manager.onFormatAllPages(target_fmt)

        # Check font sizes are updated to 50
        self.assertEqual(self.blk1.font_family, "Courier New")
        self.assertEqual(self.blk1.font_size, 50)
        self.assertEqual(self.blk2.font_size, 50)
        self.assertEqual(self.blk3.font_size, 50)

    def test_apply_to_all_pages_updates_live_canvas_items(self):
        target_fmt = FontFormat()
        target_fmt.font_family = "Impact"
        target_fmt.frgb = (128, 128, 0)
        target_fmt.font_size = 32

        live_item = mock.MagicMock()
        live_item.font_size = 20

        manager = self._create_manager(target_fmt, textblk_items=[live_item])

        with mock.patch('ballontranslator.ui.text_engine.editing.manager.QMessageBox') as mock_msgbox_cls:
            mock_dialog = mock.MagicMock()
            mock_dialog.exec.return_value = QMessageBox.StandardButton.Ok
            mock_msgbox_cls.return_value = mock_dialog
            mock_msgbox_cls.StandardButton = QMessageBox.StandardButton
            mock_msgbox_cls.Icon = QMessageBox.Icon

            with mock.patch('ballontranslator.ui.text_engine.editing.manager.QCheckBox') as mock_cb_cls:
                mock_cb = mock.MagicMock()
                mock_cb.isChecked.return_value = True
                mock_cb_cls.return_value = mock_cb

                manager.onFormatAllPages(target_fmt)

        # Verify live item was updated and size preserved
        live_item.set_fontformat.assert_called_once()
        applied_fmt = live_item.set_fontformat.call_args[0][0]
        self.assertEqual(applied_fmt.font_size, 20)
        self.assertEqual(applied_fmt.font_family, "Impact")
        manager.canvas.update.assert_called_once()

    def test_apply_to_all_pages_live_canvas_without_keep_size(self):
        target_fmt = FontFormat()
        target_fmt.font_family = "Impact"
        target_fmt.frgb = (128, 128, 0)
        target_fmt.font_size = 32

        live_item = mock.MagicMock()
        live_item.font_size = 20

        manager = self._create_manager(target_fmt, textblk_items=[live_item])

        with mock.patch('ballontranslator.ui.text_engine.editing.manager.QMessageBox') as mock_msgbox_cls:
            mock_dialog = mock.MagicMock()
            mock_dialog.exec.return_value = QMessageBox.StandardButton.Ok
            mock_msgbox_cls.return_value = mock_dialog
            mock_msgbox_cls.StandardButton = QMessageBox.StandardButton
            mock_msgbox_cls.Icon = QMessageBox.Icon

            with mock.patch('ballontranslator.ui.text_engine.editing.manager.QCheckBox') as mock_cb_cls:
                mock_cb = mock.MagicMock()
                mock_cb.isChecked.return_value = False
                mock_cb_cls.return_value = mock_cb

                manager.onFormatAllPages(target_fmt)

        live_item.set_fontformat.assert_called_once()
        applied_fmt = live_item.set_fontformat.call_args[0][0]
        self.assertEqual(applied_fmt.font_size, 32)
        self.assertEqual(applied_fmt.font_family, "Impact")

    def test_apply_to_all_pages_with_real_textblkitem(self):
        from ballontranslator.ui.text_engine.item import TextBlkItem

        real_item = TextBlkItem(blk=self.blk1, idx=0, set_format=True, show_rect=False)
        self.assertEqual(real_item.font_size, 28)

        # Test setter
        real_item.font_size = 35
        self.assertEqual(real_item.font_size, 35)
        self.assertEqual(self.blk1.font_size, 35)
        real_item.font_size = 28

        target_fmt = FontFormat()
        target_fmt.font_family = "Impact"
        target_fmt.frgb = (0, 255, 0)
        target_fmt.font_size = 48

        manager = self._create_manager(target_fmt, textblk_items=[real_item])

        with mock.patch('ballontranslator.ui.text_engine.editing.manager.QMessageBox') as mock_msgbox_cls:
            mock_dialog = mock.MagicMock()
            mock_dialog.exec.return_value = QMessageBox.StandardButton.Ok
            mock_msgbox_cls.return_value = mock_dialog
            mock_msgbox_cls.StandardButton = QMessageBox.StandardButton
            mock_msgbox_cls.Icon = QMessageBox.Icon

            with mock.patch('ballontranslator.ui.text_engine.editing.manager.QCheckBox') as mock_cb_cls:
                mock_cb = mock.MagicMock()
                mock_cb.isChecked.return_value = True
                mock_cb_cls.return_value = mock_cb

                # Must not raise AttributeError: 'TextBlkItem' object has no attribute 'font_size'
                manager.onFormatAllPages(target_fmt)

        self.assertEqual(real_item.font_size, 28)
        self.assertEqual(real_item.fontformat.font_family, "Impact")

    def test_fontformat_panel_apply_all_pages_button(self):
        shared.register_view_widget = mock.MagicMock()
        from ballontranslator.ui.text_engine.formatting.panel import FontFormatPanel

        panel = FontFormatPanel(app)
        panel.global_format = FontFormat()
        self.assertTrue(hasattr(panel, 'applyAllPagesBtn'))

        received_fmt = []
        panel.apply_all_fontfmt.connect(lambda fmt: received_fmt.append(fmt))

        panel.applyAllPagesBtn.click()
        self.assertEqual(len(received_fmt), 1)


if __name__ == '__main__':
    unittest.main()
