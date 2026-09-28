import os
import unittest
from unittest import mock
import numpy as np

from qtpy.QtWidgets import QApplication
from qtpy.QtCore import QPointF, QRectF

from ballontranslator.utils.fonts.fontformat import FontFormat
from ballontranslator.utils.text.textblock import TextBlock
from ballontranslator.utils.project.config import pcfg

app = QApplication.instance()
if app is None:
    app = QApplication([])


class TestBackgroundPipelineRender(unittest.TestCase):

    def setUp(self):
        self.blk1 = TextBlock(
            xyxy=[10, 10, 100, 50],
            lines=[[[10, 10], [100, 10], [100, 50], [10, 50]]],
            translation="Page 1 Text",
        )
        self.blk1.font_size = 20
        self.blk1.font_family = "Arial"

        self.blk2 = TextBlock(
            xyxy=[20, 20, 120, 60],
            lines=[[[20, 20], [120, 20], [120, 60], [20, 60]]],
            translation="Page 2 Text",
        )
        self.blk2.font_size = 20
        self.blk2.font_family = "Arial"

    def test_on_pagtrans_finished_no_page_jump(self):
        """When page_index != current_row, pageList row must NOT change, and offscreen render must be called."""
        from ballontranslator.ui.mainwindow import MainWindow

        win = mock.MagicMock(spec=MainWindow)
        win.imgtrans_proj = mock.MagicMock()
        win.imgtrans_proj.num_pages = 2
        win.imgtrans_proj.get_blklist_byidx.return_value = [self.blk2]
        win.backup_blkstyles = []
        win._render_global_format = None
        win._render_only = False
        win._run_imgtrans_wo_textstyle_update = False
        gf = FontFormat()
        win.textPanel = mock.MagicMock()
        win.textPanel.formatpanel.global_format = gf
        win.st_manager = mock.MagicMock()
        win.llmContextEditor = mock.MagicMock()

        # Currently active page is row 0 (Page 1)
        win.pageList = mock.MagicMock()
        win.pageList.currentIndex.return_value.row.return_value = 0

        # Offscreen render mock
        win.render_and_save_page_offscreen = mock.MagicMock()

        # Invoke on_pagtrans_finished for page 1 (which is NOT row 0)
        MainWindow.on_pagtrans_finished(win, page_index=1)

        # 1. Ensure pageList.setCurrentRow was NEVER called!
        win.pageList.setCurrentRow.assert_not_called()

        # 2. Ensure render_and_save_page_offscreen was called with page_index 1
        win.render_and_save_page_offscreen.assert_called_once()
        call_args = win.render_and_save_page_offscreen.call_args
        self.assertEqual(call_args[0][0], 1)

        # 3. Ensure project was saved
        win.imgtrans_proj.save.assert_called_once()

    def test_on_pagtrans_finished_active_page_refreshes(self):
        """When page_index == current_row, the active canvas is refreshed directly without jumping."""
        from ballontranslator.ui.mainwindow import MainWindow

        win = mock.MagicMock(spec=MainWindow)
        win.imgtrans_proj = mock.MagicMock()
        win.imgtrans_proj.num_pages = 2
        win.imgtrans_proj.get_blklist_byidx.return_value = [self.blk1]
        win.backup_blkstyles = []
        win._render_global_format = None
        win._render_only = False
        win._run_imgtrans_wo_textstyle_update = False
        gf = FontFormat()
        win.textPanel = mock.MagicMock()
        win.textPanel.formatpanel.global_format = gf
        win.canvas = mock.MagicMock()
        win.st_manager = mock.MagicMock()
        win.llmContextEditor = mock.MagicMock()
        win.saveCurrentPage = mock.MagicMock()
        win.render_and_save_page_offscreen = mock.MagicMock()

        # Currently active page is row 0
        win.pageList = mock.MagicMock()
        win.pageList.currentIndex.return_value.row.return_value = 0

        # Invoke for page 0 (active page)
        MainWindow.on_pagtrans_finished(win, page_index=0)

        # Ensure pageList.setCurrentRow was NOT called
        win.pageList.setCurrentRow.assert_not_called()

        # Canvas and st_manager were updated for live display
        win.canvas.updateCanvas.assert_called_once()
        win.st_manager.updateSceneTextitems.assert_called_once()

        # saveCurrentPage was called for active page
        win.saveCurrentPage.assert_called_once_with(False, False)

        # render_and_save_page_offscreen was NOT called
        win.render_and_save_page_offscreen.assert_not_called()

    def test_render_and_save_page_offscreen_execution(self):
        """Verify render_and_save_page_offscreen loads image, creates TextBlkItems, and saves result."""
        from ballontranslator.ui.mainwindow import MainWindow

        win = mock.MagicMock(spec=MainWindow)
        win.imgtrans_proj = mock.MagicMock()
        win.imgtrans_proj.is_empty = False
        win.imgtrans_proj.idx2pagename.return_value = "page_002.png"
        
        # 100x100 RGB image
        dummy_img = np.full((100, 100, 3), 200, dtype=np.uint8)
        win.imgtrans_proj.load_inpainted_by_imgname.return_value = dummy_img
        win.imgtrans_proj.get_blklist_byidx.return_value = [self.blk2]
        win.imgtrans_proj.result_dir.return_value = "/tmp/test_bt_result"
        win.imgtrans_proj.get_result_path.return_value = "/tmp/test_bt_result/page_002.png"

        win.st_manager = mock.MagicMock()
        win.st_manager.formatpanel.familybox.currentText.return_value = "Arial"
        win.imsave_thread = mock.MagicMock()

        # Call offscreen render
        MainWindow.render_and_save_page_offscreen(win, page_index=1, auto_textlayout=False, squeeze_bounding_rect=True)

        # Verify imsave_thread.saveImg was called with the target path and image
        win.imsave_thread.saveImg.assert_called_once()
        args, kwargs = win.imsave_thread.saveImg.call_args
        self.assertEqual(args[0], "/tmp/test_bt_result/page_002.png")
        self.assertEqual(args[2], "page_002.png")

    def test_layout_textblk_offscreen_does_not_modify_widgets(self):
        """Ensure layout_textblk with update_widgets=False doesn't touch pairwidget_list."""
        from ballontranslator.ui.text_engine.editing.manager import SceneTextManager
        from ballontranslator.ui.text_engine.item import TextBlkItem

        manager = mock.MagicMock(spec=SceneTextManager)
        manager.imgtrans_proj = mock.MagicMock()
        manager.pairwidget_list = [mock.MagicMock(), mock.MagicMock()]
        manager.auto_textlayout_flag = True

        blk_item = TextBlkItem(blk=self.blk2, idx=0, set_format=True, show_rect=False)
        dummy_img = np.full((200, 200, 3), 255, dtype=np.uint8)

        # Call layout_textblk with update_widgets=False
        with mock.patch("ballontranslator.ui.text_engine.editing.manager.seg_text", return_value=(["Hello", "world"], " ")):
            SceneTextManager.layout_textblk(
                manager,
                blkitem=blk_item,
                text="Hello world",
                img=dummy_img,
                update_widgets=False,
            )

        # Verify pairwidget_list was NOT updated
        manager.pairwidget_list[0].e_trans.setPlainText.assert_not_called()


if __name__ == '__main__':
    unittest.main()
