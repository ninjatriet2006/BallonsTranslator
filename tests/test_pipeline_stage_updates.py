import unittest
from unittest import mock

from ballontranslator.ui.mainwindow import MainWindow


class TestPipelineStageUpdates(unittest.TestCase):

    def test_on_page_stage_finished_active_page(self):
        """When the finished stage belongs to the currently active page, UI must update immediately."""
        win = mock.MagicMock(spec=MainWindow)
        win.st_manager = mock.MagicMock()
        win.canvas = mock.MagicMock()
        win.imgtrans_proj = mock.MagicMock()

        # Active page is row 0
        win.pageList = mock.MagicMock()
        win.pageList.currentIndex.return_value.isValid.return_value = True
        win.pageList.currentIndex.return_value.row.return_value = 0

        # Stage 'detect' on active page
        MainWindow.on_page_stage_finished(win, 'detect', page_index=0)
        win.st_manager.updateSceneTextitems.assert_called_once()
        win.canvas.updateCanvas.assert_not_called()

        # Reset mocks
        win.st_manager.reset_mock()
        win.canvas.reset_mock()

        # Stage 'ocr' on active page
        MainWindow.on_page_stage_finished(win, 'ocr', page_index=0)
        win.st_manager.updateSceneTextitems.assert_called_once()
        win.canvas.updateCanvas.assert_not_called()

        # Reset mocks
        win.st_manager.reset_mock()
        win.canvas.reset_mock()

        # Stage 'inpaint' on active page
        MainWindow.on_page_stage_finished(win, 'inpaint', page_index=0)
        win.imgtrans_proj.set_current_img_byidx.assert_called_once_with(0)
        win.canvas.updateCanvas.assert_called_once()
        win.st_manager.updateSceneTextitems.assert_not_called()

    def test_on_page_stage_finished_inactive_page(self):
        """When the finished stage belongs to a background page, active UI must NOT update."""
        win = mock.MagicMock(spec=MainWindow)
        win.st_manager = mock.MagicMock()
        win.canvas = mock.MagicMock()
        win.imgtrans_proj = mock.MagicMock()

        # Active page is row 0, but finished stage is for page 1
        win.pageList = mock.MagicMock()
        win.pageList.currentIndex.return_value.isValid.return_value = True
        win.pageList.currentIndex.return_value.row.return_value = 0

        MainWindow.on_page_stage_finished(win, 'detect', page_index=1)
        MainWindow.on_page_stage_finished(win, 'ocr', page_index=1)
        MainWindow.on_page_stage_finished(win, 'inpaint', page_index=1)

        win.st_manager.updateSceneTextitems.assert_not_called()
        win.canvas.updateCanvas.assert_not_called()
        win.imgtrans_proj.set_current_img_byidx.assert_not_called()


if __name__ == '__main__':
    unittest.main()
