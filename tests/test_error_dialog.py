import os
import sys
import unittest
from unittest import mock

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PySide6.QtWidgets import QApplication, QWidget, QDialog

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

from ballontranslator.ui.dialogs.pipeline_error_dialog import PipelineErrorDialog
from ballontranslator.ui.framelesswindow import OutsideClickFramelessMixin
from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.utils.project.stage_auditor import StageIssue
from ballontranslator.utils.project.message import create_error_dialog
from ballontranslator.utils.project import shared


class TestErrorDialog(unittest.TestCase):

    def test_pipeline_error_dialog_attributes_and_no_crash(self):
        issue = StageIssue(
            page_key="page_001.png",
            stage="detect",
            error_type="stage_failed",
            message="Model timeout",
            page_index=0,
        )
        dlg = PipelineErrorDialog([issue])

        # Verify title_bar and close_button exist to avoid OutsideClickFramelessMixin crash
        self.assertTrue(hasattr(dlg, 'title_bar'))
        self.assertIsNotNone(dlg.title_bar)
        self.assertTrue(hasattr(dlg, 'close_button'))
        self.assertIsNotNone(dlg.close_button)

        # Verify _can_drag_title does not crash on mouse interactions
        self.assertTrue(dlg._can_drag_title(dlg.title_bar))
        self.assertFalse(dlg._can_drag_title(dlg.close_button))
        self.assertFalse(dlg._can_drag_title(dlg))

        # Verify dimensions
        self.assertGreaterEqual(dlg.width(), 520)
        self.assertGreaterEqual(dlg.height(), 360)
        self.assertLessEqual(dlg.maximumWidth(), 700)
        self.assertLessEqual(dlg.maximumHeight(), 650)

        # Verify dismiss rejects dialog rather than leaving modal exec running
        with mock.patch.object(dlg, 'reject') as mock_reject:
            dlg._dismiss_transient_window()
            mock_reject.assert_called_once()

    def test_outside_click_frameless_mixin_safe_when_no_title_bar(self):
        class DummyWidget(OutsideClickFramelessMixin, QWidget):
            pass

        dummy = DummyWidget()
        # Should return False safely without AttributeError
        self.assertFalse(dummy._can_drag_title(dummy))

    def test_create_error_dialog_string_exception(self):
        # Should not crash when a string or None is passed instead of BaseException
        shared.HEADLESS = True
        try:
            create_error_dialog("Plain string error without traceback", "Something failed")
            create_error_dialog(None, "None error")
        finally:
            shared.HEADLESS = False

    def test_on_create_errdialog_reentrancy_guard(self):
        win = mock.MagicMock(spec=MainWindow)
        win._active_errdialog = mock.MagicMock()

        # Calling on_create_errdialog when _active_errdialog is already set must not open a new dialog
        with mock.patch('ballontranslator.ui.mainwindow.QMessageBox') as mock_qmsg:
            MainWindow.on_create_errdialog(win, "Second error")
            mock_qmsg.assert_not_called()

    def test_create_error_dialog_suppresses_cancellations(self):
        from ballontranslator.utils.system.download_util import DownloadCancelled
        from ballontranslator.modules.llm_chat import LLMRequestStopped

        with mock.patch('ballontranslator.utils.project.shared.create_errdialog_in_mainthread') as mock_err_dlg, \
             mock.patch('ballontranslator.utils.project.message.LOGGER.error') as mock_log_err:
            create_error_dialog(DownloadCancelled('Module preparation cancelled by user.'))
            create_error_dialog(LLMRequestStopped('Stopped by user'))
            mock_err_dlg.assert_not_called()
            mock_log_err.assert_not_called()

    def test_module_prepare_cancellation_suppresses_error_dialog(self):
        import threading
        from ballontranslator.ui.module_manager import ModuleManager
        from ballontranslator.utils.system.download_util import DownloadCancelled

        class MockProject:
            def __init__(self):
                self.pages = {}
            def pagename2idx(self, name):
                return 0

        manager = ModuleManager(MockProject())

        class DummyThread:
            def __init__(self):
                self.last_error = DownloadCancelled("Cancelled by user")
                self.module_key = "inpainter"
                self.last_set_module_name = "lama_large_512px"
                self.cancel_event = threading.Event()

        dummy_th = DummyThread()
        with mock.patch('ballontranslator.ui.module_manager.create_error_dialog') as mock_dlg:
            manager._show_module_prepare_failure(dummy_th)
            mock_dlg.assert_not_called()

            # Also if cancel_event is set with a generic exception
            dummy_th.last_error = RuntimeError("Aborted")
            dummy_th.cancel_event.set()
            manager._show_module_prepare_failure(dummy_th)
            mock_dlg.assert_not_called()

            # Normal error without cancel_event set MUST call create_error_dialog
            dummy_th.cancel_event.clear()
            manager._show_module_prepare_failure(dummy_th)
            mock_dlg.assert_called_once()


if __name__ == '__main__':
    unittest.main()
