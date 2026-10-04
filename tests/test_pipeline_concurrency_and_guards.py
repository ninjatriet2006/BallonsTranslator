import threading
import unittest
from unittest import mock

from qtpy.QtWidgets import QApplication

from ballontranslator.ui.module_manager import ModuleManager
from ballontranslator.modules.llm_image import LLMImageRequester, _REQUEST_CONTEXT
from ballontranslator.utils.project.llm_profiles import LLMProfile

_app = QApplication.instance() or QApplication([])


class PipelineConcurrencyAndGuardsTests(unittest.TestCase):

    def setUp(self):
        self.mock_proj = mock.MagicMock()
        self.mock_proj.pages = {'p1': [], 'p2': []}
        self.mock_proj.is_empty = False
        self.mock_proj.num_pages = 2

    def test_is_pipeline_running(self):
        manager = ModuleManager(self.mock_proj)
        manager.imgtrans_thread = mock.MagicMock()
        manager.translate_thread = mock.MagicMock()

        # Both not running
        manager.imgtrans_thread.isRunning.return_value = False
        manager.translate_thread.isRunning.return_value = False
        with mock.patch.object(manager, '_module_preparation_running', return_value=False):
            self.assertFalse(manager.is_pipeline_running())

        # Imgtrans thread running
        manager.imgtrans_thread.isRunning.return_value = True
        self.assertTrue(manager.is_pipeline_running())

        # Translate thread running
        manager.imgtrans_thread.isRunning.return_value = False
        manager.translate_thread.isRunning.return_value = True
        self.assertTrue(manager.is_pipeline_running())

        # Module preparation running
        manager.translate_thread.isRunning.return_value = False
        with mock.patch.object(manager, '_module_preparation_running', return_value=True):
            self.assertTrue(manager.is_pipeline_running())

    def test_run_imgtrans_pipeline_rejected_when_running(self):
        manager = ModuleManager(self.mock_proj)
        manager.progress_msgbox = mock.MagicMock()

        with mock.patch.object(manager, 'is_pipeline_running', return_value=True):
            with mock.patch.object(manager, '_prepare_modules_then') as mock_prep:
                manager.runImgtransPipeline()
                mock_prep.assert_not_called()
                manager.progress_msgbox.show_fitted.assert_called_once()

    def test_run_blktrans_pipeline_rejected_when_running(self):
        manager = ModuleManager(self.mock_proj)
        manager.progress_msgbox = mock.MagicMock()

        with mock.patch.object(manager, 'is_pipeline_running', return_value=True):
            with mock.patch.object(manager, '_prepare_modules_then') as mock_prep:
                manager.runBlktransPipeline([], 1, [])
                mock_prep.assert_not_called()
                manager.progress_msgbox.show_fitted.assert_called_once()

    def test_canvas_inpaint_rejected_when_running(self):
        manager = ModuleManager(self.mock_proj)
        inpaint_dict = {'img': mock.MagicMock(), 'mask': mock.MagicMock()}

        with mock.patch.object(manager, 'is_pipeline_running', return_value=True):
            with mock.patch.object(manager, '_continue_canvas_inpaint') as mock_cont:
                manager.canvas_inpaint(inpaint_dict)
                mock_cont.assert_not_called()
                self.assertIsNone(manager._pending_canvas_inpaint)

    def test_mainwindow_run_button_toggle(self):
        # Verify slot logic on a mock mainwindow
        from ballontranslator.ui.mainwindow import MainWindow

        win = mock.MagicMock(spec=MainWindow)
        win.leftBar = mock.MagicMock()
        win.leftBar.runImgtransBtn = mock.MagicMock()
        win.backup_blkstyles = []
        win.module_manager = mock.MagicMock()
        win.tr = lambda s: s

        # When pipeline starts
        MainWindow.on_imgtrans_pipeline_started(win)
        win.leftBar.runImgtransBtn.setEnabled.assert_called_with(False)

        # When pipeline finishes
        MainWindow.on_imgtrans_pipeline_finished(win)
        win.leftBar.runImgtransBtn.setEnabled.assert_called_with(True)

    def test_mainwindow_run_imgtrans_blocked_when_running(self):
        from ballontranslator.ui.mainwindow import MainWindow

        win = mock.MagicMock(spec=MainWindow)
        win.module_manager = mock.MagicMock()
        win.module_manager.is_pipeline_running.return_value = True
        win.module_manager.progress_msgbox = mock.MagicMock()

        with mock.patch('ballontranslator.ui.mainwindow.RunPipelineDialog') as mock_dialog:
            MainWindow.run_imgtrans(win)
            mock_dialog.assert_not_called()
            win.module_manager.progress_msgbox.show_fitted.assert_called_once()

    def test_llm_image_requester_thread_safe_caching(self):
        requester = LLMImageRequester()
        profile1 = LLMProfile(id="p1", name="Profile 1", api_key="key-alpha", base_url="https://api.one")
        profile2 = LLMProfile(id="p2", name="Profile 2", api_key="key-beta", base_url="https://api.two")

        with mock.patch.object(requester, '_http_client') as mock_client_factory:
            client_a = mock.MagicMock()
            client_b = mock.MagicMock()
            mock_client_factory.side_effect = [client_a, client_b]

            with mock.patch.object(requester, '_request_param', return_value=''):
                c1 = requester._initialize_client(profile1, api_key="key-alpha")
                c2 = requester._initialize_client(profile2, api_key="key-beta")
                # Repeated call gets cached instance
                c1_cached = requester._initialize_client(profile1, api_key="key-alpha")

                self.assertIs(c1, client_a)
                self.assertIs(c2, client_b)
                self.assertIs(c1_cached, client_a)
                # Verify client_a was not closed when client_b was created
                client_a.close.assert_not_called()

        # Close closes all cached clients
        requester.close()
        client_a.close.assert_called_once()
        client_b.close.assert_called_once()
        self.assertEqual(len(requester._clients_cache), 0)

    def test_llm_image_requester_pinned_key_in_request_context(self):
        requester = LLMImageRequester()
        profile = LLMProfile(id="p", name="P", api_key="keys.txt")

        with mock.patch('ballontranslator.utils.project.key_pool.resolve_active_api_key') as mock_rot:
            mock_rot.side_effect = ["key_1", "key_2", "key_3"]

            # Without context: rotates each time
            k1 = requester._api_key_for_profile(profile)
            k2 = requester._api_key_for_profile(profile)
            self.assertEqual(k1, "key_1")
            self.assertEqual(k2, "key_2")

            # Under request context: pinned to active_api_key
            _REQUEST_CONTEXT.active_api_key = "pinned_key_abc"
            try:
                self.assertEqual(requester._api_key_for_profile(profile), "pinned_key_abc")
                self.assertEqual(requester._api_key_for_profile(profile), "pinned_key_abc")
            finally:
                _REQUEST_CONTEXT.active_api_key = None

            # After context cleared: resumes dynamic resolution
            k3 = requester._api_key_for_profile(profile)
            self.assertEqual(k3, "key_3")

    def test_pipeline_stopped_audits_only_runtime_issues_not_unprocessed_pages(self):
        from ballontranslator.utils.project.stage_auditor import StageIssue

        manager = ModuleManager(self.mock_proj)
        manager.imgtrans_thread = mock.MagicMock()
        manager.translate_thread = mock.MagicMock()
        manager.progress_msgbox = mock.MagicMock()

        # Stop requested
        manager.imgtrans_thread.isStopRequested.return_value = True
        real_issue = StageIssue(
            page_key="p1",
            stage="ocr",
            error_type="stage_failed",
            message="OCR failed on p1",
            page_index=0,
        )
        manager.imgtrans_thread.runtime_issues = [real_issue]
        manager.translate_thread.runtime_issues = []
        manager.imgtrans_thread.pages_to_process = ["p1", "p2", "p3", "p4", "p5"]

        with mock.patch('ballontranslator.utils.project.stage_auditor.StageAuditor.audit_project_pages') as mock_audit:
            # Simulate pipeline stop signal handler
            manager.on_imgtrans_thread_stopped()

            # StageAuditor must NOT be called on stop!
            mock_audit.assert_not_called()

            # Pipeline issues must ONLY contain real_issue, NOT the unreached pages p2-p5!
            self.assertEqual(len(manager.pipeline_issues), 1)
            self.assertEqual(manager.pipeline_issues[0].page_key, "p1")
            self.assertEqual(manager.pipeline_issues[0].stage, "ocr")

    def test_pipeline_finished_audits_with_stage_auditor_when_not_stopped(self):
        from ballontranslator.utils.project.stage_auditor import StageIssue

        manager = ModuleManager(self.mock_proj)
        manager.imgtrans_thread = mock.MagicMock()
        manager.translate_thread = mock.MagicMock()
        manager.progress_msgbox = mock.MagicMock()

        # Normal completion (not stopped)
        manager.imgtrans_thread.isStopRequested.return_value = False
        manager.imgtrans_thread.num_pages = 2
        manager.imgtrans_thread.detect_counter = 2
        manager.imgtrans_thread.ocr_counter = 2
        manager.imgtrans_thread.translate_counter = 2
        manager.imgtrans_thread.inpaint_counter = 2
        manager.imgtrans_thread.runtime_issues = []
        manager.translate_thread.runtime_issues = []
        manager.imgtrans_thread.pages_to_process = ["p1", "p2"]

        audited_issue = StageIssue(
            page_key="p2",
            stage="inpaint",
            error_type="stage_missing",
            message="Page p2 inpainting was not completed.",
            page_index=1,
        )

        with mock.patch('ballontranslator.utils.project.stage_auditor.StageAuditor.audit_project_pages', return_value=[audited_issue]) as mock_audit:
            manager.on_imgtrans_thread_finished()

            # StageAuditor IS called when not stopped
            mock_audit.assert_called_once()
            self.assertEqual(len(manager.pipeline_issues), 1)
            self.assertEqual(manager.pipeline_issues[0].page_key, "p2")
            self.assertEqual(manager.pipeline_issues[0].stage, "inpaint")

    def test_on_imgtrans_thread_finished_ignored_when_stop_requested(self):
        manager = ModuleManager(self.mock_proj)
        manager.imgtrans_thread = mock.MagicMock()
        manager.imgtrans_thread.isStopRequested.return_value = True

        with mock.patch.object(manager, 'finishImgtransPipeline') as mock_finish:
            manager.on_imgtrans_thread_finished()
            mock_finish.assert_not_called()


if __name__ == '__main__':
    unittest.main()

