import unittest
from qtpy.QtWidgets import QApplication
from qtpy.QtCore import Qt

from ballontranslator.utils.stage_auditor import (
    parse_page_ranges,
    format_page_ranges,
    is_safety_refusal_error,
    StageAuditor,
    StageIssue,
)
from ballontranslator.ui.page_range_progress import PageRangeProgressWidget
from ballontranslator.ui.pipeline_error_dialog import PipelineErrorDialog


class PageRangeAndAuditorTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_parse_and_format_page_ranges(self):
        # Basic parsing
        self.assertEqual(parse_page_ranges("1-5, 8, 12-14"), [1, 2, 3, 4, 5, 8, 12, 13, 14])
        self.assertEqual(parse_page_ranges("199"), [199])
        self.assertEqual(parse_page_ranges("5-2"), [2, 3, 4, 5])
        self.assertEqual(parse_page_ranges("  3 , 7-8 "), [3, 7, 8])

        # Clamping with max_pages
        self.assertEqual(parse_page_ranges("1-10", max_pages=5), [1, 2, 3, 4, 5])
        self.assertEqual(parse_page_ranges("200-215", max_pages=209), list(range(200, 210)))

        # Empty / invalid inputs
        self.assertEqual(parse_page_ranges(""), [])
        self.assertEqual(parse_page_ranges("   "), [])
        self.assertEqual(parse_page_ranges("abc, xyz"), [])

        # Formatting back to string
        self.assertEqual(format_page_ranges([1, 2, 3, 4, 5, 8, 12, 13, 14]), "1-5, 8, 12-14")
        self.assertEqual(format_page_ranges([199]), "199")
        self.assertEqual(format_page_ranges([]), "")

    def test_safety_refusal_detection(self):
        self.assertTrue(is_safety_refusal_error("I'm sorry, but I cannot fulfill this request due to safety policy."))
        self.assertTrue(is_safety_refusal_error("Request was refused by content policy"))
        self.assertFalse(is_safety_refusal_error("Connection timed out after 30 seconds"))

    def test_page_range_progress_widget_input_and_selection(self):
        page_names = [f"page_{i:03d}.png" for i in range(1, 210)]  # 209 pages
        widget = PageRangeProgressWidget(page_names, start=1, end=209, current_page=199)

        # Default text is empty, meaning ALL pages
        self.assertEqual(widget.custom_range_input.text(), "")
        self.assertEqual(widget.range_start.value(), 1)
        self.assertEqual(widget.range_end.value(), 209)
        self.assertEqual(len(widget.get_selected_pages()), 209)
        self.assertEqual(len(widget.get_selected_page_indices()), 209)

        # Test Multi-range custom input
        widget.custom_range_input.setText("1-3, 5, 199")
        self.assertEqual(
            widget.get_selected_pages(),
            ["page_001.png", "page_002.png", "page_003.png", "page_005.png", "page_199.png"],
        )
        self.assertEqual(widget.get_selected_page_indices(), [0, 1, 2, 4, 198])
        self.assertEqual(widget.range_start.value(), 1)
        self.assertEqual(widget.range_end.value(), 199)

        # Clear custom input defaults back to All pages
        widget.custom_range_input.clear()
        self.assertEqual(len(widget.get_selected_pages()), 209)
        self.assertEqual(widget.range_start.value(), 1)
        self.assertEqual(widget.range_end.value(), 209)

        # Single page input
        widget.custom_range_input.setText("199")
        self.assertEqual(widget.get_selected_pages(), ["page_199.png"])
        self.assertEqual(widget.range_start.value(), 199)
        self.assertEqual(widget.range_end.value(), 199)

        # Range bar sync
        widget.set_range(10, 20)
        self.assertEqual(widget.custom_range_input.text(), "10-20")
        self.assertEqual(len(widget.get_selected_pages()), 11)

        # Full range resets to empty (All)
        widget.set_range(1, 209)
        self.assertEqual(widget.custom_range_input.text(), "")
        self.assertEqual(len(widget.get_selected_pages()), 209)

    def test_pipeline_error_dialog_and_retry(self):
        issues = [
            StageIssue(
                page_key="page_005.png",
                stage="translate",
                error_type="safety_refusal",
                message="Refused by safety policy",
                page_index=4,
            ),
            StageIssue(
                page_key="page_008.png",
                stage="ocr",
                error_type="stage_failed",
                message="OCR timeout",
                page_index=7,
            ),
        ]

        retried_pages = []
        def on_retry(pages):
            retried_pages.extend(pages)

        dialog = PipelineErrorDialog(issues, on_retry=on_retry)
        self.assertEqual(len(dialog.issues), 2)

        # Test retry action
        dialog.retry_btn.click()
        self.assertEqual(retried_pages, ["page_005.png", "page_008.png"])

    def test_stage_auditor_detection(self):
        class MockBlock:
            def __init__(self, text="", translation=""):
                self.text = text
                self.translation = translation

        class MockProject:
            def __init__(self):
                self.pages = {
                    'p1.png': [MockBlock(text="Hello", translation="Xin chào")],
                    'p2.png': [MockBlock(text="World", translation="")],  # Missing translation
                    'p3.png': [],  # Missing detection
                }

            def pagename2idx(self, name):
                return list(self.pages.keys()).index(name)

            def get_page_finish_code(self, name):
                if name == 'p1.png':
                    return 15  # All stages finished
                elif name == 'p2.png':
                    return 3   # Det + OCR finished, trans & inpaint missing
                return 0

        proj = MockProject()
        issues = StageAuditor.audit_project_pages(proj)
        issue_map = {(iss.page_key, iss.stage): iss for iss in issues}

        # p1 has no issues
        self.assertNotIn(('p1.png', 'detect'), issue_map)
        self.assertNotIn(('p1.png', 'translate'), issue_map)

        # p2 is missing translation
        self.assertIn(('p2.png', 'translate'), issue_map)

        # p3 is missing detection
        self.assertIn(('p3.png', 'detect'), issue_map)

    def test_module_manager_issue_tracking(self):
        from ballontranslator.ui.module_manager import ModuleManager

        class MockProject:
            def __init__(self):
                self.pages = {}
            def pagename2idx(self, name):
                return 0

        manager = ModuleManager(MockProject())
        self.assertEqual(len(manager.pipeline_issues), 0)

        # Track received counts via signal
        received_counts = []
        manager.pipeline_issues_updated.connect(lambda count: received_counts.append(count))

        # Add simulated runtime issue
        issue = StageIssue(
            page_key="page_001.png",
            stage="ocr",
            error_type="stage_failed",
            message="OCR failed",
        )
        manager._pipeline_issues = [issue]
        manager.pipeline_issues_updated.emit(len(manager._pipeline_issues))

        self.assertEqual(len(manager.pipeline_issues), 1)
        self.assertEqual(received_counts[-1], 1)

        # Test clear
        manager.clear_pipeline_issues()
        self.assertEqual(len(manager.pipeline_issues), 0)
        self.assertEqual(received_counts[-1], 0)

    def test_llm_action_required_dialogs_compatibility(self):
        from unittest.mock import MagicMock, patch
        from PyQt6.QtWidgets import QWidget
        from ballontranslator.ui.mainwindow import MainWindow

        class DummyWindow(QWidget):
            def tr(self, s):
                return s
            def focus_llm_profile(self, *a, **k):
                pass

        dummy_window = DummyWindow()

        with patch('ballontranslator.ui.mainwindow.QMessageBox.exec') as mock_exec:
            mock_exec.return_value = None

            # Test on_show_llm_key_dialog
            MainWindow.on_show_llm_key_dialog(dummy_window, 'test_profile', 'Test Profile')

            # Test on_show_llm_model_dialog
            MainWindow.on_show_llm_model_dialog(dummy_window, 'test_profile', 'Test Profile', 'image_model')

            # Test on_show_llm_base_url_dialog (the exact method that aborted)
            MainWindow.on_show_llm_base_url_dialog(dummy_window, 'test_profile', 'Test Profile', 'image_base_url')

            self.assertEqual(mock_exec.call_count, 3)


class TestAgnesImageRequester(unittest.TestCase):
    def test_agnes_endpoint_and_payload(self):
        from ballontranslator.modules.llm_image import LLMImageRequester
        from ballontranslator.utils.llm_profiles import LLMProfile
        import numpy as np

        profile = LLMProfile(
            id='agnes-test',
            name='Agnes AI',
            base_url='https://apihub.agnes-ai.com/v1',
            image_model='agnes-image-2.0-flash',
            image_model_options=['agnes-image-2.0-flash'],
            require_api_key=False,
        )

        requester = LLMImageRequester()
        # Test base_url derivation when image_base_url is blank
        endpoint = requester._image_base_url(profile)
        self.assertEqual(endpoint, 'https://apihub.agnes-ai.com/v1/images/generations')

        # Test request body generation for image-to-image
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        img_file = requester._png_image_file(dummy_img)
        args = requester._agnes_api_args(profile, img_file)

        self.assertEqual(args['model'], 'agnes-image-2.0-flash')
        self.assertEqual(args['size'], '1024x1024')
        self.assertIn('extra_body', args)
        self.assertIn('image', args['extra_body'])
        self.assertTrue(args['extra_body']['image'][0].startswith('data:image/png;base64,'))
        self.assertEqual(args['extra_body']['response_format'], 'b64_json')


from unittest.mock import MagicMock


class TestParallelTranslationSync(unittest.TestCase):
    def test_imgtrans_thread_guards_num_pages_and_sync(self):
        from ballontranslator.ui.module_manager import ImgtransThread

        # Test that default num_pages is 0 and checkers safely return True when uninitialized
        thread = MagicMock(spec=ImgtransThread)
        thread.imgtrans_proj = None
        thread.num_pages = 0
        thread.detect_counter = 0

        self.assertTrue(ImgtransThread.detect_finished(thread))
        self.assertTrue(ImgtransThread.ocr_finished(thread))
        self.assertTrue(ImgtransThread.translate_finished(thread))
        self.assertTrue(ImgtransThread.inpaint_finished(thread))


if __name__ == '__main__':
    unittest.main()

