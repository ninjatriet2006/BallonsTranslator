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


if __name__ == '__main__':
    unittest.main()
