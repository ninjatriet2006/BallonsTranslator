from typing import Callable, List, Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..utils.stage_auditor import StageIssue
from .framelesswindow import DialogCloseButton, OutsideClickFramelessMixin


class PipelineErrorItemWidget(QFrame):
    """Card widget representing a single stage error or refusal."""

    def __init__(self, issue: StageIssue, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.issue = issue
        self.setObjectName('PipelineErrorItemWidget')
        self.setStyleSheet("""
            QFrame#PipelineErrorItemWidget {
                background-color: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 6px;
                padding: 8px 12px;
            }
            QFrame#PipelineErrorItemWidget:hover {
                border-color: rgba(239, 68, 68, 0.4);
                background-color: rgba(239, 68, 68, 0.06);
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        # Stage Tag
        stage_tag = QLabel(issue.stage.upper(), self)
        tag_color = "#ef4444" if issue.error_type == 'safety_refusal' else "#f59e0b"
        stage_tag.setStyleSheet(f"""
            background-color: {tag_color}22;
            color: {tag_color};
            border-radius: 3px;
            padding: 2px 6px;
            font-weight: bold;
            font-size: 11px;
        """)
        header_layout.addWidget(stage_tag)

        # Page Name
        page_label = QLabel(f"Page: {issue.page_key}", self)
        page_label.setStyleSheet("font-weight: bold; color: #e2e8f0; font-size: 12px;")
        header_layout.addWidget(page_label)
        header_layout.addStretch()

        layout.addLayout(header_layout)

        # Error Message
        msg_label = QLabel(issue.message, self)
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(msg_label)


class PipelineErrorDialog(OutsideClickFramelessMixin, QDialog):
    """Frameless card dialog displaying pipeline issues with retry/clear actions."""

    retry_requested = Signal(list)
    cleared = Signal()

    def __init__(
        self,
        issues: List[StageIssue],
        parent: Optional[QWidget] = None,
        on_retry: Optional[Callable[[List[str]], None]] = None,
    ):
        super().__init__(parent)
        self.issues = list(issues)
        self.on_retry = on_retry
        self.setObjectName('PipelineErrorDialog')
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumWidth(480)
        self.setMaximumWidth(600)
        self.setMaximumHeight(500)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        container = QFrame(self)
        container.setObjectName('PipelineErrorContainer')
        container.setStyleSheet("""
            QFrame#PipelineErrorContainer {
                background-color: rgb(24, 27, 34);
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 8px;
            }
        """)
        main_layout.addWidget(container)

        content_layout = QVBoxLayout(container)
        content_layout.setContentsMargins(16, 12, 16, 16)
        content_layout.setSpacing(12)

        # Title bar
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)

        title_label = QLabel(self.tr('Pipeline Issues'), container)
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #f8fafc;")
        title_layout.addWidget(title_label)

        count_badge = QLabel(f"{len(self.issues)} issue(s)", container)
        count_badge.setStyleSheet("color: #ef4444; font-size: 12px; font-weight: bold;")
        title_layout.addWidget(count_badge)
        title_layout.addStretch()

        close_btn = DialogCloseButton(container)
        close_btn.clicked.connect(self.reject)
        title_layout.addWidget(close_btn)

        content_layout.addLayout(title_layout)

        # Scroll area for issue cards
        scroll_area = QScrollArea(container)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollArea QWidget {
                background: transparent;
            }
        """)

        cards_widget = QWidget()
        cards_layout = QVBoxLayout(cards_widget)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(8)

        for issue in self.issues:
            card = PipelineErrorItemWidget(issue, cards_widget)
            cards_layout.addWidget(card)

        cards_layout.addStretch()
        scroll_area.setWidget(cards_widget)
        content_layout.addWidget(scroll_area, 1)

        # Actions row
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 4, 0, 0)
        btn_row.setSpacing(8)

        self.clear_btn = QPushButton(self.tr('Clear All'), container)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #94a3b8;
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 4px;
                padding: 6px 14px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.08);
                color: #e2e8f0;
            }
        """)
        self.clear_btn.clicked.connect(self._on_clear_clicked)
        btn_row.addWidget(self.clear_btn)

        btn_row.addStretch()

        self.retry_btn = QPushButton(self.tr('Retry Failed Pages'), container)
        self.retry_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        self.retry_btn.clicked.connect(self._on_retry_clicked)
        btn_row.addWidget(self.retry_btn)

        content_layout.addLayout(btn_row)

    def _on_retry_clicked(self):
        failed_pages = list(dict.fromkeys(iss.page_key for iss in self.issues if iss.page_key))
        if self.on_retry and failed_pages:
            self.on_retry(failed_pages)
        self.retry_requested.emit(failed_pages)
        self.accept()

    def _on_clear_clicked(self):
        self.issues.clear()
        self.cleared.emit()
        self.accept()
