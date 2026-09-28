"""Dialog for selecting which fonts to exclude from the font list."""
from typing import List
from qtpy.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QLineEdit, QListWidget, QPushButton, QAbstractItemView, QListWidgetItem,
    QMessageBox,
)
from qtpy.QtCore import Qt
from ballontranslator.ui.framelesswindow import (
    OutsideClickFramelessMixin,
    DialogCloseButton,
)
from ballontranslator.utils import shared
from ballontranslator.utils.shared import LEGACY_FONTS
from ballontranslator.utils.config import pcfg


class FontExcludeDialog(OutsideClickFramelessMixin, QDialog):
    """Dialog for selecting which fonts to exclude from the font list."""

    def __init__(self, parent: QWidget = None) -> None:
        window_type = getattr(Qt, 'WindowType', Qt)
        super().__init__(
            parent,
            window_type.Dialog | window_type.FramelessWindowHint,
        )
        self.setObjectName('FontExcludeDialog')
        self.setWindowTitle(self.tr("Font Exclusion"))
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setMinimumSize(640, 440)
        self.resize(760, 540)
        widget_attribute = getattr(Qt, 'WidgetAttribute', Qt)
        self.setAttribute(widget_attribute.WA_TranslucentBackground)
        self.setAttribute(widget_attribute.WA_DeleteOnClose)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(5, 5, 5, 5)

        surface = QFrame(self)
        surface.setObjectName('FontExcludeSurface')
        root_layout.addWidget(surface)

        layout = QVBoxLayout(surface)
        layout.setContentsMargins(22, 16, 22, 18)
        layout.setSpacing(14)

        self.title_bar = QWidget(surface)
        self.title_bar.setObjectName('FontExcludeTitleBar')
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(0, 0, 0, 0)
        self.title_label = QLabel(self.tr('Font Exclusion'), self.title_bar)
        self.title_label.setObjectName('FontExcludeTitle')
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        self.close_button = DialogCloseButton(self.title_bar)
        self.close_button.clicked.connect(self.reject)
        title_layout.addWidget(self.close_button)
        layout.addWidget(self.title_bar)

        # Search bar
        self.search_edit = QLineEdit(surface)
        self.search_edit.setObjectName('FontExcludeSearch')
        self.search_edit.setPlaceholderText(self.tr("Filter Fonts"))
        self.search_edit.textChanged.connect(self._filter_lists)
        layout.addWidget(self.search_edit)

        # Side-by-side list widgets
        lists_layout = QHBoxLayout()

        # Available fonts list
        left_layout = QVBoxLayout()
        left_layout.setSpacing(6)
        available_title = QLabel(self.tr("Available Fonts"), surface)
        available_title.setObjectName('FontExcludeListTitle')
        left_layout.addWidget(available_title)
        self.available_list = QListWidget(surface)
        self.available_list.setObjectName('FontExcludeAvailableList')
        self._configure_font_list(self.available_list)
        left_layout.addWidget(self.available_list)
        lists_layout.addLayout(left_layout)

        # Center buttons
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(8)
        btn_layout.addStretch()
        self.hide_btn = QPushButton(">", surface)
        self.hide_btn.setObjectName('FontExcludeMoveButton')
        self.hide_btn.setFixedWidth(34)
        self.hide_btn.setToolTip(self.tr("Hide selected fonts"))
        self.hide_btn.clicked.connect(self._hide_fonts)
        btn_layout.addWidget(self.hide_btn)
        self.show_btn = QPushButton("<", surface)
        self.show_btn.setObjectName('FontExcludeMoveButton')
        self.show_btn.setFixedWidth(34)
        self.show_btn.setToolTip(self.tr("Show selected fonts"))
        self.show_btn.clicked.connect(self._show_fonts)
        btn_layout.addWidget(self.show_btn)
        btn_layout.addStretch()
        lists_layout.addLayout(btn_layout)

        # Excluded fonts list
        right_layout = QVBoxLayout()
        right_layout.setSpacing(6)
        excluded_title = QLabel(self.tr("Hidden Fonts"), surface)
        excluded_title.setObjectName('FontExcludeListTitle')
        right_layout.addWidget(excluded_title)
        self.excluded_list = QListWidget(surface)
        self.excluded_list.setObjectName('FontExcludeHiddenList')
        self._configure_font_list(self.excluded_list)
        right_layout.addWidget(self.excluded_list)
        lists_layout.addLayout(right_layout)

        layout.addLayout(lists_layout)

        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(0, 4, 0, 0)
        action_layout.setSpacing(8)

        self.legacy_btn = QPushButton(
            self.tr("Hide Legacy Fonts"),
            surface,
        )
        self.legacy_btn.setObjectName('FontExcludeLegacyButton')
        self.legacy_btn.clicked.connect(self._on_add_legacy_fonts)
        action_layout.addWidget(self.legacy_btn)
        action_layout.addStretch()

        self.cancel_button = QPushButton(self.tr('Cancel'), surface)
        self.cancel_button.setObjectName('FontExcludeSecondaryButton')
        self.cancel_button.clicked.connect(self.reject)
        action_layout.addWidget(self.cancel_button)
        self.ok_button = QPushButton(self.tr('OK'), surface)
        self.ok_button.setObjectName('FontExcludePrimaryButton')
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self.accept)
        action_layout.addWidget(self.ok_button)
        layout.addLayout(action_layout)

        # Populate lists
        self._populate_lists()

    @staticmethod
    def _configure_font_list(list_widget: QListWidget) -> None:
        list_widget.setUniformItemSizes(True)
        list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )

    def _add_font_item(self, list_widget: QListWidget, font_name: str) -> None:
        """Add a font name to a list widget.

        Legacy fonts get a "[Legacy]" suffix. The original font name is stored
        in ``Qt.UserRole``.
        """
        is_legacy = font_name in LEGACY_FONTS
        display = f"{font_name} [{self.tr('Legacy')}]" if is_legacy else font_name
        item = QListWidgetItem(display)
        item.setData(Qt.ItemDataRole.UserRole, font_name)
        list_widget.addItem(item)

    def _dismiss_transient_window(self) -> None:
        self.reject()

    @staticmethod
    def _real_name(item: QListWidgetItem) -> str:
        """Return the original font name stored in UserRole."""
        name = item.data(Qt.ItemDataRole.UserRole)
        return name if name else item.text()

    def _populate_lists(self) -> None:
        self.available_list.clear()
        self.excluded_list.clear()

        for font in shared.get_filtered_font_list(shared.FONT_FAMILIES, pcfg.excluded_fonts):
            self._add_font_item(self.available_list, font)

        for font in sorted(pcfg.excluded_fonts, key=str.casefold):
            self._add_font_item(self.excluded_list, font)

    def _filter_lists(self) -> None:
        text = self.search_edit.text().casefold()
        for list_widget in (self.available_list, self.excluded_list):
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                hidden = bool(text) and text not in self._real_name(item).casefold()
                if item.isHidden() != hidden:
                    item.setHidden(hidden)
                if hidden:
                    item.setSelected(False)

    def _move_selected(
        self,
        source: QListWidget,
        target: QListWidget,
    ) -> None:
        for item in source.selectedItems():
            if not item.isHidden():
                target.addItem(source.takeItem(source.row(item)))
        target.sortItems(Qt.SortOrder.AscendingOrder)
        self._filter_lists()

    def _hide_fonts(self) -> None:
        self._move_selected(self.available_list, self.excluded_list)

    def _show_fonts(self) -> None:
        self._move_selected(self.excluded_list, self.available_list)

    def _on_add_legacy_fonts(self) -> None:
        """Detect legacy Windows fonts and add them to the hidden list automatically."""
        exist_legacy = shared.FONT_FAMILIES & LEGACY_FONTS
        already_excluded = {
            self._real_name(self.excluded_list.item(i))
            for i in range(self.excluded_list.count())
        }
        to_add = sorted(exist_legacy - already_excluded)

        if not to_add:
            QMessageBox.information(
                self,
                self.tr("Legacy Fonts"),
                self.tr("No additional legacy fonts detected on this system."),
            )
            return

        for font_name in to_add:
            for i in range(self.available_list.count()):
                if self._real_name(self.available_list.item(i)) == font_name:
                    self.excluded_list.addItem(self.available_list.takeItem(i))
                    break
            else:
                self._add_font_item(self.excluded_list, font_name)

        self.excluded_list.sortItems(Qt.SortOrder.AscendingOrder)
        self._filter_lists()

        QMessageBox.information(
            self,
            self.tr("Legacy Fonts"),
            self.tr(
                "Added {count} legacy font(s) to the hidden list:\n\n{fonts}"
            ).replace("{count}", str(len(to_add))).replace("{fonts}", "\n".join(to_add)),
        )

    def get_excluded_fonts(self) -> List[str]:
        return sorted(
            {
                self._real_name(self.excluded_list.item(i))
                for i in range(self.excluded_list.count())
            },
            key=str.casefold,
        )
