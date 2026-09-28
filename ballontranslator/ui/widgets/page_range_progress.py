from qtpy.QtCore import QPoint, QRect, QRectF, QSignalBlocker, QSize, Qt, Signal
from qtpy.QtGui import QColor, QFont, QPainter, QPainterPath, QPalette, QPen
from qtpy.QtWidgets import (
    QAbstractSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ballontranslator.utils.stage_auditor import parse_page_ranges
from ballontranslator.ui.custom_widget.helper import borderColor, themeColor, widgetBackgroundColor
from ballontranslator.ui.icon_rendering import render_svg_pixmap
from ballontranslator.ui.misc import themed_icon_path


_ACCENT_COLOR = QColor(30, 147, 229)


class PageRangeSpinBox(QSpinBox):
    """Compact page selector with horizontal SVG step chevrons.

    >>> PageRangeSpinBox.__name__
    'PageRangeSpinBox'
    """

    ICON_SIZE = 12

    def __init__(self, parent=None):
        super().__init__(parent)
        button_symbols = getattr(QAbstractSpinBox, 'ButtonSymbols', QAbstractSpinBox)
        self.setButtonSymbols(button_symbols.NoButtons)
        self.setMouseTracking(True)
        self._hover_button = ''

    def _button_rects(self):
        button_size = 16
        gap = 1
        right = self.width() - 4
        y = (self.height() - button_size) // 2
        up_rect = QRect(right - button_size, y, button_size, button_size)
        down_rect = QRect(
            up_rect.left() - gap - button_size,
            y,
            button_size,
            button_size,
        )
        return up_rect, down_rect

    @staticmethod
    def _event_pos(event) -> QPoint:
        if hasattr(event, 'position'):
            return event.position().toPoint()
        return event.pos()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        render_hint = getattr(QPainter, 'RenderHint', QPainter)
        painter.setRenderHint(render_hint.Antialiasing, True)
        up_rect, down_rect = self._button_rects()
        for name, rect, icon_name in (
            ('down', down_rect, 'chevron-down.svg'),
            ('up', up_rect, 'chevron-up.svg'),
        ):
            if self._hover_button == name and self.isEnabled():
                hover = QColor(_ACCENT_COLOR)
                hover.setAlpha(32)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(hover)
                painter.drawRoundedRect(QRectF(rect), 3, 3)
            pixmap = render_svg_pixmap(
                themed_icon_path(icon_name),
                self.ICON_SIZE,
                self.ICON_SIZE,
                self.devicePixelRatioF(),
            )
            x = rect.center().x() - self.ICON_SIZE // 2
            y = rect.center().y() - self.ICON_SIZE // 2
            painter.drawPixmap(x, y, pixmap)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self._event_pos(event)
            up_rect, down_rect = self._button_rects()
            if up_rect.contains(pos):
                self.stepUp()
                event.accept()
                return
            if down_rect.contains(pos):
                self.stepDown()
                event.accept()
                return
        return super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = self._event_pos(event)
        up_rect, down_rect = self._button_rects()
        hover_button = (
            'up'
            if up_rect.contains(pos)
            else 'down'
            if down_rect.contains(pos)
            else ''
        )
        if hover_button != self._hover_button:
            self._hover_button = hover_button
            self.update()
        return super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self._hover_button:
            self._hover_button = ''
            self.update()
        return super().leaveEvent(event)


class PageProgressRangeBar(QWidget):
    """Paint discontinuous page completion with a draggable inclusive range.

    >>> PageProgressRangeBar.__name__
    'PageProgressRangeBar'
    """

    range_changed = Signal(int, int)

    TRACK_HEIGHT = 5
    TRACK_Y = 18
    HANDLE_RADIUS = 7
    TRACK_SIDE_MARGIN = 0
    HEIGHT = 40

    def __init__(self, page_names, parent=None):
        super().__init__(parent)
        self.page_names = list(page_names)
        self.finished_pages = [False] * len(self.page_names)
        self.start_index = 0
        self.end_index = max(0, len(self.page_names) - 1)
        self.hover_page_index = -1
        self._active_handle = ''
        self._hover_handle_index = -1
        self.setMouseTracking(True)
        self.setFixedHeight(self.HEIGHT)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

    @property
    def page_count(self):
        return len(self.page_names)

    @property
    def finished_count(self):
        return sum(self.finished_pages)

    def sizeHint(self):
        return QSize(280, self.HEIGHT)

    def set_finished_pages(self, finished_pages):
        finished = [bool(value) for value in finished_pages]
        page_count = self.page_count
        self.finished_pages = (finished + [False] * page_count)[:page_count]
        self.update()

    def set_range(self, start: int, end: int, emit: bool = True):
        if not self.page_names:
            return
        start = max(1, min(int(start), self.page_count))
        end = max(start, min(int(end), self.page_count))
        changed = (start - 1, end - 1) != (self.start_index, self.end_index)
        self.start_index = start - 1
        self.end_index = end - 1
        if changed:
            self.update()
            if emit:
                self.range_changed.emit(start, end)

    def _track_rect(self):
        margin = max(self.HANDLE_RADIUS + 2, self.TRACK_SIDE_MARGIN)
        return QRectF(
            margin,
            self.TRACK_Y,
            max(1, self.width() - margin * 2),
            self.TRACK_HEIGHT,
        )

    def _page_x(self, page_index: int):
        track = self._track_rect()
        if not self.page_names:
            return track.left()
        if self.page_count == 1:
            return track.center().x()
        step_width = track.width() / (self.page_count - 1)
        return track.left() + page_index * step_width

    def _page_index_at_x(self, x: float):
        if not self.page_names:
            return -1
        if self.page_count == 1:
            return 0
        track = self._track_rect()
        normalized = (x - track.left()) / max(1.0, track.width())
        page_index = round(normalized * (self.page_count - 1))
        return max(0, min(page_index, self.page_count - 1))

    def _selection_rect(self, track: QRectF):
        start_x = self._page_x(self.start_index)
        end_x = self._page_x(self.end_index)
        return QRectF(
            start_x,
            track.top(),
            max(0.0, end_x - start_x),
            track.height(),
        )

    @staticmethod
    def _event_pos(event):
        return event.position() if hasattr(event, 'position') else event.pos()

    def _handle_index_at(self, pos):
        if not self.page_names:
            return -1
        track_y = self._track_rect().center().y()
        hit_radius = self.HANDLE_RADIUS + 2
        for page_index in {self.start_index, self.end_index}:
            dx = pos.x() - self._page_x(page_index)
            dy = pos.y() - track_y
            if dx * dx + dy * dy <= hit_radius * hit_radius:
                return page_index
        return -1

    def _set_hover_position(self, pos):
        handle_index = self._handle_index_at(pos)
        if handle_index >= 0:
            hover_index = handle_index
        elif self._track_rect().adjusted(0, -5, 0, 5).contains(pos):
            hover_index = self._page_index_at_x(pos.x())
        else:
            hover_index = -1

        changed = (
            hover_index != self.hover_page_index
            or handle_index != self._hover_handle_index
        )
        self.hover_page_index = hover_index
        self._hover_handle_index = handle_index
        if changed:
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        render_hint = getattr(QPainter, 'RenderHint', QPainter)
        painter.setRenderHint(render_hint.Antialiasing, True)
        track = self._track_rect()
        color_role = getattr(QPalette, 'ColorRole', QPalette)
        empty_color = self.palette().color(color_role.Mid)
        empty_color.setAlpha(105)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(empty_color)
        painter.drawRoundedRect(track, self.TRACK_HEIGHT / 2, self.TRACK_HEIGHT / 2)

        if self.page_names:
            clip_path = QPainterPath()
            clip_path.addRoundedRect(
                track,
                self.TRACK_HEIGHT / 2,
                self.TRACK_HEIGHT / 2,
            )
            painter.save()
            painter.setClipPath(clip_path)
            segment_width = track.width() / self.page_count
            selection = QColor(_ACCENT_COLOR)
            selection.setAlpha(38)
            selection_rect = self._selection_rect(track)
            painter.fillRect(selection_rect, selection)
            for index, finished in enumerate(self.finished_pages):
                if not finished:
                    continue
                segment = QRectF(
                    track.left() + index * segment_width,
                    track.top(),
                    segment_width + 0.5,
                    track.height(),
                )
                painter.fillRect(segment, _ACCENT_COLOR)
            painter.restore()

            for index in {self.start_index, self.end_index}:
                center_x = self._page_x(index)
                center_y = track.center().y()
                handle_rect = QRectF(
                    center_x - self.HANDLE_RADIUS,
                    center_y - self.HANDLE_RADIUS,
                    self.HANDLE_RADIUS * 2,
                    self.HANDLE_RADIUS * 2,
                )
                painter.setPen(QPen(borderColor(), 1))
                painter.setBrush(widgetBackgroundColor())
                painter.drawEllipse(handle_rect)

                is_active = (
                    self._active_handle == 'overlap'
                    or self._active_handle == 'start' and index == self.start_index
                    or self._active_handle == 'end' and index == self.end_index
                )
                inner_radius = (
                    3
                    if is_active
                    else 4.5
                    if index == self._hover_handle_index
                    else 3.5
                )
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(themeColor())
                painter.drawEllipse(
                    QRectF(
                        center_x - inner_radius,
                        center_y - inner_radius,
                        inner_radius * 2,
                        inner_radius * 2,
                    )
                )

        if self.hover_page_index >= 0:
            self._paint_hover_info(
                painter,
                track,
                draw_line=(
                    not self._active_handle
                    and self._hover_handle_index < 0
                ),
            )
        painter.end()

    def _paint_hover_info(
        self,
        painter: QPainter,
        track: QRectF,
        draw_line: bool = True,
    ):
        page_index = self.hover_page_index
        x = self._page_x(page_index)
        if draw_line:
            line_color = QColor(_ACCENT_COLOR)
            line_color.setAlpha(190)
            painter.setPen(QPen(line_color, 1))
            painter.drawLine(
                int(x),
                13,
                int(x),
                int(track.bottom() + 13),
            )

        font = QFont(self.font())
        font.setPixelSize(12)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text_color = self.palette().color(QPalette.ColorRole.WindowText)
        text_color.setAlpha(205)
        painter.setPen(text_color)
        page_name = metrics.elidedText(
            self.page_names[page_index],
            Qt.TextElideMode.ElideMiddle,
            min(180, max(60, self.width() // 2)),
        )
        name_width = metrics.horizontalAdvance(page_name)
        name_x = max(2, min(int(x - name_width / 2), self.width() - name_width - 2))
        painter.drawText(name_x, metrics.ascent() + 1, page_name)

        index_text = str(page_index + 1)
        index_width = metrics.horizontalAdvance(index_text)
        painter.drawText(
            int(x - index_width / 2),
            int(track.bottom() + metrics.ascent() + 3),
            index_text,
        )

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self.page_names:
            return super().mousePressEvent(event)
        pos = self._event_pos(event)
        start_distance = abs(pos.x() - self._page_x(self.start_index))
        end_distance = abs(pos.x() - self._page_x(self.end_index))
        if self.start_index == self.end_index:
            self._active_handle = 'overlap'
        else:
            self._active_handle = 'start' if start_distance <= end_distance else 'end'
        self._move_active_handle(self._page_index_at_x(pos.x()))
        event.accept()

    def mouseMoveEvent(self, event):
        pos = self._event_pos(event)
        if self._active_handle:
            self._move_active_handle(self._page_index_at_x(pos.x()))
            event.accept()
            return
        self._set_hover_position(pos)
        return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._active_handle:
            self._active_handle = ''
            self._set_hover_position(self._event_pos(event))
            self.update()
            event.accept()
            return
        return super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self.hover_page_index = -1
        self._active_handle = ''
        self._hover_handle_index = -1
        self.update()
        return super().leaveEvent(event)

    def _move_active_handle(self, page_index: int):
        old_range = (self.start_index, self.end_index)
        if self._active_handle == 'overlap':
            if page_index < self.start_index:
                self._active_handle = 'start'
            elif page_index > self.end_index:
                self._active_handle = 'end'
            else:
                self.hover_page_index = self.start_index
                self._hover_handle_index = self.start_index
                self.update()
                return
        if self._active_handle == 'start':
            self.start_index = min(page_index, self.end_index)
            self.hover_page_index = self.start_index
            self._hover_handle_index = self.start_index
        elif self._active_handle == 'end':
            self.end_index = max(page_index, self.start_index)
            self.hover_page_index = self.end_index
            self._hover_handle_index = self.end_index
        self.update()
        if old_range != (self.start_index, self.end_index):
            self.range_changed.emit(self.start_index + 1, self.end_index + 1)


class _RangeSpinBoxStub:
    """Compatibility shim providing .value() for code expecting spinboxes."""
    def __init__(self, get_val_fn):
        self._get_val_fn = get_val_fn

    def value(self) -> int:
        return self._get_val_fn()

    def setValue(self, val: int):
        pass


class PageRangeProgressWidget(QWidget):
    """Clean page selection widget using a single multi-range input and visual progress track.

    >>> PageRangeProgressWidget.__name__
    'PageRangeProgressWidget'
    """

    range_changed = Signal(int, int)

    def __init__(self, page_names, start=1, end=None, current_page=None, parent=None):
        super().__init__(parent)
        self.setObjectName('RunPipelinePageRangeProgress')
        self.page_names = list(page_names)
        self.current_page = current_page

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        range_row = QWidget(self)
        range_row.setObjectName('RunPipelineGeneralSettingRow')
        range_layout = QHBoxLayout(range_row)
        range_layout.setContentsMargins(2, 0, 2, 0)
        range_layout.setSpacing(6)

        range_label = QLabel(self.tr('Pages to Run'), range_row)
        range_label.setObjectName('RunPipelineSettingLabel')
        range_label.setFixedWidth(80)
        range_layout.addWidget(range_label)

        self.custom_range_input = QLineEdit(range_row)
        self.custom_range_input.setObjectName('RunPipelineCustomRangeInput')
        self.custom_range_input.setPlaceholderText(self.tr('e.g. 1-5, 8, 12-15 (Leave empty for All)'))
        range_layout.addWidget(self.custom_range_input, 1)

        self.progress_label = QLabel(range_row)
        self.progress_label.setObjectName('RunPipelineSettingLabel')
        self.progress_label.setTextFormat(Qt.TextFormat.RichText)
        range_layout.addWidget(self.progress_label)
        layout.addWidget(range_row)

        self.range_bar = PageProgressRangeBar(self.page_names, self)
        page_count = len(self.page_names)
        maximum = max(1, page_count)
        saved_end = maximum if end is None else end
        start = max(1, min(int(start), maximum))
        saved_end = max(start, min(int(saved_end), maximum))

        # Default is empty (meaning All pages).
        # Only populate if a specific sub-range was requested
        if start != 1 or saved_end != maximum:
            if start == saved_end:
                self.custom_range_input.setText(str(start))
            else:
                self.custom_range_input.setText(f"{start}-{saved_end}")

        self.range_bar.set_range(start, saved_end, emit=False)
        self._update_progress_label()
        layout.addWidget(self.range_bar)

        self.range_bar.range_changed.connect(self._on_bar_range_changed)
        self.custom_range_input.textChanged.connect(self._on_custom_range_changed)

    @property
    def range_start(self):
        return _RangeSpinBoxStub(lambda: self.range_bar.start_index + 1)

    @property
    def range_end(self):
        return _RangeSpinBoxStub(lambda: self.range_bar.end_index + 1)

    def _on_custom_range_changed(self, text: str):
        text = text.strip()
        page_count = len(self.page_names)
        if not text:
            self.range_bar.set_range(1, max(1, page_count), emit=False)
            self._update_progress_label()
            self.range_changed.emit(1, max(1, page_count))
            return

        parsed = parse_page_ranges(text, max_pages=page_count)
        if parsed:
            min_p = min(parsed)
            max_p = max(parsed)
            self.range_bar.set_range(min_p, max_p, emit=False)
            self.range_changed.emit(min_p, max_p)
        self._update_progress_label()

    def _on_bar_range_changed(self, start: int, end: int):
        page_count = len(self.page_names)
        self.custom_range_input.blockSignals(True)
        if start == 1 and end == page_count:
            self.custom_range_input.clear()
        elif start == end:
            self.custom_range_input.setText(str(start))
        else:
            self.custom_range_input.setText(f"{start}-{end}")
        self.custom_range_input.blockSignals(False)
        self._update_progress_label()
        self.range_changed.emit(start, end)

    def get_selected_pages(self):
        if not self.page_names:
            return []
        text = self.custom_range_input.text().strip()
        if not text:
            return list(self.page_names)
        parsed = parse_page_ranges(text, max_pages=len(self.page_names))
        if parsed:
            return [self.page_names[p - 1] for p in parsed if 1 <= p <= len(self.page_names)]
        return list(self.page_names)

    def get_selected_page_indices(self):
        if not self.page_names:
            return []
        text = self.custom_range_input.text().strip()
        if not text:
            return list(range(len(self.page_names)))
        parsed = parse_page_ranges(text, max_pages=len(self.page_names))
        if parsed:
            return [p - 1 for p in parsed if 1 <= p <= len(self.page_names)]
        return list(range(len(self.page_names)))

    def set_range(self, start: int, end: int):
        page_count = len(self.page_names)
        self.custom_range_input.blockSignals(True)
        if start <= 1 and end >= page_count:
            self.custom_range_input.clear()
        elif start == end:
            self.custom_range_input.setText(str(start))
        else:
            self.custom_range_input.setText(f"{start}-{end}")
        self.custom_range_input.blockSignals(False)
        self.range_bar.set_range(start, end, emit=False)
        self._update_progress_label()

    def set_finished_pages(self, finished_pages):
        self.range_bar.set_finished_pages(finished_pages)
        self._update_progress_label()

    def _update_progress_label(self):
        finished_count = self.range_bar.finished_count
        text = self.custom_range_input.text().strip() if hasattr(self, 'custom_range_input') else ''
        if text and len(self.page_names) > 0:
            parsed = parse_page_ranges(text, max_pages=len(self.page_names))
            sel_count = len(parsed)
            sel_str = f' <span style="color: rgb(30, 147, 229);">({sel_count} {self.tr("sel")})</span>'
        else:
            sel_str = ''

        self.progress_label.setText(
            f'{self.tr("progress")} '
            f'<span style="color: rgb(30, 147, 229);">{finished_count}</span>'
            f'/{self.range_bar.page_count}'
            f'{sel_str}'
        )
