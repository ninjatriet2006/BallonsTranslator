from typing import Callable, List, Dict, Optional
import time
import datetime

from qtpy.QtWidgets import (
    QDialog,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QMessageBox,
    QSizePolicy,
    QProgressBar,
    QPushButton,
    QWidget,
)
from qtpy.QtGui import (
    QCloseEvent,
    QShowEvent,
    QMoveEvent,
    QPainter,
    QPainterPath,
    QColor,
    QPaintEvent,
)
from qtpy.QtCore import Qt, Signal, QRectF

from ballontranslator.utils.project.shared import remove_from_runtime_widget_set, add_to_runtime_widget_set
from .widget import Widget


class MessageBox(QMessageBox):

    def __init__(self, info_msg: str = None, btn_type = QMessageBox.StandardButton.Ok, frame_less: bool = False, modal: bool = False, signal_slot_map_list: List[Dict] = None, *args, **kwargs):
        super().__init__(text=info_msg, *args, **kwargs)
        self.register_signal_slot_map = []
        add_to_runtime_widget_set(self)

        if frame_less:
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        if modal:
            self.setModal(modal)
        if btn_type is not None:
            self.setStandardButtons(btn_type)

        if signal_slot_map_list is not None:
            self.connect_signals(signal_slot_map_list)

    def connect_signals(self, signal_slot_map_list):
        if signal_slot_map_list is None:
            return
        if isinstance(signal_slot_map_list, dict):
            signal_slot_map_list = [signal_slot_map_list]
        for signal_slot_map in signal_slot_map_list:
            slot = signal_slot_map['slot']
            if isinstance(slot, Callable):
                slot_func = slot
            else:
                assert isinstance(slot, str)
                slot_func = getattr(self, slot)
            signal_slot_map['signal'].connect(slot_func)
            signal_slot_map['slot_func'] = slot_func
            self.register_signal_slot_map.append(signal_slot_map)

    def disconnect_all(self):
        # https://stackoverflow.com/a/48501804/17671327
        for signal_slot_map in self.register_signal_slot_map:
            signal_slot_map['signal'].disconnect(signal_slot_map['slot_func'])
        self.register_signal_slot_map.clear()

    def clear_before_close(self):
        remove_from_runtime_widget_set(self)
        self.disconnect_all()

    def done(self, v: int = 0):
        self.clear_before_close()
        super().done(v)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.clear_before_close()
        return super().closeEvent(event)


class SegmentedProgressBar(QWidget):
    """Modern progress bar capable of showing dual-segmented progress:
    Green for successful items, Red for failed/fault-tolerated items,
    and neutral track for remaining unprocessed items.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.success_ratio: float = 0.0
        self.error_ratio: float = 0.0
        self.setFixedHeight(10)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def setValues(self, success_ratio: float, error_ratio: float = 0.0) -> None:
        self.success_ratio = max(0.0, min(1.0, float(success_ratio)))
        self.error_ratio = max(0.0, min(1.0 - self.success_ratio, float(error_ratio)))
        self.update()

    def setValue(self, value: int) -> None:
        """Backward-compatible setter for 0-100 integer progress."""
        self.setValues(value / 100.0, 0.0)

    def paintEvent(self, event: QPaintEvent) -> None:
        rect = self.rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return

        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            rect_f = QRectF(rect)
            radius = rect.height() / 2.0

            # Background track
            bg_color = QColor(128, 128, 128, 60)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(bg_color)
            painter.drawRoundedRect(rect_f, radius, radius)

            total_w = rect.width()
            sw = int(total_w * self.success_ratio)
            ew = int(total_w * self.error_ratio)

            if sw <= 0 and ew <= 0:
                return

            path = QPainterPath()
            path.addRoundedRect(rect_f, radius, radius)
            painter.setClipPath(path)

            # Success portion (Green)
            if sw > 0:
                painter.setBrush(QColor('#2ea043'))
                painter.drawRect(0, 0, sw, rect.height())

            # Error portion (Red)
            if ew > 0:
                painter.setBrush(QColor('#da3633'))
                painter.drawRect(sw, 0, ew, rect.height())
        except Exception:
            pass
        finally:
            painter.end()


class TaskProgressBar(Widget):
    def __init__(self, description: str = '', verbose=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.progressbar = SegmentedProgressBar(self)
        self.textlabel = QLabel(self)
        self.description = description
        layout = QVBoxLayout(self)

        self.verbose = verbose
        
        if verbose:
            self.start_time = 0
            self.verbose_label = QLabel(self)
            hl = QHBoxLayout()
            hl.addWidget(self.textlabel)
            hl.addStretch(1)
            hl.addWidget(self.verbose_label)
            layout.addLayout(hl)
        else:
            layout.addWidget(self.textlabel)
            
        layout.addWidget(self.progressbar)
        self.updateProgress(0)

    def updateProgress(
        self,
        progress: int,
        msg: str = '',
        verbose_msg: str = None,
        error_count: int = 0,
        total_count: int = 0,
    ):
        if total_count > 0:
            error_ratio = min(1.0, error_count / float(total_count))
            total_ratio = min(1.0, progress / 100.0)
            success_ratio = max(0.0, total_ratio - error_ratio)
            self.progressbar.setValues(success_ratio, error_ratio)
        else:
            self.progressbar.setValue(progress)

        full_msg = self.description + msg if self.description else msg

        if error_count > 0:
            err_tag = f" <span style='color: #da3633; font-weight: bold;'>({error_count} {self.tr('failed')})</span>"
            self.textlabel.setText(f"{full_msg}{err_tag}")
        else:
            self.textlabel.setText(full_msg)

        if self.verbose:
            if verbose_msg is not None:
                self.verbose_label.setText(verbose_msg)
            elif progress == 0:
                self.verbose_label.setText('')
                self.start_time = time.time()
            elif progress == 100:
                if error_count > 0:
                    self.verbose_label.setText(f"<span style='color: #da3633; font-weight: 500;'>{error_count} {self.tr('failed')}</span>")
                else:
                    self.verbose_label.setText(self.tr('100% Done'))
            else:
                cur_time = time.time()
                left_progress = 100 - progress
                eta = left_progress / progress * (cur_time - self.start_time + 1e-6)
                eta = datetime.timedelta(seconds=int(round(eta)))
                err_suffix = f" <span style='color: #da3633;'>[{error_count} err]</span>" if error_count > 0 else ""
                added_str = f'{progress}% ETA {eta}{err_suffix}'
                self.verbose_label.setText(added_str)


class FrameLessMessageBox(QMessageBox):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        

class ProgressMessageBox(QDialog):
    showed = Signal()
    stop_clicked = Signal()
    
    def __init__(self, task_name: str = None, show_stop_btn: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(20, 10, 20, 30)

        self.task_progress_bar: TaskProgressBar = None
        if task_name is not None:
            self.task_progress_bar = TaskProgressBar(task_name, True)
            layout.addWidget(self.task_progress_bar)

        # Generic progress dialogs stay unchanged unless a caller opts into Stop.
        self.stop_button = None
        if show_stop_btn:
            self.stop_button = QPushButton(self.tr('Stop'), self)
            self.stop_button.clicked.connect(self.on_stop_clicked)
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            button_layout.addWidget(self.stop_button)
            button_layout.addStretch()
            layout.addLayout(button_layout)

    def on_stop_clicked(self):
        self.stop_clicked.emit()
        if self.stop_button is not None:
            self.stop_button.setEnabled(False)
            self.stop_button.setText(self.tr('trying to stop...'))

    def updateTaskProgress(self, value: int, msg: str = '', verbose_msg: str = None):
        if self.task_progress_bar is not None:
            self.task_progress_bar.updateProgress(value, msg, verbose_msg)

    def setTaskName(self, task_name: str):
        if self.task_progress_bar is not None:
            self.task_progress_bar.description = task_name
    
    def zero_progress(self):
        if self.task_progress_bar is not None:
            self.task_progress_bar.updateProgress(0)
        if self.stop_button is not None:
            self.stop_button.setEnabled(True)
            self.stop_button.setText(self.tr('Stop'))

    def fit_to_content(self):
        # Stylesheets can change progress-bar height after construction.
        layout = self.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        self.adjustSize()

    def show_fitted(self):
        self.fit_to_content()
        self.show()
        self.raise_()
        self.activateWindow()

    def showEvent(self, e: QShowEvent) -> None:
        result = super().showEvent(e)
        self.showed.emit()
        return result


class ImgtransProgressMessageBox(ProgressMessageBox):
    stop_clicked = Signal()
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(None, False, *args, **kwargs)
        self.setModal(False)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )
        self.setWindowTitle(self.tr('Translation Progress'))
        self._user_moved = False
        
        self.upscale_bar = TaskProgressBar(self.tr('Upscaling: '), True, self)
        self.detect_bar = TaskProgressBar(self.tr('Detecting: '), True, self)
        self.ocr_bar = TaskProgressBar(self.tr('OCR: '), True, self)
        self.translate_bar = TaskProgressBar(self.tr('Translating: '), True, self)
        self.inpaint_bar = TaskProgressBar(self.tr('Inpainting: '), True, self)

        layout = self.layout()
        layout.addWidget(self.upscale_bar)
        layout.addWidget(self.detect_bar)
        layout.addWidget(self.ocr_bar)
        layout.addWidget(self.translate_bar)
        layout.addWidget(self.inpaint_bar)
        
        # Stop button
        self.stop_button = QPushButton(self.tr('Stop'), self)
        self.stop_button.clicked.connect(self.on_stop_clicked)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.setFixedWidth(540)

    def fit_to_content(self):
        layout = self.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        self.setFixedWidth(540)
        super().fit_to_content()
    
    def on_stop_clicked(self):
        self.stop_clicked.emit()
        self.stop_button.setEnabled(False)
        self.stop_button.setText(self.tr('trying to stop...'))

    def closeEvent(self, event: QCloseEvent) -> None:
        self.on_stop_clicked()
        event.accept()

    def moveEvent(self, event: QMoveEvent) -> None:
        if self.isVisible():
            self._user_moved = True
        super().moveEvent(event)

    def updateUpscaleProgress(self, value: int, msg: str = '', verbose_msg: str = None, error_count: int = 0, total_count: int = 0):
        self.upscale_bar.updateProgress(value, msg, verbose_msg, error_count=error_count, total_count=total_count)

    def updateDetectProgress(self, value: int, msg: str = '', verbose_msg: str = None, error_count: int = 0, total_count: int = 0):
        self.detect_bar.updateProgress(value, msg, verbose_msg, error_count=error_count, total_count=total_count)

    def updateOCRProgress(self, value: int, msg: str = '', verbose_msg: str = None, error_count: int = 0, total_count: int = 0):
        self.ocr_bar.updateProgress(value, msg, verbose_msg, error_count=error_count, total_count=total_count)

    def updateInpaintProgress(self, value: int, msg: str = '', verbose_msg: str = None, error_count: int = 0, total_count: int = 0):
        self.inpaint_bar.updateProgress(value, msg, verbose_msg, error_count=error_count, total_count=total_count)

    def updateTranslateProgress(self, value: int, msg: str = '', verbose_msg: str = None, error_count: int = 0, total_count: int = 0):
        self.translate_bar.updateProgress(value, msg, verbose_msg, error_count=error_count, total_count=total_count)
    
    def zero_progress(self):
        self.updateUpscaleProgress(0)
        self.updateDetectProgress(0)
        self.updateOCRProgress(0)
        self.updateTranslateProgress(0)
        self.updateInpaintProgress(0)
        self.stop_button.setEnabled(True)
        self.stop_button.setText(self.tr('Stop'))

    def show_all_bars(self):
        self.upscale_bar.show()
        self.detect_bar.show()
        self.ocr_bar.show()
        self.translate_bar.show()
        self.inpaint_bar.show()

    def hide_all_bars(self):
        self.upscale_bar.hide()
        self.detect_bar.hide()
        self.ocr_bar.hide()
        self.translate_bar.hide()
        self.inpaint_bar.hide()
