"""Action and shortcut registry controller for MainWindow."""
from functools import partial
from qtpy.QtWidgets import QShortcut
from qtpy.QtGui import QKeySequence
from ballontranslator.ui.core.misc import QKEY


class ActionRegistry:
    """Manages shortcut bindings, title bar triggers, and keyboard dispatch."""

    def __init__(self, main_window):
        self.win = main_window
        self.shortcuts = []

    def setup_shortcuts(self):
        win = self.win
        # titleBar triggers
        win.titleBar.nextpage_trigger.connect(win.shortcutNext)
        win.titleBar.prevpage_trigger.connect(win.shortcutBefore)
        win.titleBar.textedit_trigger.connect(win.shortcutTextedit)
        win.titleBar.drawboard_trigger.connect(win.shortcutDrawboard)
        win.titleBar.redo_trigger.connect(win.on_redo)
        win.titleBar.undo_trigger.connect(win.on_undo)
        win.titleBar.page_search_trigger.connect(win.on_page_search)
        win.titleBar.global_search_trigger.connect(win.on_global_search)
        win.titleBar.replacePreMTkeyword_trigger.connect(win.show_pre_MT_keyword_window)
        win.titleBar.replaceMTkeyword_trigger.connect(win.show_MT_keyword_window)
        win.titleBar.replaceOCRkeyword_trigger.connect(win.show_OCR_keyword_window)
        win.titleBar.show_module.connect(win.on_show_module)
        win.titleBar.importtstyle_trigger.connect(win.import_tstyles)
        win.titleBar.exporttstyle_trigger.connect(win.export_tstyles)
        win.titleBar.darkmode_trigger.connect(win.on_darkmode_triggered)
        win.titleBar.merge_tool_trigger.connect(win.on_open_merge_tool)
        win.titleBar.ps_bridge_trigger.connect(win.on_open_photoshop_bridge)
        win.titleBar.path_reorder_trigger.connect(win.on_path_reorder)
        win.canvas.path_reorder_mode_changed.connect(
            win.titleBar.path_reorder_action.setChecked
        )
        win.titleBar.font_exclusion_trigger.connect(
            win.configPanel.show_font_exclusion_dialog
        )

        # Keyboard shortcuts
        def _add_shortcut(key, slot):
            sc = QShortcut(QKeySequence(key), win)
            sc.activated.connect(slot)
            self.shortcuts.append(sc)
            return sc

        _add_shortcut("A", win.shortcutBefore)
        _add_shortcut(QKeySequence.StandardKey.MoveToPreviousPage, win.shortcutBefore)
        _add_shortcut("D", win.shortcutNext)
        _add_shortcut(QKeySequence.StandardKey.MoveToNextPage, win.shortcutNext)
        _add_shortcut("W", win.shortcutTextblock)
        _add_shortcut(QKeySequence.StandardKey.ZoomIn, win.canvas.gv.scale_up_signal)
        _add_shortcut(QKeySequence.StandardKey.ZoomOut, win.canvas.gv.scale_down_signal)
        _add_shortcut("Ctrl+D", win.shortcutCtrlD)
        _add_shortcut("Space", win.shortcutSpace)
        _add_shortcut(QKeySequence.StandardKey.SelectAll, win.shortcutSelectAll)
        _add_shortcut("Escape", win.shortcutEscape)
        _add_shortcut(QKeySequence.StandardKey.Bold, win.shortcutBold)
        _add_shortcut(QKeySequence.StandardKey.Italic, win.shortcutItalic)
        _add_shortcut(QKeySequence.StandardKey.Underline, win.shortcutUnderline)
        _add_shortcut("Ctrl+Q", win.st_manager.capitalize_selected_textitems)
        _add_shortcut(QKeySequence.StandardKey.Delete, win.shortcutDelete)

        drawpanel_shortcuts = {'hand': 'H', 'rect': 'R', 'inpaint': 'J', 'pen': 'B'}
        for tool_name, shortcut_key in drawpanel_shortcuts.items():
            sc = QShortcut(QKeySequence(shortcut_key), win)
            key = getattr(QKEY, f'Key_{shortcut_key}')
            sc.activated.connect(partial(
                win.drawingPanel.shortcutSetCurrentToolByName,
                tool_name,
                key,
            ))
            win.drawingPanel.setShortcutTip(tool_name, shortcut_key)
            self.shortcuts.append(sc)
