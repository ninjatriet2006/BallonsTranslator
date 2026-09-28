"""Page navigation and list controller for MainWindow."""
import os.path as osp
from typing import Optional
from qtpy.QtWidgets import QListWidgetItem, QWidget
from qtpy.QtGui import QIcon
from ballontranslator.utils import shared
from ballontranslator.utils.config import pcfg, save_config
from ballontranslator.ui.text_engine.editing.manager import SceneTextReplacementReason


class PageNavigationController:
    """Manages page list population, thumbnail management, and page switching."""

    def __init__(self, main_window):
        self.win = main_window

    def update_page_list(self):
        win = self.win
        if win.pageList.count() != 0:
            win.pageList.clear()
        if len(win.imgtrans_proj.pages) >= shared.PAGELIST_THUMBNAIL_MAXNUM:
            item_func = lambda imgname: QListWidgetItem(imgname)
        else:
            item_func = lambda imgname: QListWidgetItem(
                QIcon(osp.join(win.imgtrans_proj.directory, imgname)), imgname
            )
        for imgname in win.imgtrans_proj.pages:
            lstitem = item_func(imgname)
            win.pageList.addItem(lstitem)
            if imgname == win.imgtrans_proj.current_img:
                win.pageList.setCurrentItem(lstitem)
        win.llmContextEditor.set_project(win.imgtrans_proj)

    def handle_current_item_changed(self):
        win = self.win
        item = win.pageList.currentItem()
        win.page_changing = True
        if item is not None:
            if not win.opening_dir:
                win.st_manager.formatpanel.resolve_text_transform_edits_for_page_change()
            if win.save_on_page_changed:
                win.conditional_save()
            win.st_manager.clearSceneTextitems(
                SceneTextReplacementReason.PAGE_CHANGE
            )
            win.imgtrans_proj.set_current_img(item.text())
            win.llmContextEditor.set_page(item.text())
            win.canvas.clear_undostack(update_saved_step=True)
            win.canvas.updateCanvas()
            win.st_manager.populateSceneTextitems()
            win.titleBar.setTitleContent(page_name=win.imgtrans_proj.current_img)
            win.module_manager.handle_page_changed()
            win.drawingPanel.handle_page_changed()

        win.page_changing = False

    def handle_page_label_state_changed(self):
        win = self.win
        setup = win.leftBar.showPageListLabel.isChecked()
        if setup:
            if win.leftBar.globalSearchChecker.isChecked():
                win.leftBar.globalSearchChecker.setChecked(False)
            win._set_left_panel_content(win.pageList)
        else:
            win._set_left_panel_content(None)
        pcfg.show_page_list = setup
        save_config()
