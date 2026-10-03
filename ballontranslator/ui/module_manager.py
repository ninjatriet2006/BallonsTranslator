import time
import threading
import concurrent.futures
from typing import Callable, List, Optional, Union
import os.path as osp

import numpy as np
from packaging.requirements import Requirement
from qtpy.QtCore import QThread, Signal, QObject, QLocale, QTimer
from qtpy.QtWidgets import QFileDialog, QMessageBox, QCheckBox

from ballontranslator.ui.core.funcmaps import get_maskseg_method
from ballontranslator.ui.threads.packageinstall_thread import PackageInstallThread
from .package_manager import create_package_manager
from ballontranslator.ui.dialogs.torch_install_dialog import confirm_torch_install_device
from ballontranslator.utils.system.logger import logger as LOGGER
from ballontranslator.utils.project.registry import LazyModuleError, Registry
from ballontranslator.utils.system.imgproc_utils import enlarge_window, get_block_mask
from ballontranslator.utils.system.io_utils import text_is_empty
from ballontranslator.modules.translators import MissingTranslatorParams
from ballontranslator.modules.exceptions import (
    LLMApiKeyRequiredError,
    LLMBaseURLRequiredError,
    LLMModelRequiredError,
    LLMRequestStopped,
    LLMUserActionRequiredError,
    ModuleRunError,
)
from ballontranslator.modules.base import BaseModule, soft_empty_cache
from ballontranslator.modules import INPAINTERS, TRANSLATORS, TEXTDETECTORS, OCR, UPSCALERS, \
    GET_VALID_TRANSLATORS, GET_VALID_TEXTDETECTORS, GET_VALID_INPAINTERS, GET_VALID_OCR, GET_VALID_UPSCALERS, \
    BaseTranslator, InpainterBase, TextDetectorBase, OCRBase, UpscalerBase, merge_config_module_params
import ballontranslator.modules as modules
modules.translators.SYSTEM_LANG = QLocale.system().name()
from ballontranslator.utils.text.textblock import TextBlock, sort_regions
from ballontranslator.utils.project import shared
from ballontranslator.utils.project.message import create_error_dialog, create_info_dialog
from ballontranslator.utils.system.download_util import DownloadCancelled, sizeof_fmt
from ballontranslator.modules.prepare_local_files import MissingDependency, ensure_module_files
from ballontranslator.utils.env.py_package_manager import (
    MissingModuleRequirements,
    collect_missing_module_requirements,
)
from .custom_widget import ImgtransProgressMessageBox, ParamComboBox, ProgressMessageBox
from .configpanel import ConfigPanel
from ballontranslator.utils.project.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.project.config import pcfg, RunStatus, save_config, LLMTranslateContext
from ballontranslator.utils.project.llm_profiles import LLM_DETECTOR_KEY, LLM_INPAINT_KEY, LLM_OCR_KEY
from ballontranslator.utils.project.global_callbacks import register_global_callback
from ballontranslator.utils.project.stage_auditor import StageAuditor, StageIssue, is_safety_refusal_error
cfg_module = pcfg.module


from ballontranslator.ui.threads.pipeline_threads import (
    ModuleThread,
    TextDetectThread,
    OCRThread,
    InpaintThread,
    TranslateThread,
    ImgtransThread,
    unload_modules,
    _create_page_error_dialog,
    _show_llm_key_required_dialog,
    _show_llm_model_required_dialog,
    _show_llm_base_url_required_dialog,
    _show_llm_user_action_required_dialog,
    _reset_llm_key_required_dialogs,
    _mark_translation_finished,
    _llm_key_dialog_lock,
    _shown_llm_key_dialog_profiles,
    _llm_model_dialog_lock,
    _shown_llm_model_dialog_profiles,
    _llm_base_url_dialog_lock,
    _shown_llm_base_url_dialog_profiles,
)

class ModuleManager(QObject):
    imgtrans_proj: ProjImgTrans = None

    canvas_inpaint_finished = Signal(dict)
    canvas_inpaint_failed = Signal()
    inpaint_th_finished = Signal()

    imgtrans_pipeline_finished = Signal()
    blktrans_pipeline_finished = Signal(int, list)
    page_trans_finished = Signal(int)
    page_stage_finished = Signal(str, int)
    module_selection_changed = Signal(str, str)
    pipeline_issues_updated = Signal(int)

    run_canvas_inpaint = False
    is_waiting_th = False
    block_set_inpainter = False

    def __init__(self, 
                 imgtrans_proj: ProjImgTrans,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imgtrans_proj = imgtrans_proj
        self._pipeline_issues: List[StageIssue] = []
        self._pending_canvas_inpaint: Optional[dict] = None
        self._canvas_inpaint_source: Optional[np.ndarray] = None

        self.check_inpaint_fin_timer = QTimer(self)
        self.check_inpaint_fin_timer.timeout.connect(self.check_inpaint_th_finished)
        self.prepare_msgbox: ProgressMessageBox = None
        self._preparing_thread: ModuleThread = None
        self._pending_prepare_queue = []
        self._pending_prepare_success = None
        self._pending_prepare_failure = None
        self._pending_batch_package_queue = []
        self._pending_batch_package_success = None
        self._pending_batch_package_failure = None
        self._package_install_retried = set()
        self.package_install_thread: PackageInstallThread = None
        self.config_panel: ConfigPanel = None
        self.parent_widget = None

    @property
    def pipeline_issues(self) -> List[StageIssue]:
        return list(self._pipeline_issues)

    def clear_pipeline_issues(self):
        self._pipeline_issues.clear()
        self.pipeline_issues_updated.emit(0)

    def setupThread(
        self,
        config_panel: ConfigPanel,
        imgtrans_progress_msgbox: ImgtransProgressMessageBox,
        parent_widget=None,
    ):
        self.config_panel = config_panel
        self.parent_widget = parent_widget
        self.textdetect_thread = TextDetectThread()

        self.ocr_thread = OCRThread()
        
        self.translate_thread = TranslateThread()
        self.translate_thread.progress_changed.connect(self.on_update_translate_progress)

        self.inpaint_thread = InpaintThread()
        self.inpaint_thread.finish_inpaint.connect(self.on_finish_inpaint)
        self.inpaint_thread.inpaint_failed.connect(self.on_canvas_inpaint_failed)
        self.inpaint_thread.finished.connect(self._continue_canvas_inpaint)
        self._upscaler = None

        self.prepare_msgbox = ProgressMessageBox(self.tr('Preparing module: '), True, parent_widget)
        self.prepare_msgbox.stop_clicked.connect(self.cancelModulePreparation)

        for module_thread in [self.textdetect_thread, self.ocr_thread, self.translate_thread, self.inpaint_thread]:
            module_thread.module_prepare_progress.connect(self.on_module_prepare_progress)
            module_thread.finish_set_module.connect(lambda th=module_thread: self.on_module_prepare_finished(th))

        self.package_install_thread = PackageInstallThread()
        self.package_install_thread.package_prepare_progress.connect(self.on_module_prepare_progress)
        self.package_install_thread.finish_install.connect(self.on_batch_package_install_finished)

        self.progress_msgbox = imgtrans_progress_msgbox
        self.progress_msgbox.stop_clicked.connect(self.stopImgtransPipeline)

        self.imgtrans_thread = ImgtransThread(self.textdetect_thread, self.ocr_thread, self.translate_thread, self.inpaint_thread)
        self.imgtrans_thread.imgtrans_proj = self.imgtrans_proj
        self.imgtrans_thread.update_upscale_progress.connect(self.on_update_upscale_progress)
        self.imgtrans_thread.update_detect_progress.connect(self.on_update_detect_progress)
        self.imgtrans_thread.update_ocr_progress.connect(self.on_update_ocr_progress)
        self.imgtrans_thread.update_translate_progress.connect(self.on_update_translate_progress)
        self.imgtrans_thread.update_inpaint_progress.connect(self.on_update_inpaint_progress)
        self.imgtrans_thread.page_stage_finished.connect(self.page_stage_finished)
        self.imgtrans_thread.finish_blktrans_stage.connect(self.on_finish_blktrans_stage)
        self.imgtrans_thread.finish_blktrans.connect(self.on_finish_blktrans)
        self.imgtrans_thread.pipeline_stopped.connect(self.on_imgtrans_thread_stopped)
        self.imgtrans_thread.finished.connect(self.on_imgtrans_thread_finished)

        merge_config_module_params(
            cfg_module.translator_params, GET_VALID_TRANSLATORS(), TRANSLATORS.get)
        merge_config_module_params(
            cfg_module.inpainter_params, GET_VALID_INPAINTERS(), INPAINTERS.get)
        merge_config_module_params(
            cfg_module.textdetector_params, GET_VALID_TEXTDETECTORS(), TEXTDETECTORS.get)
        merge_config_module_params(
            cfg_module.ocr_params, GET_VALID_OCR(), OCR.get)
        merge_config_module_params(
            cfg_module.upscaler_params, GET_VALID_UPSCALERS(), UPSCALERS.get)

        config_panel.unload_models.connect(self.unload_all_models)
        config_panel.prepare_selected_modules.connect(self.prepareSelectedModules)
        config_panel.reinstall_torch.connect(self.reinstallTorch)


    def _thread_for_module_key(self, module_key: str) -> ModuleThread:
        return {
            'textdetector': self.textdetect_thread,
            'ocr': self.ocr_thread,
            'translator': self.translate_thread,
            'inpainter': self.inpaint_thread,
        }[module_key]

    def _setter_for_module_key(self, module_key: str):
        return {
            'textdetector': self.setTextDetector,
            'ocr': self.setOCR,
            'translator': self.setTranslator,
            'inpainter': self.setInpainter,
        }[module_key]

    def _registry_for_module_key(self, module_key: str) -> Registry:
        return {
            'textdetector': TEXTDETECTORS,
            'ocr': OCR,
            'translator': TRANSLATORS,
            'inpainter': INPAINTERS,
        }[module_key]

    def _module_ready(self, module_key: str, module_name: str) -> bool:
        thread = self._thread_for_module_key(module_key)
        module = thread.module
        return module is not None and module.name == module_name and module.all_model_loaded()

    def translator_metadata(self, translator: str = None):
        if translator is None:
            translator = cfg_module.translator
        if self.translator is not None and self.translator.name == translator:
            src_list = self.translator.supported_src_list
            tgt_list = self.translator.supported_tgt_list
            source = self.translator.lang_source
            target = self.translator.lang_target
        else:
            spec = TRANSLATORS.get_spec(translator)
            default_langs = list(modules.translators.LANGMAP_GLOBAL.keys())
            src_list = spec.supported_src_list if spec is not None and spec.supported_src_list else default_langs
            tgt_list = spec.supported_tgt_list if spec is not None and spec.supported_tgt_list else default_langs
            if not src_list:
                src_list = default_langs
            if not tgt_list:
                tgt_list = default_langs
            source = cfg_module.translate_source if cfg_module.translate_source in src_list else src_list[0]
            target = cfg_module.translate_target if cfg_module.translate_target in tgt_list else tgt_list[0]
        return {
            'name': translator,
            'supported_src_list': src_list,
            'supported_tgt_list': tgt_list,
            'lang_source': source,
            'lang_target': target,
        }

    def _prepare_modules_then(self, required_modules: List[tuple], on_success: Callable, on_failure: Callable = None):
        # RUN may need several stages; prepare them serially to keep one modal dialog.
        queue = [(module_key, module_name) for module_key, module_name in required_modules if not self._module_ready(module_key, module_name)]
        if not queue:
            on_success()
            return
        missing_modules = self._missing_module_requirements_for_modules(queue)
        if missing_modules:
            self._handle_batch_missing_packages(queue, missing_modules, on_success, on_failure)
            return
        self._begin_prepare_queue(queue, on_success, on_failure)

    def _begin_prepare_queue(self, queue: List[tuple], on_success: Callable, on_failure: Callable = None):
        self._package_install_retried.clear()
        self._pending_prepare_queue = queue
        self._pending_prepare_success = on_success
        self._pending_prepare_failure = on_failure
        self._continue_pending_prepare()

    def _missing_module_requirements_for_modules(self, required_modules: List[tuple]) -> List[MissingModuleRequirements]:
        module_specs = []
        for module_key, module_name in required_modules:
            spec = self._registry_for_module_key(module_key).get_spec(module_name)
            if spec is None:
                continue
            module_specs.append((module_key, module_name, spec))
        return collect_missing_module_requirements(module_specs, create_package_manager())

    def _continue_pending_prepare(self):
        while self._pending_prepare_queue:
            module_key, module_name = self._pending_prepare_queue.pop(0)
            if self._module_ready(module_key, module_name):
                continue
            self._setter_for_module_key(module_key)(module_name)
            return

        on_success = self._pending_prepare_success
        self._pending_prepare_success = None
        self._pending_prepare_failure = None
        self._package_install_retried.clear()
        if on_success is not None:
            on_success()

    @staticmethod
    def _combined_missing_requirements(missing_modules: List[MissingModuleRequirements]) -> List[str]:
        """Return a de-duplicated install list for a batch of modules.

        >>> items = [
        ...     MissingModuleRequirements('ocr', 'mit48px', ['torch', 'einops']),
        ...     MissingModuleRequirements('textdetector', 'ctd', ['torch', 'torchvision']),
        ... ]
        >>> ModuleManager._combined_missing_requirements(items)
        ['torch', 'einops', 'torchvision']
        """

        return list(dict.fromkeys(
            requirement
            for missing_module in missing_modules
            for requirement in missing_module.requirements
        ))

    @staticmethod
    def _missing_modules_text(missing_modules: List[MissingModuleRequirements]) -> str:
        """Format missing packages for the batch install dialog.

        >>> item = MissingModuleRequirements('ocr', 'mit48px', ['torch', 'einops'])
        >>> ModuleManager._missing_modules_text([item])
        'ocr/mit48px: torch, einops'
        """

        return '\n'.join(
            f'{item.module_key}/{item.module_name}: {", ".join(item.requirements)}'
            for item in missing_modules
        )

    def _install_and_auto_install_checked(self, msgbox: QMessageBox, install_btn, auto_install_checker: QCheckBox) -> bool:
        should_install = msgbox.clickedButton() is install_btn
        if should_install and auto_install_checker.isChecked():
            pcfg.package_manager.auto_install_missing_packages = True
            if self.config_panel is not None and not self.config_panel.package_auto_install_checker.isChecked():
                self.config_panel.package_auto_install_checker.setChecked(True)
            save_config()
        return should_install

    def _handle_batch_missing_packages(
        self,
        queue: List[tuple],
        missing_modules: List[MissingModuleRequirements],
        on_success: Callable,
        on_failure: Callable = None,
    ):
        requirements = self._combined_missing_requirements(missing_modules)
        missing_text = self._missing_modules_text(missing_modules)

        if pcfg.package_manager.auto_install_missing_packages:
            self._install_batch_missing_packages(queue, requirements, on_success, on_failure)
            return

        if shared.HEADLESS:
            LOGGER.error(f'Selected modules require package(s):\n{missing_text}')
            if on_failure is not None:
                on_failure()
            return

        if self._ask_install_missing_package_batch(missing_modules, requirements):
            self._install_batch_missing_packages(queue, requirements, on_success, on_failure)
            return

        if on_failure is not None:
            on_failure()

    def _ask_install_missing_package_batch(
        self,
        missing_modules: List[MissingModuleRequirements],
        requirements: List[str],
    ) -> bool:
        command_preview = create_package_manager().preview_command(requirements)
        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Icon.Question if hasattr(QMessageBox, 'Icon') else QMessageBox.Question)
        msgbox.setWindowTitle(self.tr('Missing packages'))
        msgbox.setText(
            self.tr('Selected modules require missing package(s):\n{modules}').format(
                modules=self._missing_modules_text(missing_modules),
            )
        )
        msgbox.setInformativeText(self.tr('Install the missing package(s) now?'))
        msgbox.setDetailedText(command_preview)
        install_btn = msgbox.addButton(self.tr('Install'), QMessageBox.ButtonRole.AcceptRole if hasattr(QMessageBox, 'ButtonRole') else QMessageBox.AcceptRole)
        msgbox.addButton(self.tr('Not Now'), QMessageBox.ButtonRole.RejectRole if hasattr(QMessageBox, 'ButtonRole') else QMessageBox.RejectRole)
        auto_install_checker = QCheckBox(self.tr("Install all missing packages and don't show again"))
        msgbox.setCheckBox(auto_install_checker)
        msgbox.exec()
        return self._install_and_auto_install_checked(msgbox, install_btn, auto_install_checker)

    def _install_batch_missing_packages(
        self,
        queue: List[tuple],
        requirements: List[str],
        on_success: Callable,
        on_failure: Callable = None,
    ):
        if self.package_install_thread is None:
            raise RuntimeError('Package install thread is not initialized.')
        if self.package_install_thread.isRunning():
            LOGGER.warning('Package installation is already running.')
            if on_failure is not None:
                on_failure()
            return
        accepted, torch_device, torch_cuda_version = self._confirm_torch_install_device(requirements)
        if not accepted:
            if on_failure is not None:
                on_failure()
            return
        self._pending_batch_package_queue = queue
        self._pending_batch_package_success = on_success
        self._pending_batch_package_failure = on_failure
        self._show_package_install_dialog(requirements)
        self.package_install_thread.installPackages(
            requirements,
            torch_device=torch_device,
            torch_cuda_version=torch_cuda_version,
        )

    def _confirm_torch_install_device(self, requirements: List[str]):
        if shared.HEADLESS:
            return True, None, None
        return confirm_torch_install_device(requirements, parent=self.parent_widget)

    def reinstallTorch(self) -> None:
        if self.package_install_thread is None or self.package_install_thread.isRunning():
            LOGGER.warning('Package installation is already running.')
            return
        accepted, torch_device, torch_cuda_version = confirm_torch_install_device(
            ['torch'], parent=self.config_panel,
        )
        if not accepted:
            return
        self.config_panel.setTorchInstalling()
        self._show_package_install_dialog(['torch'])
        if not self.package_install_thread.installPackages(
            ['torch'],
            torch_device=torch_device,
            torch_cuda_version=torch_cuda_version,
        ):
            self.config_panel.refreshTorchStatus()

    def _show_package_install_dialog(self, requirements: List[str]):
        if shared.HEADLESS:
            LOGGER.info(f'Installing package(s): {", ".join(requirements)}')
            return
        self.prepare_msgbox.zero_progress()
        self.prepare_msgbox.setTaskName(self.tr('Installing packages: '))
        self.prepare_msgbox.updateTaskProgress(0, self._installing_packages_summary(requirements))
        if self.prepare_msgbox.stop_button is not None:
            self.prepare_msgbox.stop_button.setEnabled(False)
            self.prepare_msgbox.stop_button.setText(self.tr('Installing...'))
        self.prepare_msgbox.setFixedWidth(self.progress_msgbox.sizeHint().width())
        self.prepare_msgbox.show_fitted()

    def on_batch_package_install_finished(self):
        if not shared.HEADLESS:
            self.prepare_msgbox.done(0)
        if self.config_panel is not None:
            self.config_panel.refreshTorchStatus()

        queue = self._pending_batch_package_queue
        on_success = self._pending_batch_package_success
        on_failure = self._pending_batch_package_failure
        self._pending_batch_package_queue = []
        self._pending_batch_package_success = None
        self._pending_batch_package_failure = None

        if self.package_install_thread.last_success:
            if shared.HEADLESS:
                self._begin_prepare_queue(queue, on_success, on_failure)
            else:
                QTimer.singleShot(0, lambda: self._begin_prepare_queue(queue, on_success, on_failure))
            return

        if self.package_install_thread.last_error is not None:
            create_error_dialog(
                self.package_install_thread.last_error,
                self.tr('Failed to install packages'),
                'InstallPackages:batch',
            )
        if on_failure is not None:
            on_failure()

    def _show_prepare_dialog(self, thread: ModuleThread, module_name: str):
        self._preparing_thread = thread
        if shared.HEADLESS:
            LOGGER.info(f'Preparing {thread.module_key} module: {module_name}')
            return
        # Match the RUN progress widget geometry/style for first-run preparation.
        self.prepare_msgbox.zero_progress()
        self.prepare_msgbox.setTaskName(self.tr('Preparing module: '))
        self.prepare_msgbox.updateTaskProgress(0, module_name)
        self.prepare_msgbox.setFixedWidth(self.progress_msgbox.sizeHint().width())
        self.prepare_msgbox.show_fitted()

    def cancelModulePreparation(self):
        if self._preparing_thread is not None:
            self._preparing_thread.requestCancelModuleInit()

    @staticmethod
    def _installing_packages_summary(requirements: List[str]) -> str:
        """Return a compact install summary for the modal progress panel.

        >>> ModuleManager._installing_packages_summary(['torch', 'torchvision'])
        'torch...'
        >>> ModuleManager._installing_packages_summary(['einops'])
        'einops'
        """

        reqs = list(dict.fromkeys(requirements))
        if not reqs:
            return 'packages'
        try:
            first = Requirement(reqs[0]).name
        except Exception:
            first = str(reqs[0])
        return first + ('...' if len(reqs) > 1 else '')

    def _package_download_progress_message(self, payload: dict):
        """Format installer download progress for the preparation dialog.

        >>> manager = ModuleManager.__new__(ModuleManager)
        >>> manager._package_download_progress_message({
        ...     'message': 'Downloading torch.whl',
        ...     'downloaded': 50,
        ...     'total': 100,
        ...     'speed': 25,
        ...     'eta': 2,
        ... })
        ('Downloading torch.whl', '50.0% | 25.0 B/s | ETA 0:00:02')
        """

        message = payload.get('message') or self.tr('Downloading package')
        downloaded = payload.get('downloaded') or 0
        total = payload.get('total')
        status = []
        if total:
            percent = downloaded / total * 100
            status.append(f'{percent:.1f}%')

        speed = payload.get('speed')
        if speed:
            status.append(self.tr('{speed}/s').format(speed=sizeof_fmt(speed)))
        eta = payload.get('eta')
        if eta is not None:
            status.append(self.tr('ETA {eta}').format(eta=self._format_duration(eta)))
        return message, ' | '.join(status)

    @staticmethod
    def _format_duration(seconds: int) -> str:
        seconds = max(int(seconds), 0)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f'{hours}:{minutes:02d}:{seconds:02d}'

    def on_module_prepare_progress(self, payload: dict):
        # Download callbacks report bytes; imports/model-loads report coarse stages.
        event = payload.get('event', '')
        module_name = payload.get('module', '')
        message = payload.get('message', '')
        verbose_message = None
        path = payload.get('path') or payload.get('file') or ''
        if path:
            path = osp.basename(path)

        progress = {
            'checking_dependencies': 5,
            'module_file_group': 10,
            'file_check': 15,
            'file_start': 20,
            'archive_extract': 75,
            'archive_move': 85,
            'installing_packages': 25,
            'package_output': 25,
            'package_download_progress': 30,
            'importing': 88,
            'instantiating': 92,
            'loading_model': 96,
            'file_done': 100,
        }.get(event, 0)
        if event == 'file_progress':
            total = payload.get('total')
            downloaded = payload.get('downloaded') or 0
            if total:
                progress = max(20, min(74, int(downloaded / total * 54) + 20))
            else:
                progress = 35
        elif event == 'package_download_progress':
            message, verbose_message = self._package_download_progress_message(payload)
            total = payload.get('total')
            downloaded = payload.get('downloaded') or 0
            if total:
                progress = max(0, min(100, int(downloaded / total * 100)))
        elif event in {'installing_packages', 'package_output'}:
            verbose_message = ''
        if not message:
            message = path or module_name

        if shared.HEADLESS:
            if event in {'checking_dependencies', 'file_start', 'archive_extract', 'installing_packages', 'package_output', 'importing', 'loading_model'}:
                LOGGER.info(f'{message}: {event}')
            return
        if self.prepare_msgbox is not None:
            self.prepare_msgbox.updateTaskProgress(progress, message, verbose_message)

    def on_module_prepare_finished(self, thread: ModuleThread):
        if self._preparing_thread is thread:
            if not shared.HEADLESS:
                self.prepare_msgbox.done(0)
            self._preparing_thread = None

        if not thread.last_set_success and thread.last_missing_requirements:
            if self._handle_missing_packages(thread):
                return

        if self._pending_prepare_success is None and self._pending_prepare_failure is None:
            if not thread.last_set_success and thread.last_error is not None:
                self._show_module_prepare_failure(thread)
            return

        if thread.last_set_success:
            if shared.HEADLESS:
                self._continue_pending_prepare()
            else:
                # Let Qt finish closing the prepare dialog before showing RUN progress.
                QTimer.singleShot(0, self._continue_pending_prepare)
        else:
            self._show_module_prepare_failure(thread)
            self._finish_pending_prepare_failure()

    def _finish_pending_prepare_failure(self):
        on_failure = self._pending_prepare_failure
        self._pending_prepare_queue = []
        self._pending_prepare_success = None
        self._pending_prepare_failure = None
        self._package_install_retried.clear()
        if on_failure is not None:
            on_failure()

    def _clear_install_retry_if_standalone_prepare(self):
        if self._pending_prepare_success is None and self._pending_prepare_failure is None:
            self._package_install_retried.clear()

    def _handle_missing_packages(self, thread: ModuleThread) -> bool:
        requirements = list(dict.fromkeys(thread.last_missing_requirements))
        retry_key = (thread.module_key, thread.last_set_module_name, tuple(requirements))
        if retry_key in self._package_install_retried:
            self._show_module_prepare_failure(thread)
            if self._pending_prepare_success is not None or self._pending_prepare_failure is not None:
                self._finish_pending_prepare_failure()
            return True

        if pcfg.package_manager.auto_install_missing_packages:
            self._install_missing_packages_and_retry(thread, requirements, retry_key)
            return True

        if shared.HEADLESS:
            LOGGER.error(
                f'Module {thread.last_set_module_name} requires package(s): '
                f'{", ".join(requirements)}'
            )
            return False

        should_install = self._ask_install_missing_packages(thread, requirements)
        if should_install:
            self._install_missing_packages_and_retry(thread, requirements, retry_key)
            return True
        return False

    def _ask_install_missing_packages(self, thread: ModuleThread, requirements: List[str]) -> bool:
        command_preview = create_package_manager().preview_command(requirements)
        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Icon.Question if hasattr(QMessageBox, 'Icon') else QMessageBox.Question)
        msgbox.setWindowTitle(self.tr('Missing packages'))
        msgbox.setText(
            self.tr('Module "{module}" requires missing package(s):\n{packages}').format(
                module=thread.last_set_module_name,
                packages=', '.join(requirements),
            )
        )
        msgbox.setInformativeText(self.tr('Install the missing package(s) now?'))
        msgbox.setDetailedText(command_preview)
        install_btn = msgbox.addButton(self.tr('Install'), QMessageBox.ButtonRole.AcceptRole if hasattr(QMessageBox, 'ButtonRole') else QMessageBox.AcceptRole)
        msgbox.addButton(self.tr('Not Now'), QMessageBox.ButtonRole.RejectRole if hasattr(QMessageBox, 'ButtonRole') else QMessageBox.RejectRole)
        auto_install_checker = QCheckBox(self.tr("Install all missing packages and don't show again"))
        msgbox.setCheckBox(auto_install_checker)
        msgbox.exec()
        return self._install_and_auto_install_checked(msgbox, install_btn, auto_install_checker)

    def _install_missing_packages_and_retry(self, thread: ModuleThread, requirements: List[str], retry_key: tuple):
        accepted, torch_device, torch_cuda_version = self._confirm_torch_install_device(requirements)
        if not accepted:
            return
        self._package_install_retried.add(retry_key)
        self._show_prepare_dialog(thread, thread.last_set_module_name)
        thread.installMissingPackagesAndSetModule(
            thread.last_set_module_name,
            requirements,
            torch_device=torch_device,
            torch_cuda_version=torch_cuda_version,
        )

    def _show_module_prepare_failure(self, thread: ModuleThread):
        if (
            thread.last_error is None
            or isinstance(thread.last_error, DownloadCancelled)
            or (hasattr(thread, 'cancel_event') and thread.cancel_event.is_set())
        ):
            return
        msg = self.tr('Failed to set module ') + thread.last_set_module_name
        create_error_dialog(thread.last_error, msg, f'FailedSetModule:{thread.module_key}:{thread.last_set_module_name}')

    def unload_all_models(self):
        unload_modules(self, ('textdetector', 'inpainter', 'ocr', 'translator'))

    @property
    def translator(self) -> BaseTranslator:
        return self.translate_thread.translator

    @property
    def inpainter(self) -> InpainterBase:
        return self.inpaint_thread.inpainter

    @property
    def textdetector(self) -> TextDetectorBase:
        return self.textdetect_thread.textdetector

    @property
    def ocr(self) -> OCRBase:
        return self.ocr_thread.ocr

    @property
    def upscaler(self) -> UpscalerBase:
        return getattr(self, '_upscaler', None)

    def inpainterBusy(self):
        return self.inpaint_thread.isRunning()

    def inpaint(self, img: np.ndarray, mask: np.ndarray, img_key: str = None, inpaint_rect = None, **kwargs):
        if self.inpaint_thread.isRunning():
            LOGGER.warning('Waiting for inpainting to finish')
            return
        self.inpaint_thread.inpaint(img, mask, img_key, inpaint_rect)

    def terminateRunningThread(self):
        if self.textdetect_thread.isRunning():
            self.textdetect_thread.quit()
        if self.ocr_thread.isRunning():
            self.ocr_thread.quit()
        if self.inpaint_thread.isRunning():
            self.inpaint_thread.quit()
        if self.translate_thread.isRunning():
            self.translate_thread.quit()

    def _module_preparation_running(self) -> bool:
        module_threads = [self.textdetect_thread, self.ocr_thread, self.translate_thread, self.inpaint_thread]
        package_running = self.package_install_thread is not None and self.package_install_thread.isRunning()
        return package_running or any(thread.isRunning() for thread in module_threads)

    def _selected_modules(self) -> List[tuple]:
        return [
            ('textdetector', cfg_module.textdetector),
            ('ocr', cfg_module.ocr),
            ('translator', cfg_module.translator),
            ('inpainter', cfg_module.inpainter),
        ]

    def prepareSelectedModules(self):
        if self.imgtrans_thread.isRunning() or self._module_preparation_running():
            create_info_dialog(self.tr('Module preparation is already running.'))
            return
        self._prepare_modules_then(
            self._selected_modules(),
            lambda: create_info_dialog(self.tr('Selected modules are ready.')),
        )

    def check_inpaint_th_finished(self):
        if self.inpaint_thread.isRunning():
            return
        self.block_set_inpainter = False
        self.check_inpaint_fin_timer.stop()
        self.inpaint_th_finished.emit()

    def runImgtransPipeline(
        self,
        pages_to_process=None,
        render_only=False,
    ):
        _reset_llm_key_required_dialogs()
        if self.imgtrans_proj.is_empty:
            LOGGER.info('proj file is empty, nothing to do')
            self.progress_msgbox.hide()
            return
        self.last_finished_index = -1
        self.terminateRunningThread()
        
        if (render_only or cfg_module.all_stages_disabled()) and \
                self.imgtrans_proj is not None and self.imgtrans_proj.num_pages > 0:
            # Rendering reuses the normal page-finished path without preparing
            # or running any pipeline module.
            page_indexes = range(self.imgtrans_proj.num_pages)
            if not render_only and pages_to_process is not None:
                page_index_by_name = {
                    page_name: index
                    for index, page_name in enumerate(self.imgtrans_proj.pages)
                }
                page_indexes = (
                    page_index_by_name[page_name]
                    for page_name in pages_to_process
                    if page_name in page_index_by_name
                )
            for ii in page_indexes:
                self.page_trans_finished.emit(ii)
            self.imgtrans_pipeline_finished.emit()
            return

        required_modules = []
        if cfg_module.enable_detect:
            required_modules.append(('textdetector', cfg_module.textdetector))
        if cfg_module.enable_ocr:
            required_modules.append(('ocr', cfg_module.ocr))
        if cfg_module.enable_translate:
            required_modules.append(('translator', cfg_module.translator))
        if cfg_module.enable_inpaint:
            required_modules.append(('inpainter', cfg_module.inpainter))
        self._prepare_modules_then(
            required_modules,
            lambda: self._startImgtransPipeline(
                pages_to_process,
            ),
            on_failure=lambda: self.imgtrans_pipeline_finished.emit(),
        )

    def _startImgtransPipeline(
        self,
        pages_to_process=None,
    ):
        if self.prepare_msgbox is not None and self.prepare_msgbox.isVisible():
            self.prepare_msgbox.done(0)
        has_upscale = bool(cfg_module.enable_upscale and cfg_module.upscaler != 'None')
        self.progress_msgbox.upscale_bar.setVisible(has_upscale)
        self.progress_msgbox.detect_bar.setVisible(cfg_module.enable_detect)
        self.progress_msgbox.ocr_bar.setVisible(cfg_module.enable_ocr)
        self.progress_msgbox.translate_bar.setVisible(cfg_module.enable_translate)
        self.progress_msgbox.inpaint_bar.setVisible(cfg_module.enable_inpaint)
        self.progress_msgbox.zero_progress()
        self.progress_msgbox.show_fitted()
        self.imgtrans_thread.runImgtransPipeline(
            self.imgtrans_proj,
            pages_to_process,
        )
    
    def stopImgtransPipeline(self):
        """停止图像翻译流程"""
        start_time = getattr(self.imgtrans_thread, 'pipeline_start_time', None)
        elapsed_str = f" after {time.time() - start_time:.1f}s" if start_time else ""
        LOGGER.stop(f"Translation pipeline stopped by user{elapsed_str}.")
        self.imgtrans_thread.requestStop()
        if hasattr(self.translate_thread, 'pipeline_pagekey_queue'):
            self.translate_thread.pipeline_pagekey_queue.clear()
        self.progress_msgbox.hide()

    def runBlktransPipeline(
        self,
        blk_list: List[TextBlock],
        mode: int,
        blk_ids: List[int],
        *,
        page_key: str = None,
    ):
        _reset_llm_key_required_dialogs()
        self.terminateRunningThread()
        required_modules = []
        if mode >= 0 and mode < 3:
            required_modules.append(('ocr', cfg_module.ocr))
        if mode != 0 and mode < 3:
            required_modules.append(('translator', cfg_module.translator))
        if mode >= 2:
            required_modules.append(('inpainter', cfg_module.inpainter))
        self._prepare_modules_then(
            required_modules,
            lambda: self._startBlktransPipeline(
                blk_list,
                mode,
                blk_ids,
                page_key=page_key,
            ),
        )

    def _startBlktransPipeline(
        self,
        blk_list: List[TextBlock],
        mode: int,
        blk_ids: List[int],
        *,
        page_key: str = None,
    ):
        if self.prepare_msgbox is not None and self.prepare_msgbox.isVisible():
            self.prepare_msgbox.done(0)
        self.progress_msgbox.hide_all_bars()
        if mode >= 0 and mode < 3:
            self.progress_msgbox.ocr_bar.show()
        if mode >= 2:
            self.progress_msgbox.inpaint_bar.show()
        if mode != 0 and mode < 3:
            self.progress_msgbox.translate_bar.show()
        self.progress_msgbox.zero_progress()
        self.progress_msgbox.show_fitted()
        self.imgtrans_thread.runBlktransPipeline(
            blk_list,
            mode,
            blk_ids,
            page_key=page_key,
        )

    def on_finish_blktrans_stage(self, stage: str, progress: int):
        if stage == 'ocr':
            self.progress_msgbox.updateOCRProgress(progress)
        elif stage == 'translate':
            self.progress_msgbox.updateTranslateProgress(progress)
        elif stage == 'inpaint':
            self.progress_msgbox.updateInpaintProgress(progress)
        else:
            raise NotImplementedError(f'Unknown stage: {stage}')
        
    def on_finish_blktrans(self, mode: int, blk_ids: List):
        self.blktrans_pipeline_finished.emit(mode, blk_ids)
        self.progress_msgbox.hide()

    def _on_update_stage_progress(self, stage: str, progress: int, update_progress: Callable[..., None], error_count: int = 0):
        try:
            ri = self.imgtrans_thread.recent_finished_index(progress)
            if stage in shared.pbar:
                shared.pbar[stage].update(1)
            num_pages = self.imgtrans_thread.num_pages
            progress_pct = int(progress / num_pages * 100) if num_pages > 0 else 0
            try:
                update_progress(progress_pct, error_count=error_count, total_count=num_pages)
            except TypeError:
                update_progress(progress_pct)
            if ri != self.last_finished_index:
                self.last_finished_index = ri
                self.page_trans_finished.emit(ri)
            if progress_pct == 100:
                self.finishImgtransPipeline()
        except Exception as e:
            LOGGER.error(f"Error in _on_update_stage_progress({stage}, {progress}): {e}")

    def on_update_upscale_progress(self, progress: int, error_count: int = 0):
        self._on_update_stage_progress('upscale', progress, self.progress_msgbox.updateUpscaleProgress, error_count=error_count)

    def on_update_detect_progress(self, progress: int, error_count: int = 0):
        self._on_update_stage_progress('detect', progress, self.progress_msgbox.updateDetectProgress, error_count=error_count)

    def on_update_ocr_progress(self, progress: int, error_count: int = 0):
        self._on_update_stage_progress('ocr', progress, self.progress_msgbox.updateOCRProgress, error_count=error_count)

    def on_update_translate_progress(self, progress: int, error_count: int = 0):
        self._on_update_stage_progress('translate', progress, self.progress_msgbox.updateTranslateProgress, error_count=error_count)

    def on_update_inpaint_progress(self, progress: int, error_count: int = 0):
        self._on_update_stage_progress('inpaint', progress, self.progress_msgbox.updateInpaintProgress, error_count=error_count)

    def progress(self):
        progress = {}
        num_pages = self.imgtrans_thread.num_pages
        if cfg_module.enable_upscale and cfg_module.upscaler != 'None':
            progress['upscale'] = self.imgtrans_thread.upscale_counter / num_pages if num_pages > 0 else 0
        if cfg_module.enable_detect:
            progress['detect'] = self.imgtrans_thread.detect_counter / num_pages
        if cfg_module.enable_ocr:
            progress['ocr'] = self.imgtrans_thread.ocr_counter / num_pages
        if cfg_module.enable_inpaint:
            progress['inpaint'] = self.imgtrans_thread.inpaint_counter / num_pages
        if cfg_module.enable_translate:
            progress['translate'] = self.imgtrans_thread.translate_counter / num_pages
        return progress

    def proj_finished(self):
        if self.imgtrans_thread.upscale_finished() \
            and self.imgtrans_thread.detect_finished() \
            and self.imgtrans_thread.ocr_finished() \
            and self.imgtrans_thread.translate_finished() \
            and self.imgtrans_thread.inpaint_finished():
            return True
        return False

    def finishImgtransPipeline(self, force: bool = False):
        if force or self.proj_finished():
            try:
                self.progress_msgbox.hide()
                start_time = getattr(self.imgtrans_thread, 'pipeline_start_time', None)
                elapsed_str = f" in {time.time() - start_time:.1f}s" if start_time else ""
                stats = []
                if cfg_module.enable_detect:
                    stats.append(f"Detect: {self.imgtrans_thread.detect_counter}")
                if cfg_module.enable_ocr:
                    stats.append(f"OCR: {self.imgtrans_thread.ocr_counter}")
                if cfg_module.enable_translate:
                    trans_count = (
                        self.translate_thread.finished_counter
                        if getattr(self.imgtrans_thread, 'parallel_trans', False)
                        else self.imgtrans_thread.translate_counter
                    )
                    stats.append(f"Translate: {trans_count}")
                if cfg_module.enable_inpaint:
                    stats.append(f"Inpaint: {self.imgtrans_thread.inpaint_counter}")
                stats_str = f" | Stats: {', '.join(stats)}" if stats else ""
                LOGGER.stop(f"Translation pipeline completed: {self.imgtrans_thread.num_pages} page(s){elapsed_str}{stats_str}")
            except Exception as e:
                LOGGER.error(f"Error logging pipeline stats: {e}")

            self._audit_and_update_issues()

            try:
                self.imgtrans_pipeline_finished.emit()
            except Exception as e:
                LOGGER.error(f"Error emitting imgtrans_pipeline_finished: {e}")
    
    def on_imgtrans_thread_finished(self):
        """Ensure cleanup and issue audit when the pipeline thread finishes."""
        self.finishImgtransPipeline(force=True)

    def on_imgtrans_thread_stopped(self):
        """线程完成时确保关闭进度对话框"""
        try:
            self.progress_msgbox.hide()
        except Exception:
            pass
        self._audit_and_update_issues()
        try:
            self.imgtrans_pipeline_finished.emit()
        except Exception as e:
            LOGGER.error(f"Error emitting imgtrans_pipeline_finished on stop: {e}")

    def _audit_and_update_issues(self):
        try:
            enabled_stages = {
                'detect': cfg_module.enable_detect,
                'ocr': cfg_module.enable_ocr,
                'translate': cfg_module.enable_translate,
                'inpaint': cfg_module.enable_inpaint,
            }
            target_pages = getattr(self.imgtrans_thread, 'pages_to_process', None)
            audited = StageAuditor.audit_project_pages(
                self.imgtrans_proj,
                target_pages=target_pages,
                enabled_stages=enabled_stages,
            )
            runtime = getattr(self.imgtrans_thread, 'runtime_issues', [])
            all_issues = list(runtime)
            seen = {(issue.page_key, issue.stage) for issue in all_issues}
            for ai in audited:
                if (ai.page_key, ai.stage) not in seen:
                    all_issues.append(ai)
                    seen.add((ai.page_key, ai.stage))
            self._pipeline_issues = all_issues
            self.pipeline_issues_updated.emit(len(self._pipeline_issues))
        except Exception as e:
            LOGGER.error(f"Error auditing pipeline issues: {e}")

    def _select_module(self, module_key: str, module_name: str, valid_modules: List[str]):
        if module_name is None:
            module_name = getattr(cfg_module, module_key)
        if module_name not in valid_modules:
            LOGGER.warning(f'Ignoring invalid {module_key} module selection: {module_name}')
            return

        setattr(cfg_module, module_key, module_name)
        self.module_selection_changed.emit(module_key, module_name)

    def selectTranslator(self, translator: str = None):
        self._select_module('translator', translator, GET_VALID_TRANSLATORS())

    def selectInpainter(self, inpainter: str = None):
        self._select_module('inpainter', inpainter, GET_VALID_INPAINTERS())

    def selectTextDetector(self, textdetector: str = None):
        self._select_module('textdetector', textdetector, GET_VALID_TEXTDETECTORS())

    def selectOCR(self, ocr: str = None):
        self._select_module('ocr', ocr, GET_VALID_OCR())

    def selectUpscaler(self, upscaler: str = None):
        self._select_module('upscaler', upscaler, GET_VALID_UPSCALERS())

    def setTranslator(self, translator: str = None):
        if translator is None:
            translator = cfg_module.translator
        if self.translate_thread.isRunning():
            LOGGER.warning('Cancelling a running translation/module preparation thread.')
            self.translate_thread.requestCancelModuleInit()
            return
        self._clear_install_retry_if_standalone_prepare()
        self._show_prepare_dialog(self.translate_thread, translator)
        self.translate_thread.setTranslator(translator)

    def setInpainter(self, inpainter: str = None):
        
        if self.block_set_inpainter:
            return
        
        if inpainter is None:
            inpainter =cfg_module.inpainter
        
        if self.inpaint_thread.isRunning():
            self.inpaint_thread.requestCancelModuleInit()
            return

        self._clear_install_retry_if_standalone_prepare()
        self._show_prepare_dialog(self.inpaint_thread, inpainter)
        self.inpaint_thread.setInpainter(inpainter)

    def setTextDetector(self, textdetector: str = None):
        if textdetector is None:
            textdetector = cfg_module.textdetector
        if self.textdetect_thread.isRunning():
            LOGGER.warning('Cancelling a running text detection/module preparation thread.')
            self.textdetect_thread.requestCancelModuleInit()
            return
        self._clear_install_retry_if_standalone_prepare()
        self._show_prepare_dialog(self.textdetect_thread, textdetector)
        self.textdetect_thread.setTextDetector(textdetector)

    def setOCR(self, ocr: str = None):
        if ocr is None:
            ocr = cfg_module.ocr
        if self.ocr_thread.isRunning():
            LOGGER.warning('Cancelling a running OCR/module preparation thread.')
            self.ocr_thread.requestCancelModuleInit()
            return
        self._clear_install_retry_if_standalone_prepare()
        self._show_prepare_dialog(self.ocr_thread, ocr)
        self.ocr_thread.setOCR(ocr)

    def on_finish_inpaint(self, inpaint_dict: dict) -> None:
        apply_result = (
            self.run_canvas_inpaint
            and self._canvas_inpaint_source is self.imgtrans_proj.inpainted_array
        )
        self.run_canvas_inpaint = False
        self._canvas_inpaint_source = None
        if apply_result:
            self.canvas_inpaint_finished.emit(inpaint_dict)

    def canvas_inpaint(self, inpaint_dict: dict) -> None:
        """Keep a canvas request until module preparation and the worker finish.

        >>> manager.canvas_inpaint(request)  # doctest: +SKIP
        """
        # The page may change while this request waits for an older inference.
        self._pending_canvas_inpaint = {
            **inpaint_dict,
            'img': inpaint_dict['img'].copy(),
            'mask': inpaint_dict['mask'].copy(),
        }
        self._continue_canvas_inpaint()

    def _continue_canvas_inpaint(self) -> None:
        if (
            self._pending_canvas_inpaint is None
            or self.inpaint_thread.isRunning()
            or self._pending_prepare_success is not None
            or self._pending_batch_package_success is not None
        ):
            return
        self._prepare_modules_then(
            [('inpainter', cfg_module.inpainter)],
            self._start_canvas_inpaint,
            self._fail_pending_canvas_inpaint,
        )

    def _start_canvas_inpaint(self) -> None:
        # finish_set_module can be delivered before QThread.run() returns.
        # QThread.finished resumes the request if the worker is still busy.
        if self._pending_canvas_inpaint is None or self.inpaint_thread.isRunning():
            return
        inpaint_dict = self._pending_canvas_inpaint
        self._pending_canvas_inpaint = None
        self._canvas_inpaint_source = self.imgtrans_proj.inpainted_array
        self.run_canvas_inpaint = True
        self.inpaint(**inpaint_dict)

    def _fail_pending_canvas_inpaint(self) -> None:
        if self._pending_canvas_inpaint is not None:
            self._pending_canvas_inpaint = None
            self.canvas_inpaint_failed.emit()

    def on_canvas_inpaint_failed(self) -> None:
        notify = (
            self.run_canvas_inpaint
            and self._canvas_inpaint_source is self.imgtrans_proj.inpainted_array
        )
        self.run_canvas_inpaint = False
        self._canvas_inpaint_source = None
        if notify:
            self.canvas_inpaint_failed.emit()
    
    def moduleParams(self, module_key: str, module_name: str) -> dict:
        return cfg_module.get_params(module_key).get(module_name)

    def moduleRuntimeActionsEnabled(
        self,
        module_key: str,
        module_name: str,
    ) -> bool:
        active_module = self._active_module(module_key)
        return active_module is not None and active_module.name == module_name

    def _active_module(self, module_key: str):
        return {
            'translator': self.translator,
            'inpainter': self.inpainter,
            'textdetector': self.textdetector,
            'ocr': self.ocr,
            'upscaler': self.upscaler,
        }.get(module_key)

    def onModuleParamEdited(
        self,
        module_key: str,
        module_name: str,
        param_key: str,
        param_content: dict,
    ) -> None:
        active_module = self._active_module(module_key)
        if active_module is not None and active_module.name == module_name:
            self.updateModuleSetupParam(active_module, param_key, param_content)
            cfg_module.get_params(module_key)[module_name] = active_module.params
            return
        self.updateStoredModuleSetupParam(module_key, module_name, param_key, param_content)

    def updateStoredModuleSetupParam(self, module_key: str, module_name: str, param_key: str, param_content: dict):
        if param_content.get('flush', False):
            return

        module_params = cfg_module.get_params(module_key).get(module_name)
        if module_params is None or param_key not in module_params:
            return

        param_value = param_content['content']
        if param_content.get('select_path', False):
            dialog = QFileDialog()
            f = module_params[param_key].get('path_filter', None)
            param_value = dialog.getOpenFileUrl(self.parent(), filter=f)[0].toLocalFile()
            if not osp.exists(param_value):
                return
            param_widget: ParamComboBox = param_content['widget']
            param_widget.setCurrentText(param_value)

        self.updateParamDictValue(module_params, param_key, param_value)

    def updateParamDictValue(self, params: dict, param_key: str, param_value, convert_dtype=True):
        p = params[param_key]
        if isinstance(p, dict):
            if convert_dtype:
                try:
                    val_type = p.get('data_type', type(p['value']))
                    param_value = val_type(param_value)
                except ValueError:
                    dtype = type(p['value'])
                    LOGGER.warning(f'Invalid param value {param_value} for defined dtype: {dtype}')
                    return
            p['value'] = param_value
        else:
            if convert_dtype:
                try:
                    param_value = type(p)(param_value)
                except ValueError:
                    LOGGER.warning(f'Invalid param value {param_value} for defined dtype: {type(p)}, revert to original value {p}')
                    return
            params[param_key] = param_value

    def updateModuleSetupParam(self, 
                               module: Union[InpainterBase, BaseTranslator],
                               param_key: str, param_content: dict):
            
        if param_content.get('flush', False):
            param_widget: ParamComboBox = param_content['widget']
            param_widget.blockSignals(True)
            current_item = param_widget.currentText()
            param_widget.clear()
            param_widget.addItems(module.flush(param_key))
            param_widget.setCurrentText(current_item)
            param_widget.blockSignals(False)
        elif param_content.get('select_path', False):
            dialog = QFileDialog()
            f = module.params[param_key].get('path_filter', None)
            p = dialog.getOpenFileUrl(self.parent(), filter=f)[0].toLocalFile()
            if osp.exists(p):
                param_widget: ParamComboBox = param_content['widget']
                param_widget.setCurrentText(p)
        else:
            module.updateParam(param_key, param_content['content'])

    def handle_page_changed(self) -> None:
        if not self.imgtrans_thread.isRunning():
            # Killing a QThread inside native model inference can abort the
            # process. Let it finish, discard its result, then run the next page.
            self._pending_canvas_inpaint = None
            self.run_canvas_inpaint = False
            self._canvas_inpaint_source = None
