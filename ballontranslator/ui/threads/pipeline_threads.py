"""Background worker threads for AI pipeline stages (detection, OCR, inpainting, translation)."""
import time
import threading
import concurrent.futures
from typing import Callable, List, Optional, Union
import os.path as osp
import cv2
import numpy as np
from packaging.requirements import Requirement
from qtpy.QtCore import QThread, Signal, QObject, QLocale, QTimer
from qtpy.QtWidgets import QFileDialog, QMessageBox, QCheckBox
from ballontranslator.ui.core.funcmaps import get_maskseg_method
from .packageinstall_thread import PackageInstallThread
from ballontranslator.ui.package_manager import create_package_manager
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
    BaseTranslator, InpainterBase, TextDetectorBase, OCRBase, merge_config_module_params
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
from ballontranslator.ui.custom_widget import ImgtransProgressMessageBox, ParamComboBox, ProgressMessageBox
from ballontranslator.ui.configpanel import ConfigPanel
from ballontranslator.utils.project.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.project.config import pcfg, RunStatus, save_config, LLMTranslateContext
from ballontranslator.utils.project.llm_profiles import LLM_DETECTOR_KEY, LLM_INPAINT_KEY, LLM_OCR_KEY
from ballontranslator.utils.project.global_callbacks import register_global_callback
from ballontranslator.utils.project.stage_auditor import StageAuditor, StageIssue, is_safety_refusal_error
cfg_module = pcfg.module


# RUN can report the same missing LLM key from multiple page workers.
_llm_key_dialog_lock = threading.Lock()
_shown_llm_key_dialog_profiles = set()
_llm_model_dialog_lock = threading.Lock()
_shown_llm_model_dialog_profiles = set()
_llm_base_url_dialog_lock = threading.Lock()
_shown_llm_base_url_dialog_profiles = set()


def _create_page_error_dialog(
    exception: Exception,
    error_msg: str,
    exception_type: str,
    page_key: Optional[str] = None,
    page_label: str = 'Page',
) -> None:

    if page_key:
        error_msg = f'{error_msg}\n{page_label}: {page_key}'

    create_error_dialog(
        exception,
        error_msg,
        exception_type,
    )


def _show_llm_key_required_dialog(error: LLMApiKeyRequiredError):
    profile_key = error.profile_id or error.profile_name
    with _llm_key_dialog_lock:
        if profile_key in _shown_llm_key_dialog_profiles:
            return
        _shown_llm_key_dialog_profiles.add(profile_key)
    if not shared.HEADLESS:
        shared.show_llm_key_dialog_in_mainthread(error.profile_id, error.profile_name)
    else:
        LOGGER.error(str(error))


def _show_llm_model_required_dialog(error: LLMModelRequiredError):
    profile_key = (error.profile_id or error.profile_name, error.target)
    with _llm_model_dialog_lock:
        if profile_key in _shown_llm_model_dialog_profiles:
            return
        _shown_llm_model_dialog_profiles.add(profile_key)
    if not shared.HEADLESS:
        shared.show_llm_model_dialog_in_mainthread(error.profile_id, error.profile_name, error.target)
    else:
        LOGGER.error(str(error))


def _show_llm_base_url_required_dialog(error: LLMBaseURLRequiredError):
    profile_key = (error.profile_id or error.profile_name, error.target)
    with _llm_base_url_dialog_lock:
        if profile_key in _shown_llm_base_url_dialog_profiles:
            return
        _shown_llm_base_url_dialog_profiles.add(profile_key)
    if not shared.HEADLESS:
        shared.show_llm_base_url_dialog_in_mainthread(error.profile_id, error.profile_name, error.target)
    else:
        LOGGER.error(str(error))


def _show_llm_user_action_required_dialog(
    error: LLMUserActionRequiredError,
    error_msg: str,
    exception_type: str,
    page_key: Optional[str] = None,
    page_label: str = 'Page',
) -> None:
    """Route known setup errors and show every other actionable LLM failure.

    >>> issubclass(LLMApiKeyRequiredError, LLMUserActionRequiredError)
    True
    """
    if isinstance(error, LLMApiKeyRequiredError):
        _show_llm_key_required_dialog(error)
        return
    if isinstance(error, LLMModelRequiredError):
        _show_llm_model_required_dialog(error)
        return
    if isinstance(error, LLMBaseURLRequiredError):
        _show_llm_base_url_required_dialog(error)
        return
    profile_id = str(getattr(error, 'profile_id', '') or '')
    _create_page_error_dialog(
        error,
        error_msg,
        f'{exception_type}:{type(error).__name__}:{profile_id}',
        page_key,
        page_label,
    )


def _reset_llm_key_required_dialogs():
    with _llm_key_dialog_lock:
        _shown_llm_key_dialog_profiles.clear()
    with _llm_model_dialog_lock:
        _shown_llm_model_dialog_profiles.clear()
    with _llm_base_url_dialog_lock:
        _shown_llm_base_url_dialog_profiles.clear()


def _mark_translation_finished(
    project: ProjImgTrans,
    page_key: str,
    translator: BaseTranslator,
) -> None:
    """Mark completion, then let the owning translator commit page context."""
    project.mark_translation_finished(page_key, translator.lang_target)
    translator.on_page_translation_finished(project, page_key)


class ModuleThread(QThread):
    """Worker thread that prepares and swaps one lazily selected module.

    Dependency checks, file preparation, imports, and model loading happen here
    so the active module remains usable if preparation is cancelled.

    Example:
        >>> thread = ModuleThread('ocr', registry)  # doctest: +SKIP
        >>> thread.requestCancelModuleInit()  # doctest: +SKIP
        >>> thread.cancel_event.is_set()  # doctest: +SKIP
        True
    """

    finish_set_module = Signal()
    module_prepare_progress = Signal(dict)
    _failed_set_module_msg = 'Failed to set module.'
    module_thread_stopped = Signal()

    def __init__(self, module_key: str, MODULE_REGISTER: Registry, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.job = None
        self.module: Union[TextDetectorBase, BaseTranslator, InpainterBase, OCRBase] = None
        self.module_register = MODULE_REGISTER
        self.module_key = module_key

        self.pipeline_pagekey_queue = []
        self.finished_counter = 0
        self.num_process_pages = 0
        self.imgtrans_proj: ProjImgTrans = None
        self.pipeline_stop_event = None
        # Shared with download helpers so Stop can cancel preparation cooperatively.
        self.cancel_event = threading.Event()
        self.last_set_success = False
        self.last_set_module_name = ''
        self.last_error = None
        self.last_missing_requirements = []

    def _emit_prepare_progress(self, payload: dict):
        payload = dict(payload)
        payload.setdefault('module_key', self.module_key)
        payload.setdefault('module', self.last_set_module_name)
        self.module_prepare_progress.emit(payload)

    def _prepare_module_class(self, module_name: str):
        # Prepare dependency/files first; resolving the class is the first real import.
        spec = self.module_register.get_spec(module_name)
        if spec is None:
            raise KeyError(f'Unknown {self.module_key} module: {module_name}')
        self._emit_prepare_progress({'event': 'checking_dependencies', 'message': self.tr('Checking dependencies')})
        if not ensure_module_files(spec, progress_callback=self._emit_prepare_progress, cancel_event=self.cancel_event):
            raise RuntimeError(f'Failed to prepare files for {self.module_key} module: {module_name}')
        if self.cancel_event.is_set():
            raise DownloadCancelled('Module preparation cancelled by user.')
        self._emit_prepare_progress({'event': 'importing', 'message': self.tr('Importing module')})
        module_class = self.module_register.resolve_module(module_name)
        if module_class is None:
            raise KeyError(f'Failed to resolve {self.module_key} module: {module_name}')
        return module_class

    def _discard_module_instance(self, module):
        if module is None:
            return
        try:
            module.unload_model(empty_cache=True)
        except Exception as e:
            LOGGER.warning(f'Failed to unload {self.module_key} module after failed preparation: {e}')

    def _set_module(self, module_name: str):
        old_module = self.module
        new_module = None
        self.last_set_module_name = module_name
        self.last_set_success = False
        self.last_error = None
        self.last_missing_requirements = []
        self.cancel_event.clear()
        try:
            same_module = old_module is not None and old_module.name == module_name
            if same_module and old_module.all_model_loaded():
                self.last_set_success = True
                return
            module: Union[TextDetectorBase, BaseTranslator, InpainterBase, OCRBase] = self._prepare_module_class(module_name)
            if same_module:
                new_module = old_module
                old_module = None
            else:
                params = cfg_module.get_params(self.module_key).get(module_name)
                self._emit_prepare_progress({'event': 'instantiating', 'message': self.tr('Creating module')})
                if params is not None:
                    new_module = module(**params)
                else:
                    new_module = module()
            if self.cancel_event.is_set():
                raise DownloadCancelled('Module preparation cancelled by user.')
            self._emit_prepare_progress({'event': 'loading_model', 'message': self.tr('Loading model')})
            new_module.load_model()
            if self.cancel_event.is_set():
                raise DownloadCancelled('Module preparation cancelled by user.')
            if old_module is not None:
                old_module.unload_model(empty_cache=True)
                old_module = None
            self.module = new_module
            new_module = None
            self.last_set_success = True
        except DownloadCancelled as e:
            # Failed preparation should leave the user-selected module unresolved, not fall back.
            self.module = None
            self._discard_module_instance(new_module)
            self._discard_module_instance(old_module)
            self.last_error = e
            LOGGER.info(f'Cancelled preparing {self.module_key} module {module_name}.')
        except MissingDependency as e:
            self.module = None
            self._discard_module_instance(new_module)
            self._discard_module_instance(old_module)
            self.last_error = e
            self.last_missing_requirements = e.requirements
        except Exception as e:
            self.module = None
            self._discard_module_instance(new_module)
            self._discard_module_instance(old_module)
            missing = self._missing_requirements_from_exception(module_name, e)
            if missing:
                self.last_error = e
                self.last_missing_requirements = missing
            else:
                self.last_error = e
        finally:
            self.finish_set_module.emit()

    def _missing_requirements_from_exception(self, module_name: str, exception: Exception) -> List[str]:
        spec = self.module_register.get_spec(module_name)
        dependencies = spec.dependencies if spec is not None and spec.dependencies else []
        if not dependencies:
            return []

        import_name = None
        probe = exception
        while probe is not None:
            if isinstance(probe, ModuleNotFoundError):
                import_name = probe.name or import_name
                if import_name:
                    break
            if isinstance(probe, LazyModuleError) and isinstance(probe.__cause__, ModuleNotFoundError):
                import_name = probe.__cause__.name or import_name
                if import_name:
                    break
            probe = probe.__cause__ or probe.__context__

        if not import_name:
            return []
        requirement = create_package_manager().requirement_for_import_name(import_name, dependencies)
        return [requirement] if requirement is not None else []

    def installMissingPackagesAndSetModule(
        self,
        module_name: str,
        requirements: List[str],
        torch_device: str = None,
        torch_cuda_version: str = None,
    ):
        self.job = lambda: self._install_missing_then_set(
            module_name,
            requirements,
            torch_device=torch_device,
            torch_cuda_version=torch_cuda_version,
        )
        self.start()

    def _install_missing_then_set(
        self,
        module_name: str,
        requirements: List[str],
        torch_device: str = None,
        torch_cuda_version: str = None,
    ):
        self.last_set_module_name = module_name
        self.last_set_success = False
        self.last_error = None
        self.last_missing_requirements = []
        self.cancel_event.clear()
        self._emit_prepare_progress({'event': 'installing_packages', 'message': self.tr('Installing packages')})
        result = create_package_manager().install(
            requirements,
            progress_callback=self._emit_prepare_progress,
            torch_device=torch_device,
            torch_cuda_version=torch_cuda_version,
        )
        if not result.ok:
            self.last_error = RuntimeError(
                f'Failed to install package(s): {", ".join(requirements)}\n'
                f'Command: {result.command_text}\n'
                f'Exit code: {result.returncode}\n'
                f'{result.stderr or result.stdout or result.error}'
            )
            self.finish_set_module.emit()
            return
        self._set_module(module_name)

    def pipeline_finished(self):
        if self.imgtrans_proj is None:
            return True
        elif self.finished_counter >= self.num_process_pages:
            return True
        return False

    def initImgtransPipeline(self, proj: ProjImgTrans, stop_event: threading.Event = None):
        if self.isRunning():
            self.terminate()
        self.imgtrans_proj = proj
        self.finished_counter = 0
        self.pipeline_pagekey_queue.clear()
        self.pipeline_stop_event = stop_event

    def requestCancelModuleInit(self):
        self.cancel_event.set()

    def run(self):
        try:
            if self.job is not None:
                self.job()
        except LLMUserActionRequiredError as e:
            _show_llm_user_action_required_dialog(
                e,
                self.tr('Module task failed.'),
                f'ModuleThreadFailed:{self.module_key}',
            )
            if self.pipeline_stop_event is not None:
                self.pipeline_stop_event.set()
        except LLMRequestStopped:
            LOGGER.info(f'{self.module_key} task stopped by user.')
        except Exception as e:
            create_error_dialog(e, self.tr('Module task failed.'), f'ModuleThreadFailed:{self.module_key}')
        finally:
            self.job = None


class InpaintThread(ModuleThread):

    finish_inpaint = Signal(dict)
    inpainting = False    
    inpaint_failed = Signal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__('inpainter', INPAINTERS, *args, **kwargs)

    @property
    def inpainter(self) -> InpainterBase:
        return self.module

    def setInpainter(self, inpainter: str):
        self.job = lambda : self._set_module(inpainter)
        self.start()

    def inpaint(self, img: np.ndarray, mask: np.ndarray, img_key: str = None, inpaint_rect=None):
        self.job = lambda : self._inpaint(img, mask, img_key, inpaint_rect)
        self.start()
    
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, img_key: str = None, inpaint_rect=None):
        inpaint_dict = {}
        self.inpainting = True
        try:
            inpainted = self.inpainter.inpaint(img, mask)
            inpaint_dict = {
                'inpainted': inpainted,
                'img': img,
                'mask': mask,
                'img_key': img_key,
                'inpaint_rect': inpaint_rect
            }
            self.finish_inpaint.emit(inpaint_dict)
        except LLMUserActionRequiredError as e:
            _show_llm_user_action_required_dialog(
                e,
                self.tr('Inpainting Failed.'),
                'InpaintFailed',
            )
            if self.pipeline_stop_event is not None:
                self.pipeline_stop_event.set()
            self.inpainting = False
            self.inpaint_failed.emit()
        except LLMRequestStopped:
            LOGGER.info('Inpainting stopped by user.')
            self.inpainting = False
            self.inpaint_failed.emit()
        except Exception as e:
            create_error_dialog(e, self.tr('Inpainting Failed.'), 'InpaintFailed')
            self.inpainting = False
            self.inpaint_failed.emit()
        self.inpainting = False


class TextDetectThread(ModuleThread):
    
    finish_detect_page = Signal(str)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__('textdetector', TEXTDETECTORS, *args, **kwargs)

    def setTextDetector(self, textdetector: str):
        self.job = lambda : self._set_module(textdetector)
        self.start()

    @property
    def textdetector(self) -> TextDetectorBase:
        return self.module


class OCRThread(ModuleThread):

    finish_ocr_page = Signal(str)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__('ocr', OCR, *args, **kwargs)

    def setOCR(self, ocr: str):
        self.job = lambda : self._set_module(ocr)
        self.start()
    
    @property
    def ocr(self) -> OCRBase:
        return self.module


class TranslateThread(ModuleThread):

    progress_changed = Signal(int)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__('translator', TRANSLATORS, *args, **kwargs)
        self.translator: BaseTranslator = self.module
        self._queue_lock = threading.Lock()

    def _set_translator(self, translator: str):
        
        old_translator = self.translator
        new_translator = None
        source, target = cfg_module.translate_source, cfg_module.translate_target
        self.last_set_module_name = translator
        self.last_set_success = False
        self.last_error = None
        self.last_missing_requirements = []
        self.cancel_event.clear()
        try:
            same_translator = old_translator is not None and old_translator.name == translator
            if same_translator and old_translator.all_model_loaded():
                self.last_set_success = True
                self.finish_set_module.emit()
                return
            params = cfg_module.translator_params.get(translator)
            translator_module: BaseTranslator = self._prepare_module_class(translator)
            if same_translator:
                new_translator = old_translator
                old_translator = None
            else:
                self._emit_prepare_progress({'event': 'instantiating', 'message': self.tr('Creating module')})
                if params is not None:
                    new_translator = translator_module(source, target, raise_unsupported_lang=False, **params)
                else:
                    new_translator = translator_module(source, target, raise_unsupported_lang=False)
            if self.cancel_event.is_set():
                raise DownloadCancelled('Module preparation cancelled by user.')
            self._emit_prepare_progress({'event': 'loading_model', 'message': self.tr('Loading model')})
            new_translator.load_model()
            if self.cancel_event.is_set():
                raise DownloadCancelled('Module preparation cancelled by user.')
            cfg_module.translate_source = new_translator.lang_source
            cfg_module.translate_target = new_translator.lang_target
            cfg_module.translator = new_translator.name
            if old_translator is not None:
                old_translator.unload_model(empty_cache=True)
                old_translator = None
            self.translator = new_translator
            new_translator = None
            self.last_set_success = True
        except DownloadCancelled as e:
            self.translator = None
            self._discard_module_instance(new_translator)
            self._discard_module_instance(old_translator)
            self.last_error = e
            LOGGER.info(f'Cancelled preparing translator {translator}.')
        except MissingDependency as e:
            self.translator = None
            self._discard_module_instance(new_translator)
            self._discard_module_instance(old_translator)
            self.last_error = e
            self.last_missing_requirements = e.requirements
        except Exception as e:
            self.translator = None
            self._discard_module_instance(new_translator)
            self._discard_module_instance(old_translator)
            missing = self._missing_requirements_from_exception(translator, e)
            if missing:
                self.last_error = e
                self.last_missing_requirements = missing
            else:
                self.last_error = e
        finally:
            self.module = self.translator
            self.finish_set_module.emit()

    def setTranslator(self, translator: str):
        self.job = lambda : self._set_translator(translator)
        self.start()

    def installMissingPackagesAndSetModule(
        self,
        translator: str,
        requirements: List[str],
        torch_device: str = None,
        torch_cuda_version: str = None,
    ):
        self.job = lambda: self._install_missing_then_set_translator(
            translator,
            requirements,
            torch_device=torch_device,
            torch_cuda_version=torch_cuda_version,
        )
        self.start()

    def _install_missing_then_set_translator(
        self,
        translator: str,
        requirements: List[str],
        torch_device: str = None,
        torch_cuda_version: str = None,
    ):
        self.last_set_module_name = translator
        self.last_set_success = False
        self.last_error = None
        self.last_missing_requirements = []
        self.cancel_event.clear()
        self._emit_prepare_progress({'event': 'installing_packages', 'message': self.tr('Installing packages')})
        result = create_package_manager().install(
            requirements,
            progress_callback=self._emit_prepare_progress,
            torch_device=torch_device,
            torch_cuda_version=torch_cuda_version,
        )
        if not result.ok:
            self.last_error = RuntimeError(
                f'Failed to install package(s): {", ".join(requirements)}\n'
                f'Command: {result.command_text}\n'
                f'Exit code: {result.returncode}\n'
                f'{result.stderr or result.stdout or result.error}'
            )
            self.module = self.translator
            self.finish_set_module.emit()
            return
        self._set_translator(translator)

    def _translate_page(
        self,
        project: ProjImgTrans,
        page_key: str,
    ):
        page = project.pages[page_key]
        page_idx = project.pagename2idx(page_key) if hasattr(project, 'pagename2idx') else None
        page_num = page_idx + 1 if page_idx is not None and page_idx >= 0 else page_key
        trans_name = getattr(self.translator, 'name', 'Translator')
        LOGGER.start(f"[Start][Translator][{page_num}] {trans_name} translating {page_key}")
        # A failed or partially completed full-page request must never leave the
        # old page eligible as history.
        project.begin_full_page_translation(page_key)
        success = True
        if hasattr(self.translator, 'set_stop_event'):
            self.translator.set_stop_event(self.pipeline_stop_event)
        try:
            self.translator.translate_textblk_lst(
                page,
                project=project,
                page_key=page_key,
                full_page=True,
            )
            _mark_translation_finished(project, page_key, self.translator)
            LOGGER.stop(f"[Stop][Translator][{page_num}] Translation completed")
        except LLMUserActionRequiredError as e:
            success = False
            _show_llm_user_action_required_dialog(
                e,
                self.tr('Translation Failed.'),
                'TranslationFailed',
                page_key,
                self.tr('Page'),
            )
            if self.pipeline_stop_event is not None:
                self.pipeline_stop_event.set()
        except LLMRequestStopped:
            success = False
            LOGGER.info('Translation stopped by user.')
        except Exception as e:
            success = False
            _create_page_error_dialog(
                e,
                self.tr('Translation Failed.'),
                'TranslationFailed',
                page_key,
                self.tr('Page'),
            )
        return success

    def initImgtransPipeline(self, proj: ProjImgTrans, stop_event: threading.Event = None):
        super().initImgtransPipeline(proj, stop_event)
        with self._queue_lock:
            self.pipeline_pagekey_queue.clear()

    def push_pagekey_queue(self, page_key: str):
        with self._queue_lock:
            self.pipeline_pagekey_queue.append(page_key)

    def runTranslatePipeline(self, imgtrans_proj: ProjImgTrans, stop_event: threading.Event):
        self.initImgtransPipeline(imgtrans_proj, stop_event)
        self.job = self._run_translate_pipeline
        self.start()

    def _handle_translate_error(self, e: Exception, page_key: str):
        if pcfg.module.pipeline_fault_tolerant:
            LOGGER.warning(f"Translation failed for {page_key}: {e}")
            return
        msg = '{}\n{}: {}'.format(
            self.tr('Translation Failed.'),
            self.tr('Page'),
            page_key,
        )
        if isinstance(e, MissingTranslatorParams):
            msg = msg + '\n' + self.tr('{param} is required for {translator}').format(
                param=str(e),
                translator=self.translator.name,
            )
        create_error_dialog(e, msg, 'TranslationFailed')

    def _run_translate_pipeline(self):
        profile = getattr(self.translator, 'profile', None)
        profile_delay = getattr(profile, 'request_delay', 0.0) if profile else 0.0
        global_delay = getattr(pcfg.module, 'llm_request_delay', 0.0)
        if profile_delay > 0:
            delay = profile_delay
        elif global_delay > 0:
            delay = global_delay
        else:
            delay = self.translator.delay() if self.translator else 0

        stop_event = self.pipeline_stop_event or threading.Event()

        max_workers = 1
        configured_workers = getattr(pcfg.module, 'llm_translate_threads', getattr(pcfg.module, 'llm_threads', 4))
        profile_concurrency = getattr(profile, 'concurrency', 0) if profile else 0
        if profile_concurrency > 0:
            effective_configured_workers = profile_concurrency
        else:
            effective_configured_workers = configured_workers

        if effective_configured_workers > 1 and self.translator is not None:
            low_vram = getattr(self.translator, 'low_vram_mode', False)
            heavy = getattr(self.translator, 'is_computational_intensive', lambda: False)()
            history_context = getattr(pcfg.module, 'llm_translate_context', '') == LLMTranslateContext.HISTORY
            summary_memory = getattr(pcfg.module, 'llm_translate_summary_memory', False)
            if not low_vram and not heavy and not history_context and not summary_memory:
                max_workers = max(1, int(effective_configured_workers))

        LOGGER.debug(
            f"Translation pipeline started with {max_workers} worker(s) "
            f"(effective: {effective_configured_workers}, global: {configured_workers}, delay: {delay}s)."
        )

        if max_workers <= 1:
            while not self.pipeline_finished():
                if stop_event.is_set():
                    self.module_thread_stopped.emit()
                    break

                page_key = None
                with self._queue_lock:
                    if len(self.pipeline_pagekey_queue) > 0:
                        page_key = self.pipeline_pagekey_queue.pop(0)

                if page_key is None:
                    stop_event.wait(0.1)
                    continue

                self.blockSignals(True)
                try:
                    self._translate_page(self.imgtrans_proj, page_key)
                except Exception as e:
                    self._handle_translate_error(e, page_key)
                finally:
                    self.blockSignals(False)

                self.finished_counter += 1
                self.progress_changed.emit(self.finished_counter)

                if not self.pipeline_finished() and delay > 0:
                    stop_event.wait(delay)
            return

        active_futures = {}

        def translate_worker(page_k: str):
            if stop_event.is_set():
                return False
            return self._translate_page(self.imgtrans_proj, page_k)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            while not self.pipeline_finished():
                if stop_event.is_set():
                    self.module_thread_stopped.emit()
                    for fut in list(active_futures.keys()):
                        fut.cancel()
                    break

                # Dispatch jobs while queue has items and capacity remains
                with self._queue_lock:
                    while len(active_futures) < max_workers and len(self.pipeline_pagekey_queue) > 0:
                        next_page_key = self.pipeline_pagekey_queue.pop(0)
                        fut = executor.submit(translate_worker, next_page_key)
                        active_futures[fut] = next_page_key
                        if delay > 0:
                            stop_event.wait(delay)

                if not active_futures and len(self.pipeline_pagekey_queue) == 0:
                    if self.pipeline_finished():
                        break
                    stop_event.wait(0.05)
                    continue

                done, _ = concurrent.futures.wait(
                    list(active_futures.keys()),
                    timeout=0.1,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for fut in done:
                    p_key = active_futures.pop(fut, '')
                    if not fut.cancelled():
                        try:
                            fut.result()
                        except Exception as e:
                            self._handle_translate_error(e, p_key)
                        self.finished_counter += 1
                        self.progress_changed.emit(self.finished_counter)


class ImgtransThread(QThread):

    pipeline_stopped = Signal()
    update_detect_progress = Signal(int)
    update_ocr_progress = Signal(int)
    update_translate_progress = Signal(int)
    update_inpaint_progress = Signal(int)
    page_stage_finished = Signal(str, int)

    finish_blktrans_stage = Signal(str, int)
    finish_blktrans = Signal(int, list)
    unload_modules = Signal(list)

    detect_counter = 0
    ocr_counter = 0
    translate_counter = 0
    inpaint_counter = 0
    num_pages = 0

    def __init__(self, 
                 textdetect_thread: TextDetectThread,
                 ocr_thread: OCRThread,
                 translate_thread: TranslateThread,
                 inpaint_thread: InpaintThread,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.textdetect_thread = textdetect_thread
        self.ocr_thread = ocr_thread
        self.translate_thread = translate_thread
        self.translate_thread.module_thread_stopped.connect(self.on_module_thread_stopped)
        self.translate_thread.finished.connect(self.on_module_thread_stopped)
        self.inpaint_thread = inpaint_thread
        self.job = None
        self.imgtrans_proj: ProjImgTrans = None
        self.num_pages = 0
        self.stop_event = threading.Event()
        self._pipeline_stop_emitted = False
        self.pages_to_process = None
        self.runtime_issues: List[StageIssue] = []
        register_global_callback('user_request_stop', self.isStopRequested)

    def on_module_thread_stopped(self):
        self._emit_pipeline_stopped_if_ready(imgtrans_running=self.isRunning())

    def isStopRequested(self):
        return self.stop_event.is_set()

    def clearStopRequest(self):
        self.stop_event.clear()
        self._pipeline_stop_emitted = False

    def _emit_pipeline_stopped_if_ready(self, imgtrans_running: bool):
        if not self.isStopRequested() or self._pipeline_stop_emitted:
            return
        if imgtrans_running or self.translate_thread.isRunning():
            return
        self._pipeline_stop_emitted = True
        self.pipeline_stopped.emit()

    @property
    def textdetector(self) -> TextDetectorBase:
        return self.textdetect_thread.textdetector

    @property
    def ocr(self) -> OCRBase:
        return self.ocr_thread.ocr
    
    @property
    def translator(self) -> BaseTranslator:
        return self.translate_thread.translator

    @property
    def inpainter(self) -> InpainterBase:
        return self.inpaint_thread.inpainter

    def runImgtransPipeline(
        self,
        imgtrans_proj: ProjImgTrans,
        pages_to_process=None,
    ):
        self.imgtrans_proj = imgtrans_proj
        self.pages_to_process = pages_to_process  # 保存需要处理的页面列表
        self.runtime_issues = []
        self.num_pages = len(self.imgtrans_proj.pages)
        self.clearStopRequest()
        # 创建处理索引到实际页面索引的映射
        self.process_idx_to_page_idx = {}
        self.detect_counter = 0
        self.ocr_counter = 0
        self.translate_counter = 0
        self.inpaint_counter = 0
        self.job = self._imgtrans_pipeline
        self.start()
    
    def requestStop(self):
        """请求停止当前任务"""
        self.stop_event.set()

    def _stop_on_stage_failure(
        self,
        exception: ModuleRunError,
        error_msg: str,
        exception_type: str,
        page_key: str = None,
    ):
        _create_page_error_dialog(
            exception,
            error_msg,
            exception_type,
            page_key,
            self.tr('Page'),
        )
        self.requestStop()

    def runBlktransPipeline(
        self,
        blk_list: List[TextBlock],
        mode: int,
        blk_ids: List[int],
        *,
        page_key: str = None,
    ):
        self.clearStopRequest()
        self.job = lambda : self._blktrans_pipeline(
            blk_list,
            mode,
            blk_ids,
            page_key=page_key,
        )
        self.start()

    def _translate_textblocks(
        self,
        blk_list: List[TextBlock],
        *,
        project: ProjImgTrans = None,
        page_key: str = None,
        full_page: bool = False,
    ) -> bool:
        translator = self.translate_thread.module or self.translate_thread.translator
        if translator is None:
            _create_page_error_dialog(
                RuntimeError('Translator is not loaded.'),
                self.tr('Translation Failed.'),
                'TranslationFailed',
                page_key,
                self.tr('Page'),
            )
            return False
        if hasattr(translator, 'set_stop_event'):
            translator.set_stop_event(self.stop_event)
        try:
            translator.translate_textblk_lst(
                blk_list,
                project=project,
                page_key=page_key,
                full_page=full_page,
            )
            return True
        except LLMUserActionRequiredError as e:
            _show_llm_user_action_required_dialog(
                e,
                self.tr('Translation Failed.'),
                'TranslationFailed',
                page_key,
                self.tr('Page'),
            )
            self.requestStop()
        except LLMRequestStopped:
            LOGGER.info('Translation stopped by user.')
        except Exception as e:
            _create_page_error_dialog(
                e,
                self.tr('Translation Failed.'),
                'TranslationFailed',
                page_key,
                self.tr('Page'),
            )
        return False

    def _translate_full_page(
        self,
        project: ProjImgTrans,
        page_key: str,
        blk_list: List[TextBlock],
    ) -> bool:
        page_idx = project.pagename2idx(page_key) if hasattr(project, 'pagename2idx') else None
        page_num = page_idx + 1 if page_idx is not None and page_idx >= 0 else page_key
        trans_name = getattr(self.translator, 'name', 'Translator')
        LOGGER.start(f"[Start][Translator][{page_num}] {trans_name} translating {page_key}")
        project.begin_full_page_translation(page_key)
        success = self._translate_textblocks(
            blk_list,
            project=project,
            page_key=page_key,
            full_page=True,
        )
        if success:
            _mark_translation_finished(project, page_key, self.translator)
            LOGGER.stop(f"[Stop][Translator][{page_num}] Translation completed")
        return success

    def _blktrans_pipeline(
        self,
        blk_list: List[TextBlock],
        mode: int,
        blk_ids: List[int],
        *,
        page_key: str = None,
    ):
        tgt_img = self.imgtrans_proj.img_array if self.imgtrans_proj is not None else None
        tgt_mask = self.imgtrans_proj.mask_array if self.imgtrans_proj is not None else None
        if tgt_img is None:
            self.finish_blktrans.emit(mode, blk_ids)
            return
        if page_key is None:
            page_key = getattr(self.imgtrans_proj, 'current_img', None)
        if mode >= 0 and mode < 3:
            ocr_module = self.ocr_thread.module
            if hasattr(ocr_module, 'set_stop_event'):
                ocr_module.set_stop_event(self.stop_event)
            try:
                ocr_module.run_ocr(tgt_img, blk_list, split_textblk=True)
            except LLMUserActionRequiredError as e:
                _show_llm_user_action_required_dialog(
                    e,
                    self.tr('OCR Failed.'),
                    'OCRFailed',
                    page_key,
                    self.tr('Page'),
                )
                self.requestStop()
                self.finish_blktrans.emit(mode, blk_ids)
                return
            except LLMRequestStopped:
                LOGGER.info('OCR stopped by user.')
                self.finish_blktrans.emit(mode, blk_ids)
                return
            except ModuleRunError as e:
                _create_page_error_dialog(
                    e,
                    self.tr('OCR Failed.'),
                    'OCRFailed',
                    page_key,
                    self.tr('Page'),
                )
                self.finish_blktrans.emit(mode, blk_ids)
                return
            self.finish_blktrans.emit(mode, blk_ids)

        if mode != 0 and mode < 3:
            success = self._translate_textblocks(
                blk_list,
                project=self.imgtrans_proj,
                page_key=page_key,
            )
            # A selected run completes the page only when no source-bearing block
            # is still waiting for a translation.
            if success and page_key is not None:
                page = self.imgtrans_proj.pages.get(page_key)
                if page is not None and all(
                    not block.get_text().strip()
                    or bool(str(getattr(block, 'translation', '') or '').strip())
                    for block in page
                ):
                    _mark_translation_finished(
                        self.imgtrans_proj,
                        page_key,
                        self.translator,
                    )
            self.finish_blktrans.emit(mode, blk_ids)
        if mode > 1:
            if hasattr(self.inpaint_thread.inpainter, 'set_stop_event'):
                self.inpaint_thread.inpainter.set_stop_event(self.stop_event)
            to_inpaint = self.imgtrans_proj.inpainted_array
            im_h, im_w = tgt_img.shape[:2]
            progress_prod = 100. / len(blk_list) if len(blk_list) > 0 else 0
            for ii, blk in enumerate(blk_list):
                xyxy_ori = np.array(blk.xyxy, dtype=np.int64)
                xyxy_ori[::2] = np.clip(xyxy_ori[::2], 0, im_w)
                xyxy_ori[1::2] = np.clip(xyxy_ori[1::2], 0, im_h)
                ox1, oy1, ox2, oy2 = xyxy_ori
                blk.region_inpaint_dict = None
                if oy2 - oy1 > 2 and ox2 - ox1 > 2:
                    xyxy = enlarge_window(xyxy_ori.tolist(), im_w, im_h)
                    xyxy = np.array(xyxy, dtype=np.int64)
                    x1, y1, x2, y2 = xyxy.astype(np.int64)
                    im = np.copy(to_inpaint[y1: y2, x1: x2])
                    maskseg_method = get_maskseg_method()
                    inpaint_mask_array, ballon_mask, bub_dict = maskseg_method(im, mask=tgt_mask[y1: y2, x1: x2])
                    mask = self.post_process_mask(inpaint_mask_array)
                    if mask.sum() > 0:
                        if self.inpaint_thread.inpainter is None:
                            LOGGER.warning('[Inpainter] Inpainter is not available or not loaded. Skipping block inpaint.')
                        else:
                            try:
                                inpainted = self.inpaint_thread.inpainter.inpaint(im, mask)
                            except LLMUserActionRequiredError as e:
                                _show_llm_user_action_required_dialog(
                                    e,
                                    self.tr('Inpainting Failed.'),
                                    'InpaintFailed',
                                    page_key,
                                    self.tr('Page'),
                                )
                                self.requestStop()
                                self.finish_blktrans.emit(mode, blk_ids)
                                return
                            except LLMRequestStopped:
                                LOGGER.info('Inpainting stopped by user.')
                                self.requestStop()
                                self.finish_blktrans.emit(mode, blk_ids)
                                return
                            rx1, ry1 = max(0, ox1 - x1), max(0, oy1 - y1)
                            rx2, ry2 = min(x2 - x1, ox2 - x1), min(y2 - y1, oy2 - y1)
                            if ry2 > ry1 and rx2 > rx1:
                                blk.region_inpaint_dict = {
                                    'img': im[ry1: ry2, rx1: rx2],
                                    'mask': mask[ry1: ry2, rx1: rx2],
                                    'inpaint_rect': [int(ox1), int(oy1), int(ox2), int(oy2)],
                                    'inpainted': inpainted[ry1: ry2, rx1: rx2]
                                }
                    self.finish_blktrans_stage.emit('inpaint', int((ii+1) * progress_prod))
        self.finish_blktrans.emit(mode, blk_ids)

    def _imgtrans_pipeline(self):
        self.detect_counter = 0
        self.ocr_counter = 0
        self.translate_counter = 0
        self.inpaint_counter = 0
        
        # 如果指定了pages_to_process，只处理这些页面
        all_pages = list(self.imgtrans_proj.pages.keys())
        if self.pages_to_process is not None and len(self.pages_to_process) > 0:
            pages_to_iterate = self.pages_to_process
            self.num_pages = num_pages = len(self.pages_to_process)
            # 建立处理索引到实际页面索引的映射
            for process_idx, page_name in enumerate(pages_to_iterate):
                if page_name in all_pages:
                    self.process_idx_to_page_idx[process_idx] = all_pages.index(page_name)
        else:
            pages_to_iterate = all_pages
            self.num_pages = num_pages = len(self.imgtrans_proj.pages)
            # 处理索引等于实际页面索引
            for i in range(num_pages):
                self.process_idx_to_page_idx[i] = i

        self.pipeline_start_time = time.time()
        active_stages = []
        if cfg_module.enable_upscale and cfg_module.upscaler != 'None':
            active_stages.append(f'Upscale({cfg_module.upscaler})')
        if cfg_module.enable_detect:
            active_stages.append(f'Detect({cfg_module.textdetector})')
        if cfg_module.enable_ocr:
            active_stages.append(f'OCR({cfg_module.ocr})')
        if cfg_module.enable_translate:
            active_stages.append(f'Translate({cfg_module.translator})')
        if cfg_module.enable_inpaint:
            active_stages.append(f'Inpaint({cfg_module.inpainter})')
        stages_str = ', '.join(active_stages) if active_stages else 'None'
        LOGGER.start(f'Translation pipeline started: {len(pages_to_iterate)} page(s) | Stages: {stages_str}')
        if cfg_module.enable_translate and self.translator is not None:
            describe_run = getattr(
                self.translator,
                'translation_run_description',
                None,
            )
            if callable(describe_run):
                try:
                    LOGGER.info(describe_run())
                except Exception as error:
                    # Diagnostics must never make an otherwise valid run fail.
                    LOGGER.debug(
                        f'Unable to describe translation run settings: {error}'
                    )
        self.textdetect_thread.num_process_pages = self.num_pages
        self.ocr_thread.num_process_pages = self.num_pages
        self.inpaint_thread.num_process_pages = self.num_pages
        self.translate_thread.num_process_pages = self.num_pages

        low_vram_trans = False
        if self.translator is not None:
            low_vram_trans = self.translator.low_vram_mode
            self.parallel_trans = not self.translator.is_computational_intensive() and not low_vram_trans
        else:
            self.parallel_trans = False
        if self.parallel_trans and cfg_module.enable_translate:
            self.translate_thread.runTranslatePipeline(self.imgtrans_proj, self.stop_event)

        upscaler_instance = None
        if cfg_module.enable_upscale and cfg_module.upscaler != 'None':
            upscaler_name = cfg_module.upscaler
            valid_upscalers = GET_VALID_UPSCALERS()
            if upscaler_name in valid_upscalers:
                try:
                    spec = UPSCALERS.get_spec(upscaler_name)
                    if spec and hasattr(spec, 'ensure_files'):
                        spec.ensure_files()
                    upscaler_cls = UPSCALERS.get(upscaler_name)
                    saved_params = cfg_module.get_params('upscaler').get(upscaler_name, {})
                    upscaler_instance = upscaler_cls(**saved_params)
                except Exception as e:
                    LOGGER.error(f'Failed to initialize upscaler {upscaler_name}: {e}')

        for imgname in pages_to_iterate:
            
            # 检查是否请求停止
            if self.isStopRequested():
                LOGGER.info('Image translation pipeline stopped by user')
                break
                
            page_idx = self.imgtrans_proj.pagename2idx(imgname)
            page_num = page_idx + 1 if page_idx is not None and page_idx >= 0 else imgname
            img = self.imgtrans_proj.read_img(imgname)

            if upscaler_instance is not None:
                try:
                    old_h, old_w = img.shape[:2]
                    scale_param = cfg_module.get_params('upscaler').get(cfg_module.upscaler, {}).get('scale', 1)
                    if isinstance(scale_param, dict) and 'value' in scale_param:
                        scale_val = int(scale_param['value'])
                    else:
                        scale_val = int(scale_param)
                    denoise_val = cfg_module.upscale_denoise
                    if scale_val <= 1 and not denoise_val:
                        LOGGER.info(f"[Upscale][{page_num}] Skipped: Scale is 1x and Denoise is Off.")
                        if hasattr(RunStatus, 'FIN_UPSCALE'):
                            self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_UPSCALE)
                    else:
                        upscaled = upscaler_instance.upscale(img, scale=scale_val, denoise=denoise_val)
                        if upscaled is not None:
                            new_h, new_w = upscaled.shape[:2]
                            img = upscaled
                            curr_img_path = osp.join(self.imgtrans_proj.directory, imgname)
                            cv2.imwrite(curr_img_path, upscaled)
                        if getattr(self.imgtrans_proj, 'current_img', None) == imgname:
                            self.imgtrans_proj.img_array = upscaled
                        if old_w > 0 and old_h > 0 and (new_w != old_w or new_h != old_h):
                            scale_x = new_w / old_w
                            scale_y = new_h / old_h
                            if imgname in self.imgtrans_proj.pages:
                                for blk in self.imgtrans_proj.pages[imgname]:
                                    blk.xywh = [
                                        round(blk.xywh[0] * scale_x),
                                        round(blk.xywh[1] * scale_y),
                                        round(blk.xywh[2] * scale_x),
                                        round(blk.xywh[3] * scale_y),
                                    ]
                                    if hasattr(blk, 'font_size') and blk.font_size:
                                        blk.font_size = round(blk.font_size * scale_y)
                        LOGGER.info(f"[Upscale][{page_num}] Page '{imgname}' upscaled with {cfg_module.upscaler} ({old_w}x{old_h} -> {new_w}x{new_h})")
                        if hasattr(RunStatus, 'FIN_UPSCALE'):
                            self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_UPSCALE)
                except Exception as e:
                    LOGGER.error(f"[Upscale][{page_num}] Failed to upscale {imgname}: {e}")

            if cfg_module.enable_upscale and cfg_module.standardize_dimensions and cfg_module.standardize_target_width > 0 and cfg_module.standardize_target_height > 0:
                tgt_w = cfg_module.standardize_target_width
                tgt_h = cfg_module.standardize_target_height
                cur_h, cur_w = img.shape[:2]
                if cur_w != tgt_w or cur_h != tgt_h:
                    scale_x = tgt_w / cur_w
                    scale_y = tgt_h / cur_h
                    interp = cv2.INTER_AREA if (cur_w > tgt_w and cur_h > tgt_h) else cv2.INTER_LANCZOS4
                    img = cv2.resize(img, (tgt_w, tgt_h), interpolation=interp)
                    save_path = osp.join(self.imgtrans_proj.directory, imgname)
                    cv2.imwrite(save_path, img)
                    if imgname in self.imgtrans_proj.pages:
                        for blk in self.imgtrans_proj.pages[imgname]:
                            blk.xywh = [
                                round(blk.xywh[0] * scale_x),
                                round(blk.xywh[1] * scale_y),
                                round(blk.xywh[2] * scale_x),
                                round(blk.xywh[3] * scale_y),
                            ]
                            if hasattr(blk, 'font_size') and blk.font_size:
                                blk.font_size = round(blk.font_size * scale_y)
                    LOGGER.info(f"[Standardize] Page '{imgname}' resized: {cur_w}x{cur_h} -> {tgt_w}x{tgt_h}")

            mask = blk_list = None
            need_save_mask = False
            blk_removed: List[TextBlock] = []
            if cfg_module.enable_detect:
                LOGGER.start(f"[Start][Detector][{page_num}] {self.textdetector.name} analyzing {imgname}")
                # Detection can replace or reorder blocks, so old translations
                # are never compatible even if detection later fails.
                self.imgtrans_proj.begin_detection(imgname)
                if hasattr(self.textdetector, 'set_stop_event'):
                    self.textdetector.set_stop_event(self.stop_event)
                try:
                    mask, blk_list = self.textdetector.detect(img, self.imgtrans_proj)
                    need_save_mask = True
                except LLMUserActionRequiredError as e:
                    _show_llm_user_action_required_dialog(
                        e,
                        self.tr('Text Detection Failed.'),
                        'TextDetectFailed',
                        imgname,
                        self.tr('Page'),
                    )
                    self.requestStop()
                    break
                except LLMRequestStopped:
                    LOGGER.info('Text detection stopped by user.')
                    self.requestStop()
                    break
                except ModuleRunError as e:
                    if pcfg.module.pipeline_fault_tolerant:
                        LOGGER.warning(f"[Detector][{page_num}] Text detection failed for {imgname}: {e}")
                        err_type = 'safety_refusal' if is_safety_refusal_error(e) else 'stage_failed'
                        self.runtime_issues.append(StageIssue(
                            page_key=imgname,
                            stage='detect',
                            error_type=err_type,
                            message=str(e),
                            page_index=page_idx if page_idx is not None else -1,
                        ))
                        self.detect_counter += 1
                        self.update_detect_progress.emit(self.detect_counter)
                        if cfg_module.enable_ocr:
                            self.ocr_counter += 1
                            self.update_ocr_progress.emit(self.ocr_counter)
                        if cfg_module.enable_translate and not self.parallel_trans:
                            self.translate_counter += 1
                            self.update_translate_progress.emit(self.translate_counter)
                        if cfg_module.enable_inpaint:
                            self.inpaint_counter += 1
                            self.update_inpaint_progress.emit(self.inpaint_counter)
                        continue
                    else:
                        self._stop_on_stage_failure(
                            e,
                            self.tr('Text Detection Failed.'),
                            'TextDetectFailed',
                            imgname,
                        )
                        break
                self.detect_counter += 1
                if pcfg.module.keep_exist_textlines:
                    blk_list = self.imgtrans_proj.pages[imgname] + blk_list
                    blk_list = sort_regions(blk_list)
                    existed_mask = self.imgtrans_proj.load_mask_by_imgname(imgname)
                    if existed_mask is not None:
                        mask = np.bitwise_or(mask, existed_mask)
                self.imgtrans_proj.pages[imgname] = blk_list

                if mask is not None and not cfg_module.enable_ocr:
                    self.imgtrans_proj.save_mask(imgname, mask)
                    need_save_mask = False
                    
                self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_DET)
                self.update_detect_progress.emit(self.detect_counter)
                self.page_stage_finished.emit('detect', page_idx)
                LOGGER.stop(f"[Stop][Detector][{page_num}] Found {len(blk_list)} text blocks")

            if blk_list is None:
                blk_list = self.imgtrans_proj.pages[imgname] if imgname in self.imgtrans_proj.pages else []

            if cfg_module.enable_ocr:
                LOGGER.start(f"[Start][OCR][{page_num}] {self.ocr.name} recognizing text on {imgname}")
                if hasattr(self.ocr, 'set_stop_event'):
                    self.ocr.set_stop_event(self.stop_event)
                try:
                    is_llm_detector = getattr(self.textdetector, 'name', '') == LLM_DETECTOR_KEY
                    already_has_text = bool(blk_list and all(not text_is_empty(b.get_text()) for b in blk_list))
                    if is_llm_detector and (already_has_text or len(blk_list) == 0):
                        LOGGER.stop(f"[Stop][OCR][{page_num}] All-in-One LLM Vision: text already recognized during detection; bypassing duplicate OCR stage.")
                        self.imgtrans_proj.pages[imgname] = blk_list
                        self.ocr_counter += 1
                    else:
                        # full_page is an LLMOCR-only extension; preserve the
                        # historical two-argument contract for custom OCR modules.
                        if getattr(self.ocr, 'name', '') == LLM_OCR_KEY:
                            ocr_result = self.ocr.run_ocr(
                                img,
                                blk_list,
                                full_page=True,
                            )
                        else:
                            ocr_result = self.ocr.run_ocr(img, blk_list)
                        if isinstance(ocr_result, list):
                            blk_list = ocr_result
                        self.imgtrans_proj.pages[imgname] = blk_list
                        self.ocr_counter += 1
                        LOGGER.stop(f"[Stop][OCR][{page_num}] OCR completed for {len(blk_list)} blocks")

                    if pcfg.restore_ocr_empty:
                        blk_list_updated = []
                        for blk in blk_list:
                            text = blk.get_text()
                            if text_is_empty(text):
                                blk_removed.append(blk)
                            else:
                                blk_list_updated.append(blk)

                        if len(blk_removed) > 0:
                            blk_list.clear()
                            blk_list += blk_list_updated

                            if mask is None:
                                mask = self.imgtrans_proj.load_mask_by_imgname(imgname)
                            if mask is not None:
                                inpainted = None
                                if not cfg_module.enable_inpaint:
                                    inpainted = self.imgtrans_proj.load_inpainted_by_imgname(imgname)
                                for blk in blk_removed:
                                    xywh = blk.bounding_rect()
                                    blk_mask, xyxy = get_block_mask(xywh, mask, blk.angle)
                                    x1, y1, x2, y2 = xyxy
                                    if blk_mask is not None:
                                        mask[y1: y2, x1: x2] = 0
                                        if inpainted is not None:
                                            mskpnt = np.where(blk_mask)
                                            inpainted[y1: y2, x1: x2][mskpnt] = img[y1: y2, x1: x2][mskpnt]
                                        need_save_mask = True
                                if inpainted is not None and need_save_mask:
                                    self.imgtrans_proj.save_inpainted(imgname, inpainted)
                                if need_save_mask:
                                    self.imgtrans_proj.save_mask(imgname, mask)
                                    need_save_mask = False
                except LLMUserActionRequiredError as e:
                    _show_llm_user_action_required_dialog(
                        e,
                        self.tr('OCR Failed.'),
                        'OCRFailed',
                        imgname,
                        self.tr('Page'),
                    )
                    self.requestStop()
                    break
                except LLMRequestStopped:
                    LOGGER.info('OCR stopped by user.')
                    self.requestStop()
                    break
                except ModuleRunError as e:
                    if pcfg.module.pipeline_fault_tolerant:
                        LOGGER.warning(f"[OCR][{page_num}] OCR failed for {imgname}: {e}")
                        err_type = 'safety_refusal' if is_safety_refusal_error(e) else 'stage_failed'
                        self.runtime_issues.append(StageIssue(
                            page_key=imgname,
                            stage='ocr',
                            error_type=err_type,
                            message=str(e),
                            page_index=page_idx if page_idx is not None else -1,
                        ))
                        self.ocr_counter += 1
                        self.update_ocr_progress.emit(self.ocr_counter)
                    else:
                        self._stop_on_stage_failure(
                            e,
                            self.tr('OCR Failed.'),
                            'OCRFailed',
                            imgname,
                        )
                        break
                self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_OCR)
                self.update_ocr_progress.emit(self.ocr_counter)
                page_idx = self.imgtrans_proj.pagename2idx(imgname)
                self.page_stage_finished.emit('ocr', page_idx)

            if need_save_mask and mask is not None:
                self.imgtrans_proj.save_mask(imgname, mask)
                need_save_mask = False

            # Headless mode uses this same router; it has no translation-only branch.
            if cfg_module.enable_translate:
                if self.parallel_trans:
                    self.translate_thread.push_pagekey_queue(imgname)
                elif not low_vram_trans:
                    try:
                        self._translate_full_page(
                            self.imgtrans_proj,
                            imgname,
                            blk_list,
                        )
                    except Exception as e:
                        if pcfg.module.pipeline_fault_tolerant:
                            LOGGER.warning(f"[Translator][{page_num}] Translation failed for {imgname}: {e}")
                            err_type = 'safety_refusal' if is_safety_refusal_error(e) else 'stage_failed'
                            self.runtime_issues.append(StageIssue(
                                page_key=imgname,
                                stage='translate',
                                error_type=err_type,
                                message=str(e),
                                page_index=page_idx if page_idx is not None else -1,
                            ))
                        else:
                            raise e
                    self.translate_counter += 1
                    self.update_translate_progress.emit(self.translate_counter)

            if self.isStopRequested():
                LOGGER.info('Image translation pipeline stopped.')
                break
                        
            if cfg_module.enable_inpaint:
                if self.inpainter is None:
                    LOGGER.warning(f"[Inpainter][{page_num}] Inpainter is not available or not loaded. Skipping.")
                else:
                    LOGGER.start(f"[Start][Inpainter][{page_num}] {self.inpainter.name} cleaning {imgname}")
                    if hasattr(self.inpainter, 'set_stop_event'):
                        self.inpainter.set_stop_event(self.stop_event)
                    if mask is None:
                        mask = self.imgtrans_proj.load_mask_by_imgname(imgname)
                        
                    if mask is not None or cfg_module.inpainter == LLM_INPAINT_KEY:
                        try:
                            inpainted = self.inpainter.inpaint(img, mask, blk_list)
                            self.imgtrans_proj.save_inpainted(imgname, inpainted)
                        except LLMUserActionRequiredError as e:
                            _show_llm_user_action_required_dialog(
                                e,
                                self.tr('Inpainting Failed.'),
                                'InpaintFailed',
                                imgname,
                                self.tr('Page'),
                            )
                            self.requestStop()
                            break
                        except LLMRequestStopped:
                            LOGGER.info('Inpainting stopped by user.')
                            self.requestStop()
                            break
                        except Exception as e:
                            if pcfg.module.pipeline_fault_tolerant:
                                LOGGER.warning(f"[Inpainter][{page_num}] Inpainting failed for {imgname}: {e}")
                                err_type = 'safety_refusal' if is_safety_refusal_error(e) else 'stage_failed'
                                self.runtime_issues.append(StageIssue(
                                    page_key=imgname,
                                    stage='inpaint',
                                    error_type=err_type,
                                    message=str(e),
                                    page_index=page_idx if page_idx is not None else -1,
                                ))
                            else:
                                _create_page_error_dialog(
                                    e,
                                    self.tr('Inpainting Failed.'),
                                    'InpaintFailed',
                                    imgname,
                                    self.tr('Page'),
                                )
                        
                    self.inpaint_counter += 1
                    self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_INPAINT)
                    self.update_inpaint_progress.emit(self.inpaint_counter)
                    self.page_stage_finished.emit('inpaint', page_idx)
                    LOGGER.stop(f"[Stop][Inpainter][{page_num}] Inpainting completed")
            else:
                if len(blk_removed) > 0:
                    self.imgtrans_proj.load_mask_by_imgname
        
        if cfg_module.enable_translate and low_vram_trans and not self.isStopRequested():
            unload_modules(self, ['textdetector', 'inpainter', 'ocr'])
            for imgname in pages_to_iterate:
                # 检查是否请求停止
                if self.isStopRequested():
                    LOGGER.info('Translation stopped by user')
                    break
                    
                blk_list = self.imgtrans_proj.pages[imgname]
                try:
                    self._translate_full_page(
                        self.imgtrans_proj,
                        imgname,
                        blk_list,
                    )
                except Exception as e:
                    if pcfg.module.pipeline_fault_tolerant:
                        LOGGER.warning(f"[Translator][{page_num}] Translation failed for {imgname}: {e}")
                        err_type = 'safety_refusal' if is_safety_refusal_error(e) else 'stage_failed'
                        self.runtime_issues.append(StageIssue(
                            page_key=imgname,
                            stage='translate',
                            error_type=err_type,
                            message=str(e),
                            page_index=page_idx if page_idx is not None else -1,
                        ))
                    else:
                        raise e
                self.translate_counter += 1
                self.update_translate_progress.emit(self.translate_counter)

        if self.parallel_trans and cfg_module.enable_translate and not self.isStopRequested():
            while not self.translate_thread.pipeline_finished() and not self.isStopRequested():
                if not self.translate_thread.isRunning():
                    break
                self.stop_event.wait(0.05)
            if self.translate_thread.isRunning():
                self.translate_thread.wait(2000)
            self.translate_counter = self.translate_thread.finished_counter

        self._emit_pipeline_stopped_if_ready(imgtrans_running=False)

    def detect_finished(self) -> bool:
        if self.imgtrans_proj is None or getattr(self, 'num_pages', 0) == 0:
            return True
        return self.detect_counter == self.num_pages or not cfg_module.enable_detect

    def ocr_finished(self) -> bool:
        if self.imgtrans_proj is None or getattr(self, 'num_pages', 0) == 0:
            return True
        return self.ocr_counter == self.num_pages or not cfg_module.enable_ocr

    def translate_finished(self) -> bool:
        if self.imgtrans_proj is None \
            or getattr(self, 'num_pages', 0) == 0 \
            or not cfg_module.enable_ocr \
            or not cfg_module.enable_translate:
            return True
        if self.parallel_trans:
            # 检查翻译计数器是否达到需要处理的页面数
            return self.translate_thread.finished_counter >= self.num_pages
        return self.translate_counter == self.num_pages or not cfg_module.enable_translate

    def inpaint_finished(self) -> bool:
        if self.imgtrans_proj is None or getattr(self, 'num_pages', 0) == 0 or not cfg_module.enable_inpaint:
            return True
        return self.inpaint_counter == self.num_pages or not cfg_module.enable_inpaint

    def run(self):
        try:
            if self.job is not None:
                self.job()
        except LLMUserActionRequiredError as e:
            _show_llm_user_action_required_dialog(
                e,
                self.tr('Image translation failed.'),
                'ImageTranslationFailed',
            )
            self.requestStop()
            self._emit_pipeline_stopped_if_ready(imgtrans_running=False)
        except LLMRequestStopped:
            LOGGER.info('Image translation task stopped by user.')
            self._emit_pipeline_stopped_if_ready(imgtrans_running=False)
        except Exception as e:
            create_error_dialog(e, self.tr('Image translation failed.'), 'ImageTranslationFailed')
        finally:
            self.job = None

    def recent_finished_index(self, ref_counter: int) -> int:
        if cfg_module.enable_detect:
            ref_counter = min(ref_counter, self.detect_counter)
        if cfg_module.enable_ocr:
            ref_counter = min(ref_counter, self.ocr_counter)
        if cfg_module.enable_inpaint:
            ref_counter = min(ref_counter, self.inpaint_counter)
        if cfg_module.enable_translate:
            if self.parallel_trans:
                ref_counter = min(ref_counter, self.translate_thread.finished_counter)
            else:
                ref_counter = min(ref_counter, self.translate_counter)

        process_idx = ref_counter - 1
        # 将处理索引转换为实际页面索引
        if hasattr(self, 'process_idx_to_page_idx') and process_idx in self.process_idx_to_page_idx:
            return self.process_idx_to_page_idx[process_idx]
        return process_idx


def unload_modules(self, module_names):
    model_deleted = False
    for module in module_names:
        module: BaseModule = getattr(self, module)
        if module is not None:
            module_deleted = module.unload_model()
            model_deleted = model_deleted or module_deleted
    if model_deleted:
        soft_empty_cache()



