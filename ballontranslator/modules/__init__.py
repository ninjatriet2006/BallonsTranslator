from .ocr import OCR, OCRBase
from .textdetector import TEXTDETECTORS, TextDetectorBase
from .translators import TRANSLATORS, BaseTranslator
from .inpaint import INPAINTERS, InpainterBase
from .upscale import UPSCALERS, UpscalerBase
from .base import DEFAULT_DEVICE, GPUINTENSIVE_SET, LOGGER, merge_config_module_params, \
    init_module_registries

def _valid_module_keys(registry):
    keys = []
    for key in registry.module_dict.keys():
        spec = registry.get_spec(key)
        if spec is not None and not spec.available:
            continue
        keys.append(key)
    return keys

GET_VALID_TEXTDETECTORS = lambda : _valid_module_keys(TEXTDETECTORS)
GET_VALID_TRANSLATORS = lambda : _valid_module_keys(TRANSLATORS)
GET_VALID_INPAINTERS = lambda : _valid_module_keys(INPAINTERS)
GET_VALID_OCR = lambda : _valid_module_keys(OCR)
GET_VALID_UPSCALERS = lambda : _valid_module_keys(UPSCALERS)


MODULETYPE_TO_REGISTRIES = {
    'textdetector': TEXTDETECTORS,
    'ocr': OCR,
    'inpainter': INPAINTERS,
    'translator': TRANSLATORS,
    'upscaler': UPSCALERS,
}

# Registers static ModuleSpec entries only; concrete module files stay unimported.
init_module_registries()

# TODO: use manga-image-translator as backend...
