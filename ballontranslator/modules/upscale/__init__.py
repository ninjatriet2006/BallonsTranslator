from .base import UPSCALERS, UpscalerBase, register_upscaler

GET_VALID_UPSCALERS = lambda: list(UPSCALERS.module_dict.keys())
