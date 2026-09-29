import os
import sys
from typing import Dict, List
import cv2
import numpy as np

from ..base import UpscalerBase, register_upscaler


@register_upscaler('RealESRGAN')
class RealESRGANUpscaler(UpscalerBase):
    """Real-ESRGAN NCNN Vulkan super-resolution module for anime/manga.

    Example:
        >>> issubclass(RealESRGANUpscaler, UpscalerBase)
        True
    """

    params: Dict = {
        'scale': {
            'type': 'selector',
            'options': [1, 2, 4],
            'value': 1,
            'display_name': 'Scale Ratio',
        },
        'model': {
            'type': 'selector',
            'options': ['realesr-animevideov3', 'realesrgan-x4plus-anime', 'realesrgan-x4plus'],
            'value': 'realesr-animevideov3',
            'display_name': 'Model Name',
        },
        'denoise': {
            'type': 'selector',
            'options': ['On', 'Off'],
            'value': 'On',
            'display_name': 'Denoise',
        },
        'tile_size': {
            'type': 'selector',
            'options': [0, 200, 400, 600],
            'value': 0,
            'display_name': 'Tile Size (0 for auto)',
        },
    }

    if sys.platform == 'win32':
        download_file_list = [{
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-v0.2.5.0-windows.zip',
            'files': ['realesrgan-ncnn-vulkan.exe'],
            'save_dir': 'data/models/upscale/realesrgan',
            'archived_files': 'realesrgan-ncnn-vulkan-v0.2.5.0-windows.zip',
        }]
    else:
        download_file_list = [{
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-v0.2.5.0-ubuntu.zip',
            'files': ['realesrgan-ncnn-vulkan'],
            'save_dir': 'data/models/upscale/realesrgan',
            'archived_files': 'realesrgan-ncnn-vulkan-v0.2.5.0-ubuntu.zip',
        }]

    def _get_bin_path(self) -> str:
        exe = 'realesrgan-ncnn-vulkan.exe' if sys.platform == 'win32' else 'realesrgan-ncnn-vulkan'
        return os.path.abspath(os.path.join('data', 'models', 'upscale', 'realesrgan', exe))

    def _get_model_dir(self) -> str:
        return os.path.abspath(os.path.join('data', 'models', 'upscale', 'realesrgan', 'models'))

    def _upscale(
        self,
        img: np.ndarray,
        scale: int = 1,
        denoise: bool = True,
    ) -> np.ndarray:
        bin_path = self._get_bin_path()
        model_dir = self._get_model_dir()
        model_name = self.params.get('model', {}).get('value', 'realesr-animevideov3')
        tile_size = self.params.get('tile_size', {}).get('value', 0)

        # RealESRGAN supports native scales: 2, 3, 4
        # If user requests scale 1 (enhance only), upscale 2x and resize back
        target_scale = scale if scale in (2, 3, 4) else 2
        cmd = ['-n', model_name, '-s', str(target_scale)]
        if os.path.exists(model_dir):
            cmd.extend(['-m', model_dir])
        if tile_size > 0:
            cmd.extend(['-t', str(tile_size)])

        h, w = img.shape[:2]
        result = self._run_ncnn_binary(bin_path, cmd, img)
        if scale == 1:
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_AREA)
        return result
