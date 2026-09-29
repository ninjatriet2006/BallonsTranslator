import os
import sys
from typing import Dict, List
import cv2
import numpy as np

from ..base import UpscalerBase, register_upscaler


@register_upscaler('Waifu2x')
class Waifu2xUpscaler(UpscalerBase):
    """Waifu2x NCNN Vulkan super-resolution and noise reduction for manga.

    Example:
        >>> issubclass(Waifu2xUpscaler, UpscalerBase)
        True
    """

    params: Dict = {
        'scale': {
            'type': 'selector',
            'options': [1, 2, 4],
            'value': 1,
            'display_name': 'Scale Ratio',
        },
        'noise_level': {
            'type': 'selector',
            'options': [-1, 0, 1, 2, 3],
            'value': 1,
            'display_name': 'Noise Reduction Level',
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
            'url': 'https://github.com/nihui/waifu2x-ncnn-vulkan/releases/download/20200818/waifu2x-ncnn-vulkan-20200818-windows.zip',
            'files': ['waifu2x-ncnn-vulkan.exe'],
            'save_dir': 'data/models/upscale/waifu2x',
            'archived_files': 'waifu2x-ncnn-vulkan-20200818-windows.zip',
        }]
    else:
        download_file_list = [{
            'url': 'https://github.com/nihui/waifu2x-ncnn-vulkan/releases/download/20200818/waifu2x-ncnn-vulkan-20200818-ubuntu.zip',
            'files': ['waifu2x-ncnn-vulkan'],
            'save_dir': 'data/models/upscale/waifu2x',
            'archived_files': 'waifu2x-ncnn-vulkan-20200818-ubuntu.zip',
        }]

    def _get_bin_path(self) -> str:
        exe = 'waifu2x-ncnn-vulkan.exe' if sys.platform == 'win32' else 'waifu2x-ncnn-vulkan'
        return os.path.abspath(os.path.join('data', 'models', 'upscale', 'waifu2x', exe))

    def _get_model_dir(self) -> str:
        return os.path.abspath(os.path.join('data', 'models', 'upscale', 'waifu2x', 'models-cunet'))

    def _upscale(
        self,
        img: np.ndarray,
        scale: int = 1,
        denoise: bool = True,
    ) -> np.ndarray:
        bin_path = self._get_bin_path()
        model_dir = self._get_model_dir()
        tile_size = self.params.get('tile_size', {}).get('value', 0)
        noise_cfg = self.params.get('noise_level', {}).get('value', 1)

        # Waifu2x native options:
        # -n: noise level (-1, 0, 1, 2, 3)
        # -s: scale ratio (1, 2)
        if not denoise:
            noise_val = -1
        else:
            noise_val = noise_cfg if noise_cfg in (-1, 0, 1, 2, 3) else 1

        run_scale = 2 if scale >= 2 else 1
        cmd = ['-s', str(run_scale), '-n', str(noise_val)]
        if os.path.exists(model_dir):
            cmd.extend(['-m', model_dir])
        if tile_size > 0:
            cmd.extend(['-t', str(tile_size)])

        result = self._run_ncnn_binary(bin_path, cmd, img)
        if scale == 4:
            # Run second pass for 4x
            cmd_4x = ['-s', '2', '-n', '-1']
            if os.path.exists(model_dir):
                cmd_4x.extend(['-m', model_dir])
            result = self._run_ncnn_binary(bin_path, cmd_4x, result)
        return result
