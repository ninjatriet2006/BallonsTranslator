import os
import sys
import tempfile
import subprocess
from typing import Dict, List, Optional
import numpy as np
import cv2

from ballontranslator.utils.project.registry import Registry
from ballontranslator.utils.system.logger import logger as LOGGER
from ..base import BaseModule

UPSCALERS = Registry('upscalers')
register_upscaler = UPSCALERS.register_module


class UpscalerBase(BaseModule):
    """Base class for comic upscalers and super-resolution enhancers.

    Example:
        >>> issubclass(UpscalerBase, BaseModule)
        True
    """

    params: Dict = {
        'scale': {
            'type': 'selector',
            'options': [1, 2, 4],
            'value': 1,
            'display_name': 'Scale Ratio',
        },
        'denoise': {
            'type': 'selector',
            'options': ['On', 'Off'],
            'value': 'On',
            'display_name': 'Denoise',
        },
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.name = ''
        for key in UPSCALERS.module_dict:
            if UPSCALERS.module_dict[key] == self.__class__:
                self.name = key
                break

    def upscale(
        self,
        img: np.ndarray,
        scale: int = 1,
        denoise: bool = True,
    ) -> np.ndarray:
        if scale <= 1 and not denoise:
            return img
        if not self.all_model_loaded():
            self.load_model()
        return self._upscale(img, scale=scale, denoise=denoise)

    def _upscale(
        self,
        img: np.ndarray,
        scale: int = 1,
        denoise: bool = True,
    ) -> np.ndarray:
        raise NotImplementedError

    def _run_ncnn_binary(
        self,
        bin_path: str,
        cmd_args: List[str],
        img: np.ndarray,
    ) -> np.ndarray:
        """Run an NCNN Vulkan binary safely using temporary I/O files."""
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f'Upscaler binary not found: {bin_path}')

        if sys.platform != 'win32' and not os.access(bin_path, os.X_OK):
            try:
                os.chmod(bin_path, 0o755)
            except Exception as e:
                LOGGER.warning(f'Could not chmod +x on {bin_path}: {e}')

        with tempfile.TemporaryDirectory() as tmp_dir:
            in_file = os.path.join(tmp_dir, 'input.png')
            out_file = os.path.join(tmp_dir, 'output.png')
            cv2.imwrite(in_file, img)

            full_cmd = [bin_path, '-i', in_file, '-o', out_file] + cmd_args
            LOGGER.info(f'Running upscaler command: {" ".join(full_cmd)}')

            process = subprocess.run(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if process.returncode != 0:
                raise RuntimeError(
                    f'Upscaler failed with exit code {process.returncode}: {process.stderr}'
                )

            if not os.path.exists(out_file):
                raise FileNotFoundError(
                    f'Upscaler produced no output image at {out_file}'
                )

            res = cv2.imread(out_file, cv2.IMREAD_UNCHANGED)
            if res is None:
                raise ValueError('Failed to read upscaled output image')
            return res
