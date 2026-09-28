"""Backward-compatibility shim for ballontranslator.utils.system.imgproc_utils."""
import sys
from ballontranslator.utils.system import imgproc_utils as _mod
sys.modules[__name__] = _mod
