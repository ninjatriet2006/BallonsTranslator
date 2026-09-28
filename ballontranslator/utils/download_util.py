"""Backward-compatibility shim for ballontranslator.utils.system.download_util."""
import sys
from ballontranslator.utils.system import download_util as _mod
sys.modules[__name__] = _mod
