"""Backward-compatibility shim for ballontranslator.utils.system.io_utils."""
import sys
from ballontranslator.utils.system import io_utils as _mod
sys.modules[__name__] = _mod
