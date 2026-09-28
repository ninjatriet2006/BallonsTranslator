"""Backward-compatibility shim for ballontranslator.utils.system.logger."""
import sys
from ballontranslator.utils.system import logger as _mod
sys.modules[__name__] = _mod
