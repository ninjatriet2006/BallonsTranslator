"""Backward-compatibility shim for ballontranslator.utils.system.lock."""
import sys
from ballontranslator.utils.system import lock as _mod
sys.modules[__name__] = _mod
