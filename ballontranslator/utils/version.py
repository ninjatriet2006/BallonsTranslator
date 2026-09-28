"""Backward-compatibility shim for ballontranslator.utils.system.version."""
import sys
from ballontranslator.utils.system import version as _mod
sys.modules[__name__] = _mod
