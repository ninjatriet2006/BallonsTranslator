"""Backward-compatibility shim for ballontranslator.utils.fonts.font_registry."""
import sys
from ballontranslator.utils.fonts import font_registry as _mod
sys.modules[__name__] = _mod
