"""Backward-compatibility shim for ballontranslator.utils.fonts.font_detect."""
import sys
from ballontranslator.utils.fonts import font_detect as _mod
sys.modules[__name__] = _mod
