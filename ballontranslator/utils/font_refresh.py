"""Backward-compatibility shim for ballontranslator.utils.fonts.font_refresh."""
import sys
from ballontranslator.utils.fonts import font_refresh as _mod
sys.modules[__name__] = _mod
