"""Backward-compatibility shim for ballontranslator.utils.fonts.fontformat."""
import sys
from ballontranslator.utils.fonts import fontformat as _mod
sys.modules[__name__] = _mod
