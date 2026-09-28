"""Backward-compatibility shim for ballontranslator.ui.core.font_refresh."""
import sys
from ballontranslator.ui.core import font_refresh as _mod
sys.modules[__name__] = _mod
