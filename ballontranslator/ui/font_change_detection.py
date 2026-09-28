"""Backward-compatibility shim for ballontranslator.ui.core.font_change_detection."""
import sys
from ballontranslator.ui.core import font_change_detection as _mod
sys.modules[__name__] = _mod
