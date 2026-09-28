"""Backward-compatibility shim for ballontranslator.utils.text.text_layout."""
import sys
from ballontranslator.utils.text import text_layout as _mod
sys.modules[__name__] = _mod
