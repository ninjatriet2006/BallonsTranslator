"""Backward-compatibility shim for ballontranslator.utils.text.text_alpha_mask."""
import sys
from ballontranslator.utils.text import text_alpha_mask as _mod
sys.modules[__name__] = _mod
