"""Backward-compatibility shim for ballontranslator.utils.text.split_text_region."""
import sys
from ballontranslator.utils.text import split_text_region as _mod
sys.modules[__name__] = _mod
