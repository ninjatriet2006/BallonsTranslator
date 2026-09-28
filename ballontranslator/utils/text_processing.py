"""Backward-compatibility shim for ballontranslator.utils.text.text_processing."""
import sys
from ballontranslator.utils.text import text_processing as _mod
sys.modules[__name__] = _mod
