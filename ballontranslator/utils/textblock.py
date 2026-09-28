"""Backward-compatibility shim for ballontranslator.utils.text.textblock."""
import sys
from ballontranslator.utils.text import textblock as _mod
sys.modules[__name__] = _mod
