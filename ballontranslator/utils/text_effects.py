"""Backward-compatibility shim for ballontranslator.utils.text.text_effects."""
import sys
from ballontranslator.utils.text import text_effects as _mod
sys.modules[__name__] = _mod
