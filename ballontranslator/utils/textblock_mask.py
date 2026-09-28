"""Backward-compatibility shim for ballontranslator.utils.text.textblock_mask."""
import sys
from ballontranslator.utils.text import textblock_mask as _mod
sys.modules[__name__] = _mod
