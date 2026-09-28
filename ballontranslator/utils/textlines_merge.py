"""Backward-compatibility shim for ballontranslator.utils.text.textlines_merge."""
import sys
from ballontranslator.utils.text import textlines_merge as _mod
sys.modules[__name__] = _mod
