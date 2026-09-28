"""Backward-compatibility shim for ballontranslator.ui.core.spellcheck."""
import sys
from ballontranslator.ui.core import spellcheck as _mod
sys.modules[__name__] = _mod
