"""Backward-compatibility shim for ballontranslator.ui.core.adaptive_wrap_layout."""
import sys
from ballontranslator.ui.core import adaptive_wrap_layout as _mod
sys.modules[__name__] = _mod
