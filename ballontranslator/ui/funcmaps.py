"""Backward-compatibility shim for ballontranslator.ui.core.funcmaps."""
import sys
from ballontranslator.ui.core import funcmaps as _mod
sys.modules[__name__] = _mod
