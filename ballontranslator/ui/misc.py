"""Backward-compatibility shim for ballontranslator.ui.core.misc."""
import sys
from ballontranslator.ui.core import misc as _mod
sys.modules[__name__] = _mod
