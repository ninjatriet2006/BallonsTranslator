"""Backward-compatibility shim for ballontranslator.ui.canvas.canvas."""
import sys
from ballontranslator.ui.canvas import canvas as _mod
sys.modules[__name__] = _mod
