"""Backward-compatibility shim for ballontranslator.ui.canvas.drawing_commands."""
import sys
from ballontranslator.ui.canvas import drawing_commands as _mod
sys.modules[__name__] = _mod
