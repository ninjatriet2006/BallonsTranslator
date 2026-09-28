"""Backward-compatibility shim for ballontranslator.ui.canvas.cursor."""
import sys
from ballontranslator.ui.canvas import cursor as _mod
sys.modules[__name__] = _mod
