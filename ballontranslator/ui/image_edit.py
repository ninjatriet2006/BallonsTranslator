"""Backward-compatibility shim for ballontranslator.ui.canvas.image_edit."""
import sys
from ballontranslator.ui.canvas import image_edit as _mod
sys.modules[__name__] = _mod
