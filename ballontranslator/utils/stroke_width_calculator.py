"""Backward-compatibility shim for ballontranslator.utils.text.stroke_width_calculator."""
import sys
from ballontranslator.utils.text import stroke_width_calculator as _mod
sys.modules[__name__] = _mod
