"""Backward-compatibility shim for ballontranslator.ui.core.menu_style."""
import sys
from ballontranslator.ui.core import menu_style as _mod
sys.modules[__name__] = _mod
