"""Backward-compatibility shim for ballontranslator.ui.core.mainwindowbars."""
import sys
from ballontranslator.ui.core import mainwindowbars as _mod
sys.modules[__name__] = _mod
