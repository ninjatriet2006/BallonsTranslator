"""Backward-compatibility shim for ballontranslator.ui.widgets.global_search_widget."""
import sys
from ballontranslator.ui.widgets import global_search_widget as _mod
sys.modules[__name__] = _mod
