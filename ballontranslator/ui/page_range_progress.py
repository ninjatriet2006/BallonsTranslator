"""Backward-compatibility shim for ballontranslator.ui.widgets.page_range_progress."""
import sys
from ballontranslator.ui.widgets import page_range_progress as _mod
sys.modules[__name__] = _mod
