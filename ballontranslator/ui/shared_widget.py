"""Backward-compatibility shim for ballontranslator.ui.widgets.shared_widget."""
import sys
from ballontranslator.ui.widgets import shared_widget as _mod
sys.modules[__name__] = _mod
