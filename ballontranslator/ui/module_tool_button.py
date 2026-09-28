"""Backward-compatibility shim for ballontranslator.ui.widgets.module_tool_button."""
import sys
from ballontranslator.ui.widgets import module_tool_button as _mod
sys.modules[__name__] = _mod
