"""Backward-compatibility shim for ballontranslator.ui.widgets.module_parse_widgets."""
import sys
from ballontranslator.ui.widgets import module_parse_widgets as _mod
sys.modules[__name__] = _mod
