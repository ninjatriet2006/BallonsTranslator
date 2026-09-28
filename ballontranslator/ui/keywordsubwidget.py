"""Backward-compatibility shim for ballontranslator.ui.widgets.keywordsubwidget."""
import sys
from ballontranslator.ui.widgets import keywordsubwidget as _mod
sys.modules[__name__] = _mod
