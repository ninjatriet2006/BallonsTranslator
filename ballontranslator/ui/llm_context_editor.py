"""Backward-compatibility shim for ballontranslator.ui.widgets.llm_context_editor."""
import sys
from ballontranslator.ui.widgets import llm_context_editor as _mod
sys.modules[__name__] = _mod
