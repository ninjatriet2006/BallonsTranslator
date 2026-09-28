"""Backward-compatibility shim for ballontranslator.ui.widgets.llm_profile_widgets."""
import sys
from ballontranslator.ui.widgets import llm_profile_widgets as _mod
sys.modules[__name__] = _mod
