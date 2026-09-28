"""Backward-compatibility shim for ballontranslator.ui.dialogs.pipeline_error_dialog."""
import sys
from ballontranslator.ui.dialogs import pipeline_error_dialog as _mod
sys.modules[__name__] = _mod
