"""Backward-compatibility shim for ballontranslator.ui.dialogs.run_pipeline_dialog."""
import sys
from ballontranslator.ui.dialogs import run_pipeline_dialog as _mod
sys.modules[__name__] = _mod
