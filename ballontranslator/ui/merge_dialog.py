"""Backward-compatibility shim for ballontranslator.ui.dialogs.merge_dialog."""
import sys
from ballontranslator.ui.dialogs import merge_dialog as _mod
sys.modules[__name__] = _mod
