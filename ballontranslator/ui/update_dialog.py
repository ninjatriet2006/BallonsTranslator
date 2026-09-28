"""Backward-compatibility shim for ballontranslator.ui.dialogs.update_dialog."""
import sys
from ballontranslator.ui.dialogs import update_dialog as _mod
sys.modules[__name__] = _mod
