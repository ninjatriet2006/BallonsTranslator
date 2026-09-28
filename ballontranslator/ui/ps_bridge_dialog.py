"""Backward-compatibility shim for ballontranslator.ui.dialogs.ps_bridge_dialog."""
import sys
from ballontranslator.ui.dialogs import ps_bridge_dialog as _mod
sys.modules[__name__] = _mod
