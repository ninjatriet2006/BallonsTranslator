"""Backward-compatibility shim for ballontranslator.ui.threads.update_thread."""
import sys
from ballontranslator.ui.threads import update_thread as _mod
sys.modules[__name__] = _mod
