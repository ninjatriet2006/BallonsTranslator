"""Backward-compatibility shim for ballontranslator.ui.threads.packageinstall_thread."""
import sys
from ballontranslator.ui.threads import packageinstall_thread as _mod
sys.modules[__name__] = _mod
