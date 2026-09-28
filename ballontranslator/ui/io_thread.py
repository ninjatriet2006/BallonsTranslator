"""Backward-compatibility shim for ballontranslator.ui.threads.io_thread."""
import sys
from ballontranslator.ui.threads import io_thread as _mod
sys.modules[__name__] = _mod
