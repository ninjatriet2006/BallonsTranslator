"""Backward-compatibility shim for ballontranslator.ui.core.icon_rendering."""
import sys
from ballontranslator.ui.core import icon_rendering as _mod
sys.modules[__name__] = _mod
