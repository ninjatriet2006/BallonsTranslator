"""Backward-compatibility shim for ballontranslator.utils.system.secret_store."""
import sys
from ballontranslator.utils.system import secret_store as _mod
sys.modules[__name__] = _mod
