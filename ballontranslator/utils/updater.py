"""Backward-compatibility shim for ballontranslator.utils.system.updater."""
import sys
from ballontranslator.utils.system import updater as _mod
sys.modules[__name__] = _mod
