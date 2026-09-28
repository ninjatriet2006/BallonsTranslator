"""Backward-compatibility shim for ballontranslator.utils.project.merger."""
import sys
from ballontranslator.utils.project import merger as _mod
sys.modules[__name__] = _mod
