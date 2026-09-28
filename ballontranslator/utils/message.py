"""Backward-compatibility shim for ballontranslator.utils.project.message."""
import sys
from ballontranslator.utils.project import message as _mod
sys.modules[__name__] = _mod
