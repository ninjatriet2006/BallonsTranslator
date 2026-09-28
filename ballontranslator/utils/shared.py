"""Backward-compatibility shim for ballontranslator.utils.project.shared."""
import sys
from ballontranslator.utils.project import shared as _mod
sys.modules[__name__] = _mod
