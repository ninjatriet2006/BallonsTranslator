"""Backward-compatibility shim for ballontranslator.utils.project.rgba."""
import sys
from ballontranslator.utils.project import rgba as _mod
sys.modules[__name__] = _mod
