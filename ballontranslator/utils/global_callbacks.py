"""Backward-compatibility shim for ballontranslator.utils.project.global_callbacks."""
import sys
from ballontranslator.utils.project import global_callbacks as _mod
sys.modules[__name__] = _mod
