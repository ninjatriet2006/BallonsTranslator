"""Backward-compatibility shim for ballontranslator.utils.project.registry."""
import sys
from ballontranslator.utils.project import registry as _mod
sys.modules[__name__] = _mod
