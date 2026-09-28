"""Backward-compatibility shim for ballontranslator.utils.project.config."""
import sys
from ballontranslator.utils.project import config as _mod
sys.modules[__name__] = _mod
