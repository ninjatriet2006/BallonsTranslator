"""Backward-compatibility shim for ballontranslator.utils.project.structures."""
import sys
from ballontranslator.utils.project import structures as _mod
sys.modules[__name__] = _mod
