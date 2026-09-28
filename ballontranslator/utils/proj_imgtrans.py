"""Backward-compatibility shim for ballontranslator.utils.project.proj_imgtrans."""
import sys
from ballontranslator.utils.project import proj_imgtrans as _mod
sys.modules[__name__] = _mod
