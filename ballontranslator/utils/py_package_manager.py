"""Backward-compatibility shim for ballontranslator.utils.env.py_package_manager."""
import sys
from ballontranslator.utils.env import py_package_manager as _mod
sys.modules[__name__] = _mod
