"""Backward-compatibility shim for ballontranslator.utils.env.package_installer."""
import sys
from ballontranslator.utils.env import package_installer as _mod
sys.modules[__name__] = _mod
