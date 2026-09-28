"""Backward-compatibility shim for ballontranslator.utils.env.network_mirrors."""
import sys
from ballontranslator.utils.env import network_mirrors as _mod
sys.modules[__name__] = _mod
