"""Backward-compatibility shim for ballontranslator.utils.env.torch_install_helper."""
import sys
from ballontranslator.utils.env import torch_install_helper as _mod
sys.modules[__name__] = _mod
