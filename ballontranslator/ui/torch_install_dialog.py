"""Backward-compatibility shim for ballontranslator.ui.dialogs.torch_install_dialog."""
import sys
from ballontranslator.ui.dialogs import torch_install_dialog as _mod
sys.modules[__name__] = _mod
