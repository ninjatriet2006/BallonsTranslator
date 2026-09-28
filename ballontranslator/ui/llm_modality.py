"""Backward-compatibility shim for ballontranslator.ui.core.llm_modality."""
import sys
from ballontranslator.ui.core import llm_modality as _mod
sys.modules[__name__] = _mod
