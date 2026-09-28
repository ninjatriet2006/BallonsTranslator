"""Backward-compatibility shim for ballontranslator.utils.project.llm_profiles."""
import sys
from ballontranslator.utils.project import llm_profiles as _mod
sys.modules[__name__] = _mod
