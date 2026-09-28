"""Backward-compatibility shim for ballontranslator.utils.project.stage_auditor."""
import sys
from ballontranslator.utils.project import stage_auditor as _mod
sys.modules[__name__] = _mod
