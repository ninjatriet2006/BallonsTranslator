"""Backward-compatibility shim for ballontranslator.utils.env.core_requirements."""

import sys
from ballontranslator.utils.env import core_requirements as _mod

sys.modules[__name__] = _mod
