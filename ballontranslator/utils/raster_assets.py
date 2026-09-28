"""Backward-compatibility shim for ballontranslator.utils.system.raster_assets."""
import sys
from ballontranslator.utils.system import raster_assets as _mod
sys.modules[__name__] = _mod
