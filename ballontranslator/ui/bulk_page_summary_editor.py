"""Backward-compatibility shim for ballontranslator.ui.dialogs.bulk_page_summary_editor."""
import sys
from ballontranslator.ui.dialogs import bulk_page_summary_editor as _mod
sys.modules[__name__] = _mod
