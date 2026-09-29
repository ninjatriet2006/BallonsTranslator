"""Unit tests for UI Profile presets and protocol selection dropdown."""

import unittest
from qtpy.QtWidgets import QApplication

from ballontranslator.utils.project.config import pcfg
from ballontranslator.utils.project.llm_profiles import (
    LLMProfile,
    PROTOCOL_OPENAI_COMPLETION,
    PROTOCOL_ANTHROPIC_MESSAGES,
    PROVIDER_DEFAULTS,
)
from ballontranslator.ui.widgets.llm_profile_widgets import (
    LLMProfilesWidget,
    ProfileCardWidget,
)


class TestLLMProfileUIPresetAndProtocol(unittest.TestCase):
    """Test suite for UI Profile presets and protocol selection."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_new_profile_from_preset(self):
        widget = LLMProfilesWidget()
        initial_count = len(pcfg.module.llm_profiles)

        # Create profile from Agnes AI preset
        widget.newProfile("Agnes AI")
        self.assertEqual(len(pcfg.module.llm_profiles), initial_count + 1)
        created = pcfg.module.llm_profiles[-1]
        self.assertFalse(created.built_in)
        self.assertEqual(created.name, "Agnes AI (New)")
        self.assertEqual(created.base_url, "https://vyceai.com/v1")
        self.assertEqual(created.model, "agnes-3.0-flash")
        self.assertEqual(created.image_model, "agnes-image-2.0-flash")

        # Create profile from Anthropic preset
        widget.newProfile("Anthropic")
        self.assertEqual(len(pcfg.module.llm_profiles), initial_count + 2)
        claude_profile = pcfg.module.llm_profiles[-1]
        self.assertEqual(claude_profile.protocol, PROTOCOL_ANTHROPIC_MESSAGES)
        self.assertEqual(claude_profile.base_url, "https://api.anthropic.com/v1")

    def test_editor_protocol_combo_sync(self):
        profile = LLMProfile(
            id="test-proto",
            name="Test Proto",
            protocol=PROTOCOL_ANTHROPIC_MESSAGES,
        )
        card = ProfileCardWidget(profile)
        protocol_widget = card.details.param_widgets.get("protocol")
        self.assertIsNotNone(protocol_widget)
        self.assertEqual(protocol_widget.currentText(), PROTOCOL_ANTHROPIC_MESSAGES)

        # Test changing protocol via detail event
        card.on_detail_edited("protocol", {"content": PROTOCOL_OPENAI_COMPLETION})
        self.assertEqual(profile.protocol, PROTOCOL_OPENAI_COMPLETION)


if __name__ == "__main__":
    unittest.main()
