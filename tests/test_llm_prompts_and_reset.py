import unittest
from qtpy.QtWidgets import QApplication
from qtpy.QtCore import Qt

from ballontranslator.utils.llm_profiles import (
    DEFAULT_DETECTOR_BOX_PROMPT,
    DEFAULT_DETECTOR_OCR_PROMPT,
    DEFAULT_INPAINT_PROMPT,
    DEFAULT_OCR_PROMPT,
    DEFAULT_TRANSLATION_PROMPT,
    LLMProfile,
)
from ballontranslator.ui.module_parse_widgets import ParamWidget, ParamEditor
from ballontranslator.ui.llm_profile_widgets import ProfileCardWidget


class TestLLMPromptsAndReset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def test_default_prompts_content(self):
        # Ensure default prompts are informative and contain key manga/comic scanlation instructions
        self.assertIn("dialogue", DEFAULT_TRANSLATION_PROMPT.lower())
        self.assertIn("speech bubbles", DEFAULT_TRANSLATION_PROMPT.lower())
        self.assertIn("verbatim", DEFAULT_OCR_PROMPT.lower())
        self.assertIn("furigana", DEFAULT_OCR_PROMPT.lower())
        self.assertIn("flat background", DEFAULT_INPAINT_PROMPT.lower())
        self.assertIn("inpainting rules", DEFAULT_INPAINT_PROMPT.lower())
        self.assertIn("speech balloons", DEFAULT_DETECTOR_BOX_PROMPT.lower())
        self.assertIn("sfx", DEFAULT_DETECTOR_BOX_PROMPT.lower())
        self.assertIn("verbatim", DEFAULT_DETECTOR_OCR_PROMPT.lower())
        self.assertIn("ruby", DEFAULT_DETECTOR_OCR_PROMPT.lower())

    def test_llm_profile_default_values(self):
        profile = LLMProfile(id="test", name="Test")
        self.assertEqual(profile.prompt, DEFAULT_TRANSLATION_PROMPT)
        self.assertEqual(profile.vision_prompt, DEFAULT_OCR_PROMPT)
        self.assertEqual(profile.detector_box_prompt, DEFAULT_DETECTOR_BOX_PROMPT)
        self.assertEqual(profile.detector_ocr_prompt, DEFAULT_DETECTOR_OCR_PROMPT)
        self.assertEqual(profile.image_prompt, DEFAULT_INPAINT_PROMPT)

    def test_param_widget_reset_to_default_button(self):
        from qtpy.QtWidgets import QToolButton
        params = {
            'custom_prompt': {
                'type': 'editor',
                'value': 'User Modified Text',
                'display_name': 'Prompt',
                'label_above': True,
                'default_value': 'Default Value Here',
            }
        }
        widget = ParamWidget(params)
        editor = widget.param_widgets['custom_prompt']
        self.assertEqual(editor.text(), 'User Modified Text')

        # Find the reset button
        reset_btn = None
        for btn in widget.findChildren(QToolButton):
            if btn.objectName() == 'ParamResetDefaultBtn':
                reset_btn = btn
                break

        self.assertIsNotNone(reset_btn, "Reset button should be created when default_value is provided")
        # Click reset button
        reset_btn.click()
        self.assertEqual(editor.text(), 'Default Value Here')

    def test_profile_card_prompts_reset(self):
        profile = LLMProfile(
            id="test_card",
            name="Test Card",
            prompt="Modified Translation",
            vision_prompt="Modified OCR",
            detector_box_prompt="Modified Box",
            detector_ocr_prompt="Modified Transcribe",
            image_prompt="Modified Inpaint",
        )
        card = ProfileCardWidget(profile)
        card.setExpanded(True)

        # Check that editors have the modified values
        prompt_editor = card.details.param_widgets['prompt']
        vision_editor = card.details.param_widgets['vision_prompt']
        box_editor = card.details.param_widgets['detector_box_prompt']
        det_ocr_editor = card.details.param_widgets['detector_ocr_prompt']
        image_editor = card.details.param_widgets['image_prompt']

        self.assertEqual(prompt_editor.text(), "Modified Translation")
        self.assertEqual(vision_editor.text(), "Modified OCR")
        self.assertEqual(box_editor.text(), "Modified Box")
        self.assertEqual(det_ocr_editor.text(), "Modified Transcribe")
        self.assertEqual(image_editor.text(), "Modified Inpaint")

        # Find reset buttons in card
        from qtpy.QtWidgets import QToolButton
        reset_buttons = [btn for btn in card.findChildren(QToolButton) if btn.objectName() == 'ParamResetDefaultBtn']
        self.assertEqual(len(reset_buttons), 5, "There should be 5 reset buttons (Text, Vision, DET Box, DET OCR, Image)")

        # Trigger each reset button
        for btn in reset_buttons:
            btn.click()

        # Check that editors and profile are restored to default
        self.assertEqual(prompt_editor.text(), DEFAULT_TRANSLATION_PROMPT)
        self.assertEqual(vision_editor.text(), DEFAULT_OCR_PROMPT)
        self.assertEqual(box_editor.text(), DEFAULT_DETECTOR_BOX_PROMPT)
        self.assertEqual(det_ocr_editor.text(), DEFAULT_DETECTOR_OCR_PROMPT)
        self.assertEqual(image_editor.text(), DEFAULT_INPAINT_PROMPT)

        self.assertEqual(card.profile.prompt, DEFAULT_TRANSLATION_PROMPT)
        self.assertEqual(card.profile.vision_prompt, DEFAULT_OCR_PROMPT)
        self.assertEqual(card.profile.detector_box_prompt, DEFAULT_DETECTOR_BOX_PROMPT)
        self.assertEqual(card.profile.detector_ocr_prompt, DEFAULT_DETECTOR_OCR_PROMPT)
        self.assertEqual(card.profile.image_prompt, DEFAULT_INPAINT_PROMPT)


if __name__ == '__main__':
    unittest.main()
