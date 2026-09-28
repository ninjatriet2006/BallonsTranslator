import unittest
from qtpy.QtWidgets import QApplication
from qtpy.QtCore import Qt

from ballontranslator.utils.project.config import pcfg, ModuleConfig
from ballontranslator.utils.project.llm_profiles import LLMProfile, _normalize_profile_thinking
from ballontranslator.modules.translators.llm.trans_llm import LLMTranslator
from ballontranslator.modules.translators.llm_translation_contract import render_user_prompt
from ballontranslator.ui.configpanel import ConfigTextLabel, ConfigPanel

_app = QApplication.instance() or QApplication([])

class UncappedThreadsAndSettingsLayoutTests(unittest.TestCase):

    def test_module_config_uncapped_threads(self):
        cfg = ModuleConfig()
        cfg.llm_translate_threads = 64
        cfg.llm_ocr_threads = 32
        cfg.llm_inpaint_threads = 24
        cfg.local_detect_threads = 16
        cfg.local_ocr_threads = 16
        cfg.local_translate_threads = 16
        cfg.local_inpaint_threads = 16

        cfg.__post_init__()

        self.assertEqual(cfg.llm_translate_threads, 64)
        self.assertEqual(cfg.llm_ocr_threads, 32)
        self.assertEqual(cfg.llm_inpaint_threads, 24)
        self.assertEqual(cfg.local_detect_threads, 16)
        self.assertEqual(cfg.local_ocr_threads, 16)
        self.assertEqual(cfg.local_translate_threads, 16)
        self.assertEqual(cfg.local_inpaint_threads, 16)

    def test_llm_profile_uncapped_concurrency(self):
        profile = LLMProfile(id="test", name="test", concurrency=128)
        sanitized = _normalize_profile_thinking(profile)
        self.assertEqual(sanitized.concurrency, 128)

    def test_config_text_label_pixel_size(self):
        label = ConfigTextLabel("Test", 13)
        self.assertEqual(label.font().pixelSize(), 13)

    def test_config_panel_thread_spinbox_ranges(self):
        panel = ConfigPanel()
        self.assertEqual(panel.llm_translate_threads_spin.maximum(), 999)
        self.assertEqual(panel.llm_ocr_threads_spin.maximum(), 999)
        self.assertEqual(panel.llm_inpaint_threads_spin.maximum(), 999)
        self.assertEqual(panel.local_detect_threads_spin.maximum(), 999)
        self.assertEqual(panel.local_ocr_threads_spin.maximum(), 999)
        self.assertEqual(panel.local_translate_threads_spin.maximum(), 999)
        self.assertEqual(panel.local_inpaint_threads_spin.maximum(), 999)
        panel.deleteLater()

    def test_llm_translator_auto_language_support(self):
        translator = LLMTranslator('Auto', 'English', raise_unsupported_lang=True)
        self.assertIn('Auto', translator.supported_src_list)
        self.assertEqual(translator.supported_src_list[0], 'Auto')
        self.assertNotIn('Auto', translator.supported_tgt_list)

        prompt = render_user_prompt(('Sample text',), 'Auto', 'Vietnamese')
        self.assertIn('automatically detect the source language', prompt)
        self.assertIn('Vietnamese', prompt)

if __name__ == '__main__':
    unittest.main()
