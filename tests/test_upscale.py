import os
import tempfile
import unittest
from PIL import Image

from qtpy.QtWidgets import QApplication

from ballontranslator.modules import GET_VALID_UPSCALERS, UPSCALERS
from ballontranslator.modules.upscale.base import UpscalerBase
from ballontranslator.utils.project.config import ModuleConfig
from ballontranslator.ui.dialogs.run_pipeline_dialog import RunPipelineDialog


class DummyProj:
    def __init__(self, directory, pages):
        self.directory = directory
        self.pages = {p: [] for p in pages}

    def get_page_progress(self, page):
        return 0


class TestUpscaleAndStandardization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([''])

    def test_upscalers_registered(self):
        valid = GET_VALID_UPSCALERS()
        self.assertIn('RealESRGAN', valid)
        self.assertIn('Waifu2x', valid)

        re_spec = UPSCALERS.get_spec('RealESRGAN')
        self.assertIsNotNone(re_spec)
        self.assertIn('scale', re_spec.params)
        self.assertIn('denoise', re_spec.params)
        self.assertEqual(re_spec.params['scale']['value'], 1)

        w2x_spec = UPSCALERS.get_spec('Waifu2x')
        self.assertIsNotNone(w2x_spec)
        self.assertIn('scale', w2x_spec.params)
        self.assertIn('denoise', w2x_spec.params)
        self.assertEqual(w2x_spec.params['scale']['value'], 1)

    def test_module_config_upscaler_fields(self):
        cfg = ModuleConfig()
        self.assertEqual(cfg.upscaler, 'None')
        self.assertTrue(cfg.upscale_denoise)
        self.assertFalse(cfg.enable_upscale)
        self.assertFalse(cfg.standardize_dimensions)
        self.assertEqual(cfg.standardize_target_width, 0)
        self.assertEqual(cfg.standardize_target_height, 0)

        # Test saving params includes upscaler_params
        saving = cfg.get_saving_params()
        self.assertIn('upscaler_params', saving)

    def test_scan_project_page_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create 3 pages with 800x1200 and 1 page with 600x900
            for i in range(3):
                img = Image.new('RGB', (800, 1200), color=(255, 255, 255))
                img.save(os.path.join(tmp_dir, f'page_{i}.png'))
            odd_img = Image.new('RGB', (600, 900), color=(255, 255, 255))
            odd_img.save(os.path.join(tmp_dir, 'page_odd.png'))

            proj = DummyProj(tmp_dir, ['page_0.png', 'page_1.png', 'page_2.png', 'page_odd.png'])
            dialog = RunPipelineDialog(project=proj)

            mode_w, mode_h, count, total = dialog._scan_project_page_dimensions()
            self.assertEqual((mode_w, mode_h), (800, 1200))
            self.assertEqual(count, 3)
            self.assertEqual(total, 4)
            dialog.close()

    def test_stage_enabled_config(self):
        cfg = ModuleConfig()
        self.assertFalse(cfg.stage_enabled(4))
        cfg.set_stage_enabled(4, True)
        self.assertTrue(cfg.stage_enabled(4))
        self.assertTrue(cfg.enable_upscale)
        cfg.set_stage_enabled(4, False)
        self.assertFalse(cfg.stage_enabled(4))
        self.assertFalse(cfg.enable_upscale)

    def test_run_pipeline_dialog_upscale_stage_integration(self):
        dialog = RunPipelineDialog()
        try:
            self.assertEqual(len(dialog.module_buttons), 5)
            self.assertEqual(len(dialog.module_activators), 5)
            upscale_activator = dialog.module_activators[4]
            self.assertEqual(upscale_activator.module_type, 'upscaler')
            self.assertEqual(dialog.module_buttons[4].property('stageIndex'), 4)
            self.assertIn(4, dialog.settings_sections)
            self.assertTrue(hasattr(dialog, 'upscale_denoise_cb'))
            self.assertTrue(hasattr(dialog, 'upscale_scale_selector'))
        finally:
            dialog.close()

    def test_upscale_bypass_when_scale_1_and_denoise_off(self):
        import numpy as np
        from ballontranslator.modules.upscale.base import UpscalerBase

        class MockUpscaler(UpscalerBase):
            def _upscale(self, img, scale=1, denoise=True):
                raise RuntimeError("Should not be called")

        mock_inst = MockUpscaler()
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Scale 1 and Denoise False should bypass and return same image instance
        res = mock_inst.upscale(dummy_img, scale=1, denoise=False)
        self.assertIs(res, dummy_img)


if __name__ == '__main__':
    unittest.main()
