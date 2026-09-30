import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import numpy as np

import ballontranslator.utils.project.shared as shared
from ballontranslator.utils.system.download_util import check_local_file, download_and_check_files
from ballontranslator.modules.base import BaseModule
from ballontranslator.modules.inpaint import INPAINTERS
from ballontranslator.modules import GET_VALID_INPAINTERS
from ballontranslator.modules.inpaint.offline.inpaint_default import PatchmatchInpainter, Flux2Klein
from ballontranslator.modules.prepare_local_files import _missing_module_requirements, _spec_from_module


class TestInpaintAutoDownloadAndSafety(unittest.TestCase):

    def test_check_local_file_handles_uninitialized_cache(self):
        """Verify check_local_file does not throw TypeError when shared.cache_data is None."""
        orig_cache = shared.cache_data
        temp_path = "tests/test_temp_check_file.tmp"
        with open(temp_path, "wb") as f:
            f.write(b"test_content_for_hash")
        try:
            shared.cache_data = None
            file_exists, valid_hash, sha_calc = check_local_file(
                temp_path,
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                cache_hash=True
            )
            self.assertTrue(file_exists)
            self.assertIsNotNone(shared.cache_data)
            self.assertIsInstance(shared.cache_data, dict)
            self.assertIn(temp_path, shared.cache_data)
        finally:
            shared.cache_data = orig_cache
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_base_module_load_model_calls_ensure_module_files(self):
        """Verify BaseModule.load_model automatically ensures files if download_file_list is present."""
        class DummyModuleWithFiles(BaseModule):
            name = 'dummy_module'
            download_file_list = [{'url': 'http://example.com/test.ckpt', 'files': 'data/models/test.ckpt'}]
            def _load_model(self):
                self.loaded = True

        dummy = DummyModuleWithFiles()
        dummy.loaded = False

        with patch('ballontranslator.modules.prepare_local_files.ensure_module_files', return_value=True) as mock_ensure:
            dummy.load_model()
            mock_ensure.assert_called_once_with(dummy)
            self.assertTrue(dummy.loaded)

    def test_patchmatch_safe_import_and_error(self):
        """Verify PatchmatchInpainter initializes without crashing and provides clear error if lib is absent."""
        inp = PatchmatchInpainter()
        self.assertIsNotNone(inp)
        if sys.platform == 'linux':
            # On Linux without libpatchmatch.so, inpaint should raise clear FileNotFoundError
            img = np.zeros((32, 32, 3), dtype=np.uint8)
            mask = np.zeros((32, 32), dtype=np.uint8)
            try:
                inp.inpaint(img, mask)
            except Exception as e:
                self.assertIsInstance(e, FileNotFoundError)
                self.assertIn("data/libs/libpatchmatch.so", str(e))

    def test_flux2_klein_dependency_resolution(self):
        """Verify Flux2Klein does not strictly pin transformers==4.57.6 which breaks with modern transformers."""
        spec = INPAINTERS.get_spec('flux2-klein')
        self.assertIsNotNone(spec)
        self.assertIn('transformers>=4.40.0', spec.dependencies)
        missing = _missing_module_requirements(spec)
        # transformers should not be in missing requirements if transformers is installed
        try:
            import transformers
            has_transformers = True
        except ImportError:
            has_transformers = False

        if has_transformers:
            missing_req_names = [m.requirement for m in missing]
            for req in missing_req_names:
                self.assertNotIn('transformers', req)

    def test_get_valid_inpainters_includes_supported(self):
        """Verify GET_VALID_INPAINTERS returns available inpainters."""
        valid = GET_VALID_INPAINTERS()
        self.assertIn('opencv-tela', valid)
        self.assertIn('aot', valid)
        self.assertIn('lama_mpe', valid)
        self.assertIn('lama_large_512px', valid)
        self.assertIn('LLMInpaint', valid)


if __name__ == '__main__':
    unittest.main()
