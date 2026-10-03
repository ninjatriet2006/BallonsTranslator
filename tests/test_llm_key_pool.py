import json
import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from ballontranslator.utils.project.key_pool import (
    KeyPool,
    clean_key_file_path,
    count_profile_keys,
    get_key_pool,
    is_key_file,
    load_keys_from_file,
    resolve_active_api_key,
)
from ballontranslator.utils.project.llm_profiles import LLMProfile, store_api_key


class TestLLMKeyPool(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_is_key_file_detection(self):
        # Direct keys should NOT be detected as files
        self.assertFalse(is_key_file("sk-proj-1234567890abcdef"))
        self.assertFalse(is_key_file("AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz"))
        self.assertFalse(is_key_file(""))
        self.assertFalse(is_key_file(None))
        self.assertFalse(is_key_file("plain_secret_string_without_path_or_ext"))

        # Real file on disk must be recognized
        txt_path = os.path.join(self.temp_dir.name, "keys.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("key1\nkey2\n")
        self.assertTrue(is_key_file(txt_path))

        # Path-like strings with key extensions should be recognized
        self.assertTrue(is_key_file("my_keys.txt"))
        self.assertTrue(is_key_file("configs/keys.json"))
        self.assertTrue(is_key_file("~/keys.keys"))

    def test_load_keys_from_txt_file(self):
        txt_path = os.path.join(self.temp_dir.name, "test_keys.txt")
        content = """
        # This is a comment
        key_one
        // Another comment style
        
        key_two
        key_three,
        "key_four"
        key_one # duplicate should be deduplicated
        """
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(content)

        keys = load_keys_from_file(txt_path)
        self.assertEqual(keys, ["key_one", "key_two", "key_three", "key_four"])

    def test_load_keys_from_json_file(self):
        # Array format
        json_path1 = os.path.join(self.temp_dir.name, "keys_array.json")
        with open(json_path1, "w", encoding="utf-8") as f:
            json.dump(["json_key1", "json_key2", "json_key1"], f)

        keys1 = load_keys_from_file(json_path1)
        self.assertEqual(keys1, ["json_key1", "json_key2"])

        # Dict format
        json_path2 = os.path.join(self.temp_dir.name, "keys_dict.json")
        with open(json_path2, "w", encoding="utf-8") as f:
            json.dump({"keys": ["dict_k1", "dict_k2"]}, f)

        keys2 = load_keys_from_file(json_path2)
        self.assertEqual(keys2, ["dict_k1", "dict_k2"])

    def test_key_pool_round_robin_and_cooldown(self):
        txt_path = os.path.join(self.temp_dir.name, "pool_keys.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("keyA\nkeyB\nkeyC\n")

        pool = KeyPool(txt_path)
        self.assertEqual(pool.count(), 3)

        # Round-robin rotation
        self.assertEqual(pool.get_next_key(), "keyA")
        self.assertEqual(pool.get_next_key(), "keyB")
        self.assertEqual(pool.get_next_key(), "keyC")
        self.assertEqual(pool.get_next_key(), "keyA")

        # Mark keyB as rate limited
        pool.mark_rate_limited("keyB", cooldown_seconds=10.0)
        self.assertTrue(pool.is_rate_limited("keyB"))
        self.assertFalse(pool.is_rate_limited("keyA"))

        # Next key should skip keyB
        self.assertEqual(pool.get_next_key(), "keyC")
        self.assertEqual(pool.get_next_key(), "keyA")
        self.assertEqual(pool.get_next_key(), "keyC")

    def test_key_pool_hot_reload(self):
        txt_path = os.path.join(self.temp_dir.name, "reload_keys.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("key1\nkey2\n")

        pool = KeyPool(txt_path)
        self.assertEqual(pool.get_all_keys(), ["key1", "key2"])

        # Update file on disk
        time.sleep(0.05)  # ensure mtime advances
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("key1\nkey2\nkey3\n")

        pool.reload(force=True)
        self.assertEqual(pool.get_all_keys(), ["key1", "key2", "key3"])

    def test_profile_resolution_direct_vs_file(self):
        # 1. Direct key
        profile_direct = LLMProfile(name="DirectTest", model="gpt-4o")
        store_api_key(profile_direct, "sk-direct-12345")
        self.assertEqual(resolve_active_api_key(profile_direct), "sk-direct-12345")
        self.assertEqual(count_profile_keys(profile_direct), 1)

        # 2. Keys file
        txt_path = os.path.join(self.temp_dir.name, "profile_keys.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("rot_key_1\nrot_key_2\n")

        profile_file = LLMProfile(name="FileTest", model="gpt-4o")
        store_api_key(profile_file, txt_path)

        self.assertEqual(count_profile_keys(profile_file), 2)
        # Rotates across calls
        first = resolve_active_api_key(profile_file)
        second = resolve_active_api_key(profile_file)
        third = resolve_active_api_key(profile_file)

        self.assertEqual(first, "rot_key_1")
        self.assertEqual(second, "rot_key_2")
        self.assertEqual(third, "rot_key_1")


if __name__ == "__main__":
    unittest.main()
