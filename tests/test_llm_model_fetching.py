"""Tests for LLM model fetching and dropdown auto-population with manual entry fallback."""

import unittest
from unittest.mock import MagicMock, patch

from qtpy.QtWidgets import QApplication

from ballontranslator.utils.project.llm_profiles import (
    LLMProfile,
    get_candidate_model_endpoints,
    _extract_model_ids,
    fetch_remote_models,
)
from ballontranslator.ui.widgets.llm_profile_widgets import (
    ProfileCardWidget,
    ModelFetchThread,
)


class TestCandidateEndpoints(unittest.TestCase):

    def test_openai_candidates(self):
        c = get_candidate_model_endpoints('https://api.openai.com/v1')
        self.assertEqual(c, ['https://api.openai.com/v1/models'])

    def test_deepseek_candidates(self):
        c = get_candidate_model_endpoints('https://api.deepseek.com')
        self.assertIn('https://api.deepseek.com/models', c)
        self.assertIn('https://api.deepseek.com/v1/models', c)

    def test_chat_completions_suffix_stripped(self):
        c = get_candidate_model_endpoints('https://api.example.com/v1/chat/completions')
        self.assertEqual(c, ['https://api.example.com/v1/models'])

    def test_gemini_candidates(self):
        c = get_candidate_model_endpoints('https://generativelanguage.googleapis.com/v1beta/openai/', api_key='secret')
        self.assertIn('https://generativelanguage.googleapis.com/v1beta/openai/models', c)
        self.assertIn('https://generativelanguage.googleapis.com/v1beta/models?key=secret', c)

    def test_ollama_candidates(self):
        c = get_candidate_model_endpoints('http://localhost:11434/v1/')
        self.assertIn('http://localhost:11434/v1/models', c)
        self.assertIn('http://localhost:11434/api/tags', c)


class TestModelExtraction(unittest.TestCase):

    def test_openai_format(self):
        data = {
            'data': [
                {'id': 'gpt-4o', 'object': 'model'},
                {'id': 'gpt-5.5', 'object': 'model'},
            ]
        }
        self.assertEqual(_extract_model_ids(data), ['gpt-4o', 'gpt-5.5'])

    def test_gemini_format_with_filter(self):
        data = {
            'models': [
                {'name': 'models/gemini-2.5-flash', 'supportedGenerationMethods': ['generateContent']},
                {'name': 'models/text-embedding-004', 'supportedGenerationMethods': ['embedContent']},
                {'name': 'models/gemini-pro', 'supportedGenerationMethods': ['generateContent']},
            ]
        }
        self.assertEqual(_extract_model_ids(data), ['gemini-2.5-flash', 'gemini-pro'])

    def test_ollama_tags_format(self):
        data = {
            'models': [
                {'name': 'llama3.1:latest'},
                {'name': 'qwen2.5:latest'},
            ]
        }
        self.assertEqual(_extract_model_ids(data), ['llama3.1:latest', 'qwen2.5:latest'])

    def test_deduplication(self):
        data = {'data': [{'id': 'gpt-4o'}, {'id': 'gpt-4o'}, {'id': 'gpt-4.1'}]}
        self.assertEqual(_extract_model_ids(data), ['gpt-4o', 'gpt-4.1'])


class TestFetchRemoteModels(unittest.TestCase):

    @patch('requests.get')
    def test_fetch_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            'data': [{'id': 'model-alpha'}, {'id': 'model-beta'}]
        }
        mock_get.return_value = mock_resp

        models = fetch_remote_models('https://api.openai.com/v1', 'sk-test')
        self.assertEqual(models, ['model-alpha', 'model-beta'])

    @patch('requests.get')
    def test_fetch_auth_failure(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = 'Invalid API key'
        mock_resp.json.return_value = {'error': {'message': 'Invalid API key'}}
        mock_get.return_value = mock_resp

        with self.assertRaises(RuntimeError) as ctx:
            fetch_remote_models('https://api.openai.com/v1', 'bad-key')
        self.assertIn('Authentication failed', str(ctx.exception))

    @patch('requests.get')
    def test_fetch_fallback_candidate(self, mock_get):
        # First candidate 404, second 200
        resp_404 = MagicMock(status_code=404, text='Not found')
        resp_200 = MagicMock(status_code=200)
        resp_200.json.return_value = {'data': [{'id': 'deepseek-chat'}]}
        mock_get.side_effect = [resp_404, resp_200]

        models = fetch_remote_models('https://api.deepseek.com', 'sk-test')
        self.assertEqual(models, ['deepseek-chat'])


class TestUIModelFetchingAndManualFallback(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([''])

    def test_buttons_and_on_models_fetched(self):
        profile = LLMProfile.from_provider('OpenAI')
        widget = ProfileCardWidget(profile)

        # Verify buttons exist
        self.assertTrue(hasattr(widget, 'fetch_model_btn'))
        self.assertTrue(hasattr(widget, 'fetch_vision_model_btn'))
        self.assertTrue(hasattr(widget, 'fetch_image_model_btn'))
        self.assertTrue(hasattr(widget, 'add_model_btn'))

        # Simulate fetched models update
        widget._on_models_fetched('text', ['gpt-5.6', 'gpt-5.6-turbo'])
        self.assertIn('gpt-5.6', profile.model_options)
        self.assertIn('gpt-5.6-turbo', profile.model_options)
        self.assertEqual(widget.model_combo.itemText(0), 'gpt-5.6')

        # Test manual fallback: startModelEdit, user typing, finishModelEdit
        widget.startModelEdit()
        widget.model_combo.lineEdit().setText('custom-finetuned-model')
        widget.finishModelEdit()

        self.assertEqual(profile.model, 'custom-finetuned-model')
        self.assertIn('custom-finetuned-model', profile.model_options)
        self.assertEqual(widget.model_combo.currentText(), 'custom-finetuned-model')


if __name__ == '__main__':
    unittest.main()
