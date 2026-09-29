"""Unit tests for LLM protocols, Anthropic messages, and protocol selection."""

import unittest
from unittest.mock import MagicMock, patch
from ballontranslator.modules.llm.protocols import (
    create_llm_protocol,
    OpenAICompletionProtocol,
    OpenAIResponseProtocol,
    AnthropicMessagesProtocol,
)
from ballontranslator.utils.project.llm_profiles import (
    LLMProfile,
    PROTOCOL_OPENAI_COMPLETION,
    PROTOCOL_OPENAI_RESPONSE,
    PROTOCOL_ANTHROPIC_MESSAGES,
    profile_from_config,
    profile_to_dict,
)
from ballontranslator.modules.llm_chat import LLMChatRequester, LLMChatResult


class TestLLMProtocolsAndAnthropic(unittest.TestCase):
    """Test suite for protocol selection and Anthropic adapter."""

    def test_protocol_factory(self):
        profile = LLMProfile(name="Test")
        p_comp = create_llm_protocol(profile, "openai_completion")
        self.assertIsInstance(p_comp, OpenAICompletionProtocol)

        p_resp = create_llm_protocol(profile, "openai_response")
        self.assertIsInstance(p_resp, OpenAIResponseProtocol)

        p_anth = create_llm_protocol(profile, "anthropic_messages")
        self.assertIsInstance(p_anth, AnthropicMessagesProtocol)

    def test_protocol_field_serialization(self):
        profile = LLMProfile(
            id="test-claude",
            name="Claude Test",
            protocol=PROTOCOL_ANTHROPIC_MESSAGES,
        )
        data = profile_to_dict(profile)
        self.assertEqual(data["protocol"], PROTOCOL_ANTHROPIC_MESSAGES)

        loaded = profile_from_config(data)
        self.assertEqual(loaded.protocol, PROTOCOL_ANTHROPIC_MESSAGES)

    def test_anthropic_convert_messages(self):
        profile = LLMProfile(name="Claude")
        adapter = AnthropicMessagesProtocol(profile)
        messages = [
            {"role": "system", "content": "You are a translator."},
            {"role": "user", "content": "Hello world"},
        ]
        system_text, converted = adapter._convert_messages(messages)
        self.assertEqual(system_text, "You are a translator.")
        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0]["role"], "user")
        self.assertEqual(converted[0]["content"], "Hello world")

    def test_anthropic_build_payload(self):
        profile = LLMProfile(name="Claude", model="claude-3-7-sonnet-latest", max_tokens=2048)
        adapter = AnthropicMessagesProtocol(profile)
        payload = adapter.build_payload(
            [{"role": "user", "content": "Test"}],
            system_prompt="System instructions",
        )
        self.assertEqual(payload["model"], "claude-3-7-sonnet-latest")
        self.assertEqual(payload["max_tokens"], 2048)
        self.assertEqual(payload["system"][0]["text"], "System instructions")

    @patch("httpx.Client.post")
    def test_anthropic_send_chat_completion_mock(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [
                {"type": "text", "text": "Translated text"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 12, "output_tokens": 8},
        }
        mock_post.return_value = mock_response

        profile = LLMProfile(
            name="Claude",
            api_key="sk-ant-test",
            protocol=PROTOCOL_ANTHROPIC_MESSAGES,
        )
        requester = LLMChatRequester()
        result = requester.request_chat_completion(
            profile,
            {"model": "claude-3-7-sonnet-latest", "messages": [{"role": "user", "content": "Hi"}]},
        )
        self.assertEqual(result.content, "Translated text")
        self.assertEqual(result.finish_reason, "end_turn")


if __name__ == "__main__":
    unittest.main()
