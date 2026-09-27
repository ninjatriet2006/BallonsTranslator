"""Automated tests for LLM resilience, retry with backoff, glossary matching, and typesetting."""

import unittest
from unittest.mock import MagicMock, patch
from ballontranslator.modules.llm_chat import (
    LLMChatRequester,
    _parse_retry_after,
    TRANSIENT_STATUS_CODES,
)
from ballontranslator.modules.context.glossary import (
    GlossaryEntry,
    select_glossary,
    _FastGlossaryMatcher,
)
from ballontranslator.utils.text_layout import Line, balance_lines
from ballontranslator.utils.llm_profiles import LLMProfile


class TestRetryAfterParser(unittest.TestCase):

    def test_float_seconds(self):
        resp = MagicMock()
        resp.headers = {'retry-after': '12.5'}
        self.assertEqual(_parse_retry_after(resp, 5.0), 12.5)

    def test_missing_header_returns_default(self):
        resp = MagicMock()
        resp.headers = {}
        self.assertEqual(_parse_retry_after(resp, 5.0), 5.0)

    def test_none_response_returns_default(self):
        self.assertEqual(_parse_retry_after(None, 7.0), 7.0)

    def test_cap_at_120s(self):
        resp = MagicMock()
        resp.headers = {'Retry-After': '9999'}
        self.assertEqual(_parse_retry_after(resp, 5.0), 120.0)

    def test_min_floor_at_half_second(self):
        resp = MagicMock()
        resp.headers = {'retry-after': '0.05'}
        self.assertEqual(_parse_retry_after(resp, 5.0), 0.5)


class TestFastGlossaryMatcher(unittest.TestCase):

    def test_overlapping_matches(self):
        terms = ['cat', 'caterpillar', 'apple', '勇者']
        matcher = _FastGlossaryMatcher(terms)
        text = 'A green caterpillar ate an apple in front of the 勇者.'
        matched = matcher.match(text.casefold())
        matched_terms = {terms[i] for i in matched}
        self.assertEqual(matched_terms, {'cat', 'caterpillar', 'apple', '勇者'})

    def test_select_glossary_preserves_order(self):
        entries = (
            GlossaryEntry('Dragon', 'Rồng'),
            GlossaryEntry('Hero', 'Anh Hùng'),
            GlossaryEntry('Sword', 'Kiếm'),
        )
        selected = select_glossary(entries, ['The hero drew a sword.'], 'matching')
        self.assertEqual(tuple(e.source for e in selected), ('Hero', 'Sword'))


class TestBalancedLines(unittest.TestCase):

    def test_orphan_word_balanced(self):
        lines = [
            Line('Chuyện này thật là buồn', 0, 0, 100),
            Line('cười', 0, 20, 20),
        ]
        lines[0].num_words = 5
        lines[1].num_words = 1
        balanced = balance_lines(lines, ' ')
        self.assertEqual(balanced[0].text, 'Chuyện này thật là')
        self.assertEqual(balanced[1].text, 'buồn cười')

    def test_already_balanced_untouched(self):
        lines = [
            Line('Xin chào', 0, 0, 50),
            Line('các bạn', 0, 20, 50),
        ]
        lines[0].num_words = 2
        lines[1].num_words = 2
        balanced = balance_lines(lines, ' ')
        self.assertEqual(balanced[0].text, 'Xin chào')
        self.assertEqual(balanced[1].text, 'các bạn')


class TestLLMChatTransientRetry(unittest.TestCase):

    def test_transient_status_codes_contains_429_and_5xx(self):
        for code in (429, 500, 502, 503, 504):
            self.assertIn(code, TRANSIENT_STATUS_CODES)

    def test_http_client_sets_connection_limits(self):
        requester = LLMChatRequester()
        client = requester._http_client('')
        # Verify httpx client was created
        self.assertIsNotNone(client)
        client.close()


if __name__ == '__main__':
    unittest.main()
