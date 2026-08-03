"""Tests for reasoning suppression helpers in activity_utils."""

import unittest
from unittest.mock import MagicMock

from activity_utils import (
    create_completion_skip_thinking,
    strip_reasoning,
)


class TestStripReasoning(unittest.TestCase):
    def test_closed_pair_stripped(self):
        self.assertEqual(strip_reasoning("<think>a</think>answer"), "answer")

    def test_tag_variants_case_insensitive(self):
        self.assertEqual(strip_reasoning("<THINKING>a</THINKING>b"), "b")
        self.assertEqual(strip_reasoning("<reasoning>x</reasoning>y"), "y")
        self.assertEqual(
            strip_reasoning("<REASONING_SCRATCHPAD>x</REASONING_SCRATCHPAD>y"),
            "y",
        )

    def test_template_preopened_orphan_close(self):
        # Chat template pre-opens the think block: output has only a closing
        # tag, everything before it is reasoning.
        self.assertEqual(
            strip_reasoning("step 1... step 2...</think>\nfinal"), "final"
        )

    def test_multiple_orphan_closes_keeps_after_last(self):
        self.assertEqual(
            strip_reasoning("a</think>b</reasoning>final"), "final"
        )

    def test_salvage_all_think(self):
        # Model spent every token inside the block; return de-tagged trace
        # rather than an empty string.
        self.assertEqual(
            strip_reasoning("<think>the answer is 42</think>"),
            "the answer is 42",
        )

    def test_salvage_unterminated(self):
        self.assertEqual(
            strip_reasoning("<think>truncated mid-reason"),
            "truncated mid-reason",
        )

    def test_plain_and_empty_passthrough(self):
        self.assertEqual(strip_reasoning("plain"), "plain")
        self.assertEqual(strip_reasoning(""), "")
        self.assertIsNone(strip_reasoning(None))


class TestCreateCompletionSkipThinking(unittest.TestCase):
    def test_injects_chat_template_kwargs(self):
        client = MagicMock()
        create_completion_skip_thinking(
            client, model="m", messages=[], max_tokens=5
        )
        _, kwargs = client.chat.completions.create.call_args
        self.assertEqual(
            kwargs["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.assertEqual(kwargs["model"], "m")

    def test_retries_without_kwargs_on_rejection(self):
        # Hosted providers (Groq, Mistral, Gemini) reject the param with a
        # 4xx naming it; the helper retries the call without extra_body.
        client = MagicMock()
        rejection = Exception(
            "property 'chat_template_kwargs' is unsupported"
        )
        rejection.status_code = 400
        ok = MagicMock()
        client.chat.completions.create.side_effect = [rejection, ok]

        result = create_completion_skip_thinking(
            client, model="m", messages=[]
        )

        self.assertIs(result, ok)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        _, retry_kwargs = client.chat.completions.create.call_args
        self.assertNotIn("extra_body", retry_kwargs)

    def test_unrelated_error_propagates(self):
        client = MagicMock()
        boom = Exception("connection reset")
        client.chat.completions.create.side_effect = boom
        with self.assertRaises(Exception):
            create_completion_skip_thinking(client, model="m", messages=[])
        self.assertEqual(client.chat.completions.create.call_count, 1)


if __name__ == "__main__":
    unittest.main()
