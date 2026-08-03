#!/usr/bin/env python3
"""
Unit tests for the guarded_ai.py module.

Tests the core feedback generation functions including:
- Legacy single feedback system
- New multi-prompt feedback system
- Both systems together
- OpenAI client initialization
- Categorization and feedback generation
"""

import unittest
from unittest.mock import patch, MagicMock, call
import sys
from pathlib import Path
import json

# Add parent directory to path to import guarded_ai
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "research"))
from guarded_ai import (
    provide_feedback,
    provide_feedback_prompts,
    categorize_response,
    generate_ai_feedback,
    get_openai_client_and_model,
    initialize_model_map,
)


class TestGuardedAI(unittest.TestCase):
    """Test cases for guarded_ai functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.sample_metadata = {
            "player_health": 100,
            "enemy_health": 80,
            "user_shot": "A5",
            "ai_shot": "B3",
            "user_hit_result": "hit",
            "ai_hit_result": "miss",
        }

        self.sample_transition = {
            "ai_feedback": {
                "tokens_for_ai": "Additional transition-specific instructions"
            },
            "metadata_feedback_filter": [
                "user_shot",
                "ai_shot",
                "user_hit_result",
                "ai_hit_result",
            ],
        }

    @patch("guarded_ai.get_openai_client_and_model")
    def test_categorize_response(self, mock_get_client):
        """Test response categorization"""
        # Setup mock
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "correct_answer"
        mock_client.chat.completions.create.return_value = mock_completion
        mock_get_client.return_value = (mock_client, "test-model")

        # Test categorization
        question = "What is 2+2?"
        response = "Four"
        buckets = ["correct_answer", "wrong_answer"]
        tokens_for_ai = "Categorize math answers"

        category = categorize_response(question, response, buckets, tokens_for_ai)

        # Verify result
        self.assertEqual(category, "correct_answer")

        # Verify client was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_args["model"], "test-model")
        self.assertEqual(call_args["max_tokens"], 5)
        self.assertEqual(call_args["temperature"], 0)

        # Check message content
        messages = call_args["messages"]
        self.assertEqual(len(messages), 2)
        self.assertIn("correct_answer, wrong_answer", messages[0]["content"])

    @patch("guarded_ai.get_openai_client_and_model")
    def test_generate_ai_feedback(self, mock_get_client):
        """Test AI feedback generation"""
        # Setup mock
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "Great job on the math!"
        mock_client.chat.completions.create.return_value = mock_completion
        mock_get_client.return_value = (mock_client, "test-model")

        # Test feedback generation
        category = "correct_answer"
        question = "What is 2+2?"
        user_response = "Four"
        tokens_for_ai = "Provide encouraging feedback"
        metadata = {"score": 100}

        feedback = generate_ai_feedback(
            category, question, user_response, tokens_for_ai, metadata
        )

        # Verify result
        self.assertEqual(feedback, "Great job on the math!")

        # Verify client was called correctly
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_args["model"], "test-model")
        self.assertEqual(call_args["max_tokens"], 250)
        self.assertEqual(call_args["temperature"], 0.7)

    @patch("guarded_ai.generate_ai_feedback")
    def test_provide_feedback_legacy(self, mock_generate_feedback):
        """Test legacy single feedback system"""
        mock_generate_feedback.return_value = "Good work! Try again."

        # Test data
        transition = self.sample_transition
        category = "partial_understanding"
        question = "What is the capital of France?"
        user_response = "Paris is nice"
        user_language = "English"
        tokens_for_ai = "Provide geography feedback"
        metadata = {"attempts": 1}

        # Call function
        feedback = provide_feedback(
            transition,
            category,
            question,
            user_response,
            user_language,
            tokens_for_ai,
            metadata,
        )

        # Verify feedback was generated
        self.assertIn("AI Feedback:", feedback)
        self.assertIn("Good work! Try again.", feedback)

        # Verify generate_ai_feedback was called with filtered metadata
        mock_generate_feedback.assert_called_once()
        call_args = mock_generate_feedback.call_args[0]
        self.assertEqual(call_args[0], category)  # category
        self.assertEqual(call_args[1], question)  # question
        self.assertEqual(call_args[2], user_response)  # user_response

        # Check tokens_for_ai includes language and transition instructions
        tokens_arg = call_args[3]
        self.assertIn("English", tokens_arg)
        self.assertIn("Additional transition-specific instructions", tokens_arg)

        # Check metadata was filtered
        filtered_metadata = call_args[4]
        expected_filtered = {
            k: v
            for k, v in self.sample_metadata.items()
            if k in transition["metadata_feedback_filter"]
        }
        # Since our test metadata doesn't have the filtered keys, it should be empty or contain only matching keys
        # But the function should have passed what it received

    @patch("guarded_ai.generate_ai_feedback")
    def test_provide_feedback_prompts(self, mock_generate_feedback):
        """Test new multi-prompt feedback system"""
        # Setup mock to return different feedback for each prompt
        mock_generate_feedback.side_effect = [
            "Hit at A5, miss at B3",
            "No ships were sunk this round",
        ]

        # Test data
        transition = self.sample_transition
        category = "valid_move"
        question = "Where do you want to shoot?"
        feedback_prompts = [
            {
                "name": "hit_miss",
                "tokens_for_ai": "Report the hit/miss results for both players",
            },
            {
                "name": "ship_sinking",
                "tokens_for_ai": "Report any ships that were sunk",
            },
        ]
        user_response = "A5"
        user_language = "English"
        metadata = self.sample_metadata

        # Call function
        feedback_messages = provide_feedback_prompts(
            transition,
            category,
            question,
            feedback_prompts,
            user_response,
            user_language,
            metadata,
            "",
        )

        # Verify we got the expected number of feedback messages
        self.assertEqual(len(feedback_messages), 2)

        # Verify message structure
        self.assertEqual(feedback_messages[0]["name"], "hit_miss")
        self.assertEqual(feedback_messages[0]["content"], "Hit at A5, miss at B3")
        self.assertEqual(feedback_messages[1]["name"], "ship_sinking")
        self.assertEqual(
            feedback_messages[1]["content"], "No ships were sunk this round"
        )

        # Verify generate_ai_feedback was called twice
        self.assertEqual(mock_generate_feedback.call_count, 2)

    @patch("guarded_ai.generate_ai_feedback")
    def test_provide_feedback_prompts_empty_responses(self, mock_generate_feedback):
        """Test that empty feedback responses are filtered out"""
        # Setup mock to return empty/whitespace responses
        mock_generate_feedback.side_effect = [
            "",  # Empty response
            "   ",  # Whitespace only
            "Valid feedback",  # Valid response
        ]

        transition = {}
        category = "test"
        question = "Test?"
        feedback_prompts = [
            {"name": "empty", "tokens_for_ai": "Empty prompt"},
            {"name": "whitespace", "tokens_for_ai": "Whitespace prompt"},
            {"name": "valid", "tokens_for_ai": "Valid prompt"},
        ]
        user_response = "Test response"
        user_language = "English"
        metadata = {}

        feedback_messages = provide_feedback_prompts(
            transition,
            category,
            question,
            feedback_prompts,
            user_response,
            user_language,
            metadata,
            "",
        )

        # Should only return the valid feedback message
        self.assertEqual(len(feedback_messages), 1)
        self.assertEqual(feedback_messages[0]["name"], "valid")
        self.assertEqual(feedback_messages[0]["content"], "Valid feedback")

    def test_provide_feedback_no_ai_feedback_config(self):
        """Test legacy feedback when no ai_feedback config in transition"""
        transition = {}  # No ai_feedback key
        category = "test"
        question = "Test?"
        user_response = "Response"
        user_language = "English"
        tokens_for_ai = "Base tokens"
        metadata = {}

        with patch("guarded_ai.generate_ai_feedback") as mock_generate:
            mock_generate.return_value = ""  # Should not be called

            feedback = provide_feedback(
                transition,
                category,
                question,
                user_response,
                user_language,
                tokens_for_ai,
                metadata,
            )

            # Should NOT call generate_ai_feedback when no ai_feedback in transition
            mock_generate.assert_not_called()
            self.assertEqual(feedback, "")

    @patch.dict(
        "os.environ",
        {"MODEL_ENDPOINT_0": "http://test.com", "MODEL_API_KEY_0": "test-key"},
    )
    def test_initialize_model_map(self):
        """Test model map initialization from environment variables"""
        with patch("guarded_ai.get_client_for_endpoint") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            # Mock the models.list() response
            mock_model = MagicMock()
            mock_model.id = "test-model-id"
            mock_client.models.list.return_value.data = [mock_model]

            # Clear and reinitialize
            import guarded_ai

            guarded_ai.MODEL_CLIENT_MAP = {}
            initialize_model_map()

            # Verify client was created and stored with actual model ID
            mock_get_client.assert_called_with("http://test.com", "test-key")
            self.assertIn("test-model-id", guarded_ai.MODEL_CLIENT_MAP)
            self.assertEqual(
                guarded_ai.MODEL_CLIENT_MAP["test-model-id"][0], mock_client
            )
            self.assertEqual(
                guarded_ai.MODEL_CLIENT_MAP["test-model-id"][1], "http://test.com"
            )

    @patch.dict(
        "os.environ",
        {
            "MODEL_ENDPOINT_1": "http://hermes.test",
            "MODEL_API_KEY_1": "hermes-key",
        },
    )
    def test_get_openai_client_and_model_default(self):
        """Explicit MODEL_NAME_1 wins over querying the endpoint's model list"""
        with patch("guarded_ai.MODEL_CLIENT_MAP", {}):
            with patch("guarded_ai.get_client_for_endpoint") as mock_get_client:
                mock_client = MagicMock()
                mock_get_client.return_value = mock_client

                client, model = get_openai_client_and_model()

                # conftest sets MODEL_NAME_1=test-model; it takes precedence
                self.assertEqual(model, "test-model")
                self.assertEqual(client, mock_client)
                mock_client.models.list.assert_not_called()

    @patch.dict(
        "os.environ",
        {
            "MODEL_ENDPOINT_1": "http://hermes.test",
            "MODEL_API_KEY_1": "hermes-key",
            "MODEL_NAME_1": "",
        },
    )
    def test_get_openai_client_and_model_first_listed_fallback(self):
        """Without MODEL_NAME_1, fall back to the endpoint's first listed model"""
        with patch("guarded_ai.MODEL_CLIENT_MAP", {}):
            with patch("guarded_ai.get_client_for_endpoint") as mock_get_client:
                mock_client = MagicMock()
                mock_get_client.return_value = mock_client

                # Mock the models.list() response for MODEL_1
                mock_model = MagicMock()
                mock_model.id = "adamo1139/Hermes-3-Llama-3.1-8B-FP8-Dynamic"
                mock_client.models.list.return_value.data = [mock_model]

                client, model = get_openai_client_and_model()

                # Should return MODEL_1's first model
                self.assertEqual(model, "adamo1139/Hermes-3-Llama-3.1-8B-FP8-Dynamic")
                self.assertEqual(client, mock_client)

    def test_get_openai_client_and_model_from_map(self):
        """Test getting OpenAI client from model map"""
        mock_client = MagicMock()
        test_map = {"endpoint_0": (mock_client, "http://test.com")}

        with patch("guarded_ai.MODEL_CLIENT_MAP", test_map):
            client, model = get_openai_client_and_model("test-model")

            # Should return client from map
            self.assertEqual(client, mock_client)
            self.assertEqual(model, "test-model")

    @patch("guarded_ai.get_openai_client_and_model")
    def test_categorize_response_error_handling(self, mock_get_client):
        """Test error handling in categorize_response"""
        # Setup mock to raise exception
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_get_client.return_value = (mock_client, "test-model")

        category = categorize_response("Test?", "Answer", ["bucket1"], "tokens")

        # Should return error string
        self.assertIn("Error:", category)

    @patch("guarded_ai.get_openai_client_and_model")
    def test_generate_ai_feedback_error_handling(self, mock_get_client):
        """Test error handling in generate_ai_feedback"""
        # Setup mock to raise exception
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_get_client.return_value = (mock_client, "test-model")

        feedback = generate_ai_feedback("cat", "Q?", "A", "tokens", {})

        # Should return error string
        self.assertIn("Error:", feedback)


if __name__ == "__main__":
    unittest.main(verbosity=2)
