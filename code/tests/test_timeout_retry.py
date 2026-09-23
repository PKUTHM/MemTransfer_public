"""Offline regression checks for unlimited network-timeout retries."""

import os
import sys
import unittest
from unittest import mock

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
sys.path.insert(0, CODE_DIR)

import agent  # noqa: E402


class _Response:
    status_code = 200
    text = ""

    def json(self):
        return {
            "choices": [{
                "message": {"content": "Move_Forward 1m"},
                "finish_reason": "stop",
            }],
            "usage": {},
        }


class _ErrorResponse:
    status_code = 500
    text = "server error"


class TimeoutRetryTests(unittest.TestCase):
    def test_model_timeout_does_not_consume_api_attempt(self):
        post = mock.Mock(side_effect=[
            requests.exceptions.ReadTimeout("first"),
            requests.exceptions.ConnectTimeout("second"),
            _Response(),
        ])
        agent._KEY_COOLDOWN.clear()
        settings = {
            "EXPERIMENT": None,
            "API_KEYS": ["test-key"],
            "API_URL": "https://example.invalid/chat",
            "API_PROVIDER": "openai",
            "MODEL_NAME": "test-model",
            "MODEL_EXTRA_BODY": {},
            "MAX_API_ATTEMPTS": 1,
            "STEP_WALL_CLOCK": 60,
            "CONNECT_TIMEOUT": 1,
            "API_TIMEOUT": 1,
            "RETRY_INTERVAL": 0,
        }
        with mock.patch.multiple(agent, **settings), \
                mock.patch.object(agent, "build_system_prompt", return_value="system"), \
                mock.patch.object(agent, "build_user_content",
                                  return_value=[{"type": "text", "text": "user"}]), \
                mock.patch.object(agent.requests, "post", post), \
                mock.patch.object(agent.time, "sleep"):
            result = agent.call_internvl(
                0, "mission", (0, 0), "", None, [], None, None, "test")

        self.assertEqual(result[-1], "ok")
        self.assertEqual(post.call_count, 3)

    def test_local_server_timeout_does_not_consume_retry(self):
        post = mock.Mock(side_effect=[
            requests.exceptions.ReadTimeout("first"),
            requests.exceptions.ReadTimeout("second"),
            _Response(),
        ])
        with mock.patch.object(agent.requests, "post", post), \
                mock.patch.object(agent.time, "sleep"):
            response = agent._post_with_retry(
                "http://127.0.0.1:1", {}, timeout=1, retries=1)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(post.call_count, 3)

    def test_model_non_timeout_after_timeout_still_consumes_attempt(self):
        post = mock.Mock(side_effect=[
            requests.exceptions.ReadTimeout("first"),
            _ErrorResponse(),
        ])
        agent._KEY_COOLDOWN.clear()
        settings = {
            "EXPERIMENT": None,
            "API_KEYS": ["test-key"],
            "API_URL": "https://example.invalid/chat",
            "API_PROVIDER": "openai",
            "MODEL_NAME": "test-model",
            "MODEL_EXTRA_BODY": {},
            "MAX_API_ATTEMPTS": 1,
            "STEP_WALL_CLOCK": 60,
            "CONNECT_TIMEOUT": 1,
            "API_TIMEOUT": 1,
            "RETRY_INTERVAL": 0,
        }
        with mock.patch.multiple(agent, **settings), \
                mock.patch.object(agent, "build_system_prompt", return_value="system"), \
                mock.patch.object(agent, "build_user_content",
                                  return_value=[{"type": "text", "text": "user"}]), \
                mock.patch.object(agent.requests, "post", post), \
                mock.patch.object(agent.time, "sleep"):
            result = agent.call_internvl(
                0, "mission", (0, 0), "", None, [], None, None, "test")

        self.assertEqual(result[-1], "technical")
        self.assertEqual(post.call_count, 2)

    def test_local_connection_error_keeps_configured_limit(self):
        post = mock.Mock(side_effect=requests.exceptions.ConnectionError("down"))
        with mock.patch.object(agent.requests, "post", post), \
                mock.patch.object(agent.time, "sleep"):
            with self.assertRaises(requests.exceptions.ConnectionError):
                agent._post_with_retry(
                    "http://127.0.0.1:1", {}, timeout=1, retries=3)

        self.assertEqual(post.call_count, 3)


if __name__ == "__main__":
    unittest.main()
