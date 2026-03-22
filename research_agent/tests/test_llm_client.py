from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import tools.llm_client as llm_client  # noqa: E402


def test_generate_text_returns_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_generate_text_returns_content を実行する。
    
    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
    """
    def fake_completion(**_: object):
        """
        fake_completion を実行する。
        
        Args:
            **_: 未使用の可変引数。
        """
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(llm_client, "completion", fake_completion)
    result = llm_client.generate_text("sys", "user", model="gpt-test")
    assert result == "ok"


def test_generate_text_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_generate_text_retries_then_succeeds を実行する。
    
    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
    """
    calls = {"count": 0}

    def fake_completion(**_: object):
        """
        fake_completion を実行する。
        
        Args:
            **_: 未使用の可変引数。
        """
        calls["count"] += 1
        if calls["count"] < 2:
            raise RuntimeError("temporary error")
        return {"choices": [{"message": {"content": "recovered"}}]}

    monkeypatch.setattr(llm_client, "completion", fake_completion)
    monkeypatch.setattr(llm_client.time, "sleep", lambda _x: None)

    result = llm_client.generate_text(
        "sys",
        "user",
        model="gpt-test",
        max_retries=2,
        base_backoff_seconds=0.0,
    )
    assert result == "recovered"
    assert calls["count"] == 2


def test_generate_text_raises_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_generate_text_raises_after_max_retries を実行する。
    
    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
    """
    def fake_completion(**_: object):
        """
        fake_completion を実行する。
        
        Args:
            **_: 未使用の可変引数。
        """
        raise RuntimeError("always fail")

    monkeypatch.setattr(llm_client, "completion", fake_completion)
    monkeypatch.setattr(llm_client.time, "sleep", lambda _x: None)

    with pytest.raises(llm_client.LLMClientError):
        llm_client.generate_text(
            "sys",
            "user",
            model="gpt-test",
            max_retries=1,
            base_backoff_seconds=0.0,
        )


def test_generate_text_raises_on_invalid_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_generate_text_raises_on_invalid_response を実行する。
    
    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
    """
    def fake_completion(**_: object):
        """
        fake_completion を実行する。
        
        Args:
            **_: 未使用の可変引数。
        """
        return {"unexpected": "shape"}

    monkeypatch.setattr(llm_client, "completion", fake_completion)

    with pytest.raises(llm_client.LLMClientError, match="Invalid response format"):
        llm_client.generate_text("sys", "user", model="gpt-test")


def test_generate_with_tools_normalizes_object_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_generate_with_tools_normalizes_object_tool_calls を実行する。

    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
    """

    class FakeFunction:
        def __init__(self, name: str, arguments: str) -> None:
            self.name = name
            self.arguments = arguments

    class FakeToolCall:
        def __init__(self) -> None:
            self.id = "call_1"
            self.type = "function"
            self.function = FakeFunction(
                name="read_file",
                arguments=json.dumps({"file_path": "main.py"}),
            )

    def fake_completion(**_: object):
        """
        fake_completion を実行する。

        Args:
            **_: 未使用の可変引数。
        """
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [FakeToolCall()],
                    }
                }
            ]
        }

    monkeypatch.setattr(llm_client, "completion", fake_completion)

    result = llm_client.generate_with_tools(
        "sys",
        "user",
        model="gpt-test",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                    },
                },
            }
        ],
    )

    assert result["content"] == ""
    assert result["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"file_path": "main.py"}',
            },
        }
    ]
