from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.messages import SystemMessage

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.prompt_manager import (  # noqa: E402
    PromptRenderError,
    PromptValidationError,
    build_system_message,
    load_prompt,
    render_prompt,
)


def test_load_prompt_reads_planner_prompt() -> None:
    prompt = load_prompt("system_planner")
    assert prompt.name == "system_planner"
    assert "task_title" in prompt.input_variables


def test_render_prompt_formats_list_constraints() -> None:
    rendered = render_prompt(
        "system_planner",
        {
            "task_title": "Transformer 実装",
            "task_description": "最小実装で動作確認する",
            "task_constraints": ["PyTorchを使う", "10分以内に実行"],
        },
    )
    assert "Transformer 実装" in rendered
    assert "- PyTorchを使う" in rendered
    assert "- 10分以内に実行" in rendered


def test_render_prompt_raises_when_variable_is_missing() -> None:
    with pytest.raises(PromptRenderError):
        render_prompt(
            "system_coder",
            {
                "task_title": "モデル実装",
                "task_description": "推論コードも含める",
                "task_constraints": ["CPUのみ"],
                # research_context をあえて省略
            },
        )


def test_load_prompt_rejects_undefined_placeholder(tmp_path: Path) -> None:
    invalid_prompt = tmp_path / "invalid.yaml"
    invalid_prompt.write_text(
        "\n".join(
            [
                "name: invalid",
                "description: test",
                "input_variables:",
                "  - known",
                "template: \"value={unknown}\"",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(PromptValidationError):
        load_prompt("invalid", prompts_dir=tmp_path)


def test_build_system_message_returns_langchain_message() -> None:
    message = build_system_message(
        "system_coder",
        {
            "task_title": "MNIST分類",
            "task_description": "ローカルで1エポックだけ学習",
            "task_constraints": ["外部APIを呼ばない"],
            "research_context": "CNNの最小構成で十分",
        },
    )
    assert isinstance(message, SystemMessage)
    assert "MNIST分類" in message.content
