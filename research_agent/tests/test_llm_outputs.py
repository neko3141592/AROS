from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.append(str(Path(__file__).resolve().parents[1]))

from schema.llm_outputs import CoderOutput, PlannerOutput  # noqa: E402


def test_planner_output_parses_valid_subtasks() -> None:
    payload = {
        "subtasks": [
            {
                "title": "paper search",
                "description": "Find related papers",
                "assigned_agent": "researcher",
                "status": "pending",
            }
        ]
    }
    result = PlannerOutput(**payload)
    assert len(result.subtasks) == 1
    assert result.subtasks[0].assigned_agent == "researcher"


def test_planner_output_rejects_empty_subtasks() -> None:
    with pytest.raises(ValidationError):
        PlannerOutput(subtasks=[])


def test_planner_output_rejects_invalid_agent() -> None:
    with pytest.raises(ValidationError):
        PlannerOutput(
            subtasks=[
                {
                    "title": "invalid",
                    "description": "invalid",
                    "assigned_agent": "planner",
                    "status": "pending",
                }
            ]
        )


def test_coder_output_accepts_multi_file_map() -> None:
    payload = {
        "files": {
            "main.py": "print('hello')\n",
            "utils/helper.py": "def f():\n    return 1\n",
        }
    }
    result = CoderOutput(**payload)
    assert "main.py" in result.files
    assert "utils/helper.py" in result.files
