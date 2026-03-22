from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.append(str(Path(__file__).resolve().parents[1]))

from schema.llm_outputs import (  # noqa: E402
    CoderOutput,
    EvaluatorAnalysisOutput,
    PlannerOutput,
)


def test_planner_output_parses_valid_subtasks() -> None:
    """
    test_planner_output_parses_valid_subtasks を実行する。
    
    Args:
        なし。
    """
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
    """
    test_planner_output_rejects_empty_subtasks を実行する。
    
    Args:
        なし。
    """
    with pytest.raises(ValidationError):
        PlannerOutput(subtasks=[])


def test_planner_output_rejects_invalid_agent() -> None:
    """
    test_planner_output_rejects_invalid_agent を実行する。
    
    Args:
        なし。
    """
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
    """
    test_coder_output_accepts_multi_file_map を実行する。
    
    Args:
        なし。
    """
    payload = {
        "files": {
            "main.py": "print('hello')\n",
            "utils/helper.py": "def f():\n    return 1\n",
        }
    }
    result = CoderOutput(**payload)
    assert "main.py" in result.files
    assert "utils/helper.py" in result.files


def test_evaluator_analysis_output_accepts_valid_payload() -> None:
    """
    test_evaluator_analysis_output_accepts_valid_payload を実行する。

    Args:
        なし。
    """
    payload = {
        "likely_cause": "Variable was not defined.",
        "suggested_fixes": ["Define it before use"],
        "can_self_fix": True,
        "needs_research": False,
    }
    result = EvaluatorAnalysisOutput(**payload)
    assert result.can_self_fix is True
    assert result.suggested_fixes == ["Define it before use"]
