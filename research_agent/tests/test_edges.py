from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from graph.edges import should_continue  # noqa: E402


def _build_state(**overrides: Any) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "status": "pending",
        "current_step": "init",
        "retry_count": 0,
    }
    state.update(overrides)
    return state


def test_should_continue_returns_coder_for_self_fixable_failure() -> None:
    result = should_continue(_build_state(status="coding", current_step="coder"))
    assert result == "coder"


def test_should_continue_returns_researcher_for_needs_research_failure() -> None:
    result = should_continue(
        _build_state(status="researching", current_step="researcher")
    )
    assert result == "researcher"


def test_should_continue_returns_done_for_terminal_status() -> None:
    result = should_continue(_build_state(status="completed", current_step="done"))
    assert result == "done"
