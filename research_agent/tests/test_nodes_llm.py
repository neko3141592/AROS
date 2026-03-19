from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import graph.nodes.coder as coder_mod  # noqa: E402
import graph.nodes.planner as planner_mod  # noqa: E402
import graph.nodes.researcher as researcher_mod  # noqa: E402
from schema.task import Task  # noqa: E402
from tools.file_io import create_run_paths  # noqa: E402


def _build_state(task: Task, run_paths: Any = None) -> Dict[str, Any]:
    return {
        "task": task,
        "status": "starting",
        "messages": [],
        "current_step": "init",
        "research_context": "",
        "generated_code": None,
        "generated_files": None,
        "execution_logs": None,
        "retry_count": 0,
        "result": None,
        "run_paths": run_paths,
        "error": None,
    }


def test_planner_creates_subtasks_from_json(monkeypatch: pytest.MonkeyPatch) -> None:
    task = Task(
        title="Planner Test",
        description="Split work into subtasks",
        constraints=["local only"],
        subtasks=[],
    )
    state = _build_state(task)

    planner_json = json.dumps(
        [
            {
                "title": "Research papers",
                "description": "Collect 3 relevant papers",
                "assigned_agent": "researcher",
                "status": "pending",
            },
            {
                "title": "Write code",
                "description": "Implement baseline",
                "assigned_agent": "coder",
                "status": "pending",
            },
        ]
    )
    monkeypatch.setattr(planner_mod, "generate_text", lambda **_kwargs: planner_json)

    result = planner_mod.planner_node(state)

    assert result["status"] == "planning"
    assert result["current_step"] == "researcher"
    assert len(result["task"].subtasks) == 2
    assert result["task"].subtasks[0].title == "Research papers"
    assert result["error"] is None


def test_planner_handles_invalid_json_as_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    task = Task(
        title="Planner Error Test",
        description="Return invalid payload",
        constraints=["json only"],
        subtasks=[],
    )
    state = _build_state(task)

    monkeypatch.setattr(planner_mod, "generate_text", lambda **_kwargs: "not json")

    result = planner_mod.planner_node(state)

    assert result["status"] == "failed"
    assert "parse error" in result["error"]


def test_researcher_puts_summary_into_state(monkeypatch: pytest.MonkeyPatch) -> None:
    task = Task(
        title="Researcher Test",
        description="Summarize relevant findings",
        constraints=[],
        subtasks=[],
    )
    state = _build_state(task)

    monkeypatch.setattr(researcher_mod, "build_arxiv_query", lambda **_kwargs: "query")
    monkeypatch.setattr(
        researcher_mod, "fetch_arxiv_raw", lambda **_kwargs: ["raw_result"]
    )
    monkeypatch.setattr(
        researcher_mod, "parse_arxiv_response", lambda _raw: ["parsed_paper"]
    )
    monkeypatch.setattr(
        researcher_mod, "format_papers_for_llm", lambda _papers: "paper context"
    )
    monkeypatch.setattr(
        researcher_mod,
        "generate_text",
        lambda **_kwargs: "## 実装コンテキスト\n- 手順A\n- 手順B",
    )

    result = researcher_mod.researcher_node(state)

    assert result["status"] == "researching"
    assert result["current_step"] == "coder"
    assert "## 実装コンテキスト" in result["research_context"]
    assert result["error"] is None


def test_coder_saves_workspace_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    task = Task(
        title="Coder Test",
        description="Generate multiple files",
        constraints=["local"],
        subtasks=[],
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    state = _build_state(task, run_paths=run_paths)
    state["research_context"] = "Use a small baseline."

    coder_json = json.dumps(
        {
            "files": {
                "main.py": "def main():\n    print('ok')\n\nif __name__ == '__main__':\n    main()\n",
                "analysis/run.py": "print('analysis')\n",
            }
        }
    )
    monkeypatch.setattr(coder_mod, "generate_text", lambda **_kwargs: coder_json)

    result = coder_mod.coder_node(state)

    assert result["status"] == "coding"
    assert "generated_files" in result
    assert "analysis/run.py" in result["generated_files"]
    assert run_paths.code_path.read_text(encoding="utf-8") == result["generated_code"]
    assert (
        run_paths.workspace_dir / "analysis" / "run.py"
    ).read_text(encoding="utf-8") == "print('analysis')\n"
