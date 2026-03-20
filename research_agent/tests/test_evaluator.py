from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from graph.nodes.evaluator import evaluator_node  # noqa: E402
from schema.task import Task  # noqa: E402
from tools.file_io import create_run_paths, read_execution_log, save_workspace_files  # noqa: E402


def _build_state(task: Task, run_paths: Any, retry_count: int = 0) -> Dict[str, Any]:
    return {
        "task": task,
        "status": "coding",
        "messages": [],
        "current_step": "evaluator",
        "research_context": "",
        "generated_code": None,
        "generated_files": None,
        "execution_logs": None,
        "execution_stdout": None,
        "execution_stderr": None,
        "execution_return_code": None,
        "retry_count": retry_count,
        "result": None,
        "run_paths": run_paths,
        "error": None,
    }


def test_evaluator_marks_completed_on_success(tmp_path: Path) -> None:
    task = Task(
        title="Evaluator Success",
        description="Run simple script",
        constraints=[],
        subtasks=[],
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    save_workspace_files(paths=run_paths, files={"main.py": "print('ok')\n"})
    state = _build_state(task=task, run_paths=run_paths, retry_count=0)

    result = evaluator_node(state)

    assert result["status"] == "completed"
    assert result["current_step"] == "done"
    assert result["retry_count"] == 0
    assert result["execution_return_code"] == 0
    assert result["execution_stdout"].strip() == "ok"
    assert result["result"].success is True
    assert "=== STDOUT ===" in result["execution_logs"]
    assert "ok" in read_execution_log(run_paths)


def test_evaluator_returns_to_coder_on_failure(tmp_path: Path) -> None:
    task = Task(
        title="Evaluator Failure",
        description="Run failing script",
        constraints=[],
        subtasks=[],
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    save_workspace_files(
        paths=run_paths,
        files={"main.py": "raise RuntimeError('boom')\n"},
    )
    state = _build_state(task=task, run_paths=run_paths, retry_count=0)

    result = evaluator_node(state)

    assert result["status"] == "coding"
    assert result["current_step"] == "coder"
    assert result["retry_count"] == 1
    assert result["execution_return_code"] != 0
    assert result["result"].success is False
    assert "boom" in result["execution_stderr"]
