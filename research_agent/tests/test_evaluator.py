from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from graph.nodes.evaluator import evaluator_node  # noqa: E402
from schema.task import Task  # noqa: E402
from tools.file_io import create_run_paths, read_execution_log, save_workspace_files  # noqa: E402


def _build_state(task: Task, run_paths: Any, retry_count: int = 0) -> Dict[str, Any]:
    """
    _build_state を実行する。
    
    Args:
        task: 対象タスク。
        run_paths: 実行ディレクトリ群を保持する RunPaths。
        retry_count: retry_count に関する値。
    """
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
        "evaluator_feedback": None,
        "error_signature": None,
        "same_error_count": 0,
        "stop_reason": None,
        "result": None,
        "run_paths": run_paths,
        "error": None,
    }


def test_evaluator_marks_completed_on_success(tmp_path: Path) -> None:
    """
    test_evaluator_marks_completed_on_success を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    task = Task(
        title="Evaluator Success",
        description="Run simple script",
        constraints=[],
        subtasks=[],
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    save_workspace_files(paths=run_paths, files={"main.py": "print('ok')\n"})
    state = _build_state(task=task, run_paths=run_paths, retry_count=0)
    state["error_signature"] = "old_signature"
    state["same_error_count"] = 7

    result = evaluator_node(state)

    assert result["status"] == "completed"
    assert result["current_step"] == "done"
    assert result["retry_count"] == 0
    assert result["execution_return_code"] == 0
    assert result["execution_stdout"].strip() == "ok"
    assert result["result"].success is True
    assert result["evaluator_feedback"]["summary"] == "success"
    assert result["error_signature"] is None
    assert result["same_error_count"] == 0
    assert "=== STDOUT ===" in result["execution_logs"]
    assert "=== FEEDBACK ===" in result["execution_logs"]
    assert "summary: success" in result["execution_logs"]
    assert "ok" in read_execution_log(run_paths)


def test_evaluator_returns_to_coder_on_failure(tmp_path: Path) -> None:
    """
    test_evaluator_returns_to_coder_on_failure を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    task = Task(
        title="Evaluator Failure",
        description="Run failing script",
        constraints=[],
        subtasks=[],
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    save_workspace_files(
        paths=run_paths,
        files={"main.py": "print(missing_name)\n"},
    )
    state = _build_state(task=task, run_paths=run_paths, retry_count=0)

    result = evaluator_node(state)

    assert result["status"] == "coding"
    assert result["current_step"] == "coder"
    assert result["retry_count"] == 1
    assert result["execution_return_code"] != 0
    assert result["result"].success is False
    assert result["evaluator_feedback"]["summary"] == "name_or_type_error"
    assert "likely_cause" in result["evaluator_feedback"]
    assert "suggested_fixes" in result["evaluator_feedback"]
    assert result["evaluator_feedback"]["can_self_fix"] is True
    assert result["evaluator_feedback"]["needs_research"] is False
    assert result["error_signature"] is not None
    assert result["same_error_count"] == 1
    assert "=== FEEDBACK ===" in result["execution_logs"]
    assert "missing_name" in result["execution_stderr"]


def test_evaluator_increments_same_error_count_on_repeated_failure(tmp_path: Path) -> None:
    """
    test_evaluator_increments_same_error_count_on_repeated_failure を実行する。

    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    task = Task(
        title="Evaluator Same Error",
        description="Repeat same runtime error",
        constraints=[],
        subtasks=[],
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    save_workspace_files(
        paths=run_paths,
        files={"main.py": "raise RuntimeError('boom')\n"},
    )

    first_state = _build_state(task=task, run_paths=run_paths, retry_count=0)
    first = evaluator_node(first_state)
    assert first["same_error_count"] == 1
    assert first["error_signature"] is not None

    second_state = _build_state(task=task, run_paths=run_paths, retry_count=1)
    second_state["error_signature"] = first["error_signature"]
    second_state["same_error_count"] = first["same_error_count"]

    second = evaluator_node(second_state)
    assert second["same_error_count"] == 2
    assert second["error_signature"] == first["error_signature"]


def test_evaluator_marks_module_not_found_as_needs_research(tmp_path: Path) -> None:
    """
    test_evaluator_marks_module_not_found_as_needs_research を実行する。

    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    task = Task(
        title="Evaluator Missing Module",
        description="Detect missing dependency",
        constraints=[],
        subtasks=[],
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    save_workspace_files(
        paths=run_paths,
        files={"main.py": "import not_installed_module_abcdefg\n"},
    )
    state = _build_state(task=task, run_paths=run_paths, retry_count=0)

    result = evaluator_node(state)
    feedback = result["evaluator_feedback"]

    assert result["status"] == "researching"
    assert result["current_step"] == "researcher"
    assert feedback["summary"] == "missing_module"
    assert feedback["needs_research"] is True
    assert feedback["can_self_fix"] is False
