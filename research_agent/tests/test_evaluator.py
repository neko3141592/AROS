from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.append(str(Path(__file__).resolve().parents[1]))

import graph.nodes.evaluator as evaluator_mod  # noqa: E402
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
        "execution_entrypoint": task.execution_entrypoint,
        "generated_code": None,
        "generated_files": None,
        "execution_logs": None,
        "execution_stdout": None,
        "execution_stderr": None,
        "execution_return_code": None,
        "last_execution_duration_sec": None,
        "total_execution_duration_sec": 0.0,
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
    assert result["stop_reason"] is None
    assert result["result"].stop_reason is None
    assert result["last_execution_duration_sec"] is not None
    assert result["total_execution_duration_sec"] is not None
    assert "=== STDOUT ===" in result["execution_logs"]
    assert "=== FEEDBACK ===" in result["execution_logs"]
    assert "summary: success" in result["execution_logs"]
    assert "ok" in read_execution_log(run_paths)


def test_evaluator_uses_dynamic_execution_entrypoint(tmp_path: Path) -> None:
    """
    test_evaluator_uses_dynamic_execution_entrypoint を実行する。
    """
    task = Task(
        title="Evaluator Dynamic Entrypoint",
        description="Run a non-main entrypoint",
        constraints=[],
        subtasks=[],
        execution_entrypoint="src/app.py",
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    save_workspace_files(paths=run_paths, files={"src/app.py": "print('dynamic ok')\n"})
    state = _build_state(task=task, run_paths=run_paths, retry_count=0)

    result = evaluator_node(state)

    assert result["status"] == "completed"
    assert result["execution_entrypoint"] == "src/app.py"
    assert result["execution_stdout"].strip() == "dynamic ok"
    assert "Command: python src/app.py" in result["execution_logs"]


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
    assert result["stop_reason"] is None
    assert result["result"].stop_reason is None
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
    assert second["stop_reason"] is None


def test_evaluator_stops_on_repeated_same_error(tmp_path: Path) -> None:
    """
    test_evaluator_stops_on_repeated_same_error を実行する。

    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    task = Task(
        title="Evaluator Repeated Error Stop",
        description="Stop after the same error repeats",
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

    second_state = _build_state(task=task, run_paths=run_paths, retry_count=1)
    second_state["error_signature"] = first["error_signature"]
    second_state["same_error_count"] = first["same_error_count"]
    second = evaluator_node(second_state)

    third_state = _build_state(task=task, run_paths=run_paths, retry_count=2)
    third_state["error_signature"] = second["error_signature"]
    third_state["same_error_count"] = second["same_error_count"]
    third = evaluator_node(third_state)

    assert third["status"] == "failed"
    assert third["current_step"] == "done"
    assert third["same_error_count"] == 3
    assert third["stop_reason"] == "repeated_error"
    assert third["result"].stop_reason == "repeated_error"
    assert "同一エラーが 3 回連続" in third["messages"][0].content


def test_evaluator_stops_on_max_retry_for_non_repeated_failures(tmp_path: Path) -> None:
    """
    test_evaluator_stops_on_max_retry_for_non_repeated_failures を実行する。

    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    task = Task(
        title="Evaluator Max Retry",
        description="Stop after retry limit on changing errors",
        constraints=[],
        subtasks=[],
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    save_workspace_files(
        paths=run_paths,
        files={"main.py": "print(missing_name)\n"},
    )

    state = _build_state(task=task, run_paths=run_paths, retry_count=2)
    state["error_signature"] = "different_signature"
    state["same_error_count"] = 1

    result = evaluator_node(state)

    assert result["status"] == "failed"
    assert result["current_step"] == "done"
    assert result["retry_count"] == 3
    assert result["stop_reason"] == "max_retry"
    assert result["result"].stop_reason == "max_retry"
    assert "リトライ上限（3回）" in result["messages"][0].content


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
    assert result["stop_reason"] is None


def test_evaluator_stops_on_total_execution_timeout(
    monkeypatch, tmp_path: Path
) -> None:
    """
    test_evaluator_stops_on_total_execution_timeout を実行する。

    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
        tmp_path: pytestの一時ディレクトリパス。
    """
    task = Task(
        title="Evaluator Total Timeout",
        description="Stop when cumulative runtime exceeds the budget",
        constraints=[],
        subtasks=[],
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    state = _build_state(task=task, run_paths=run_paths, retry_count=1)
    state["total_execution_duration_sec"] = 1.2

    monkeypatch.setenv("MAX_TOTAL_EXECUTION_TIME_SEC", "3.0")
    monkeypatch.setattr(
        evaluator_mod,
        "run_workspace_python",
        lambda **_kwargs: evaluator_mod.LocalExecutionResult(
            stdout="",
            stderr="RuntimeError: boom",
            returncode=1,
            duration_sec=2.1,
        ),
    )

    result = evaluator_mod.evaluator_node(state)

    assert result["status"] == "failed"
    assert result["current_step"] == "done"
    assert result["stop_reason"] == "total_timeout"
    assert result["result"].stop_reason == "total_timeout"
    assert result["total_execution_duration_sec"] == 3.3


def test_calculate_next_timeout_uses_remaining_budget() -> None:
    """
    test_calculate_next_timeout_uses_remaining_budget を実行する。
    """
    assert evaluator_mod.calculate_next_timeout(60.0, 180.0, 0.0) == 60.0
    assert evaluator_mod.calculate_next_timeout(60.0, 180.0, 179.0) == 1.0
    assert evaluator_mod.calculate_next_timeout(60.0, 180.0, 180.0) == 0.0
    assert evaluator_mod.calculate_next_timeout(60.0, 180.0, 300.0) == 0.0


def test_evaluator_skips_execution_when_total_budget_is_exhausted(
    monkeypatch, tmp_path: Path
) -> None:
    """
    test_evaluator_skips_execution_when_total_budget_is_exhausted を実行する。

    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
        tmp_path: pytestの一時ディレクトリパス。
    """
    task = Task(
        title="Evaluator Exhausted Budget",
        description="Stop before running when total budget is exhausted",
        constraints=[],
        subtasks=[],
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    save_workspace_files(paths=run_paths, files={"main.py": "print('ok')\n"})
    state = _build_state(task=task, run_paths=run_paths, retry_count=1)
    state["total_execution_duration_sec"] = 180.0

    monkeypatch.setenv("EXECUTION_TIMEOUT_SEC", "60.0")
    monkeypatch.setenv("MAX_TOTAL_EXECUTION_TIME_SEC", "180.0")
    monkeypatch.setattr(
        evaluator_mod,
        "run_workspace_python",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("run_workspace_python must not be called.")
        ),
    )

    result = evaluator_mod.evaluator_node(state)

    assert result["status"] == "failed"
    assert result["current_step"] == "done"
    assert result["stop_reason"] == "total_timeout"
    assert result["result"].stop_reason == "total_timeout"
    assert result["execution_return_code"] is None
    assert result["last_execution_duration_sec"] == 0.0
    assert result["total_execution_duration_sec"] == 180.0
    assert result["retry_count"] == 1
    assert result["evaluator_feedback"]["summary"] == "timeout"
    assert "Skipped: total timeout budget exhausted before execution." in result["execution_logs"]


def test_evaluator_uses_llm_analysis_when_enabled(
    monkeypatch, tmp_path: Path
) -> None:
    """
    test_evaluator_uses_llm_analysis_when_enabled を実行する。

    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
        tmp_path: pytestの一時ディレクトリパス。
    """
    task = Task(
        title="Evaluator LLM Analysis",
        description="Use LLM for better fixes",
        constraints=[],
        subtasks=[],
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    save_workspace_files(paths=run_paths, files={"main.py": "print(missing_name)\n"})
    state = _build_state(task=task, run_paths=run_paths, retry_count=0)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        evaluator_mod,
        "generate_text",
        lambda **_kwargs: json.dumps(
            {
                "likely_cause": "The variable was never defined before use.",
                "suggested_fixes": [
                    "Define missing_name before printing it",
                    "Re-read main.py and update only the failing line",
                ],
                "can_self_fix": True,
                "needs_research": False,
            }
        ),
    )

    result = evaluator_mod.evaluator_node(state)
    feedback = result["evaluator_feedback"]

    assert feedback["likely_cause"] == "The variable was never defined before use."
    assert feedback["suggested_fixes"][0] == "Define missing_name before printing it"
    assert feedback["can_self_fix"] is True
    assert feedback["needs_research"] is False


def test_evaluator_falls_back_when_llm_analysis_fails(
    monkeypatch, tmp_path: Path
) -> None:
    """
    test_evaluator_falls_back_when_llm_analysis_fails を実行する。

    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
        tmp_path: pytestの一時ディレクトリパス。
    """
    task = Task(
        title="Evaluator LLM Fallback",
        description="Fallback to heuristic feedback",
        constraints=[],
        subtasks=[],
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    save_workspace_files(paths=run_paths, files={"main.py": "print(missing_name)\n"})
    state = _build_state(task=task, run_paths=run_paths, retry_count=0)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        evaluator_mod,
        "generate_text",
        lambda **_kwargs: "not-json",
    )

    result = evaluator_mod.evaluator_node(state)
    feedback = result["evaluator_feedback"]

    assert feedback["summary"] == "name_or_type_error"
    assert "Mismatched variable name" in feedback["likely_cause"]


def test_evaluator_keeps_heuristic_routing_flags_for_name_error(
    monkeypatch, tmp_path: Path
) -> None:
    """
    test_evaluator_keeps_heuristic_routing_flags_for_name_error を実行する。

    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
        tmp_path: pytestの一時ディレクトリパス。
    """
    task = Task(
        title="Evaluator Stable NameError Routing",
        description="Do not let LLM flip routing flags for name errors",
        constraints=[],
        subtasks=[],
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    save_workspace_files(paths=run_paths, files={"main.py": "print(missing_name)\n"})
    state = _build_state(task=task, run_paths=run_paths, retry_count=0)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        evaluator_mod,
        "generate_text",
        lambda **_kwargs: json.dumps(
            {
                "likely_cause": "Extra domain knowledge may be needed.",
                "suggested_fixes": ["Investigate the broader context"],
                "can_self_fix": False,
                "needs_research": True,
            }
        ),
    )

    result = evaluator_mod.evaluator_node(state)
    feedback = result["evaluator_feedback"]

    assert feedback["summary"] == "name_or_type_error"
    assert feedback["can_self_fix"] is True
    assert feedback["needs_research"] is False
    assert result["current_step"] == "coder"


def test_evaluator_keeps_heuristic_routing_flags_for_missing_module(
    monkeypatch, tmp_path: Path
) -> None:
    """
    test_evaluator_keeps_heuristic_routing_flags_for_missing_module を実行する。

    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
        tmp_path: pytestの一時ディレクトリパス。
    """
    task = Task(
        title="Evaluator Stable Missing Module Routing",
        description="Do not let LLM flip routing flags for missing modules",
        constraints=[],
        subtasks=[],
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    save_workspace_files(
        paths=run_paths,
        files={"main.py": "import not_installed_module_abcdefg\n"},
    )
    state = _build_state(task=task, run_paths=run_paths, retry_count=0)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        evaluator_mod,
        "generate_text",
        lambda **_kwargs: json.dumps(
            {
                "likely_cause": "This can likely be fixed locally.",
                "suggested_fixes": ["Edit the import to use an available module"],
                "can_self_fix": True,
                "needs_research": False,
            }
        ),
    )

    result = evaluator_mod.evaluator_node(state)
    feedback = result["evaluator_feedback"]

    assert feedback["summary"] == "missing_module"
    assert feedback["can_self_fix"] is False
    assert feedback["needs_research"] is True
    assert result["current_step"] == "researcher"


def test_sanitize_execution_outputs_for_llm_masks_and_limits() -> None:
    """
    test_sanitize_execution_outputs_for_llm_masks_and_limits を実行する。
    """
    long_token = "Abc1234567890Def4567890Ghijklmno"
    long_url = "https://internal.example.com/" + ("verylongpath/" * 10)
    stdout = (
        f"api_key={long_token}\n"
        f"Authorization: Bearer {long_token}\n"
        f"{long_url}\n"
        + ("A" * 600)
    )

    stderr_lines = [f"noise line {i}" for i in range(120)]
    stderr_lines.extend(
        [
            "Traceback (most recent call last):",
            '  File "/Users/alice/projects/aros/main.py", line 1, in <module>',
            "    raise RuntimeError('boom')",
            "RuntimeError: boom",
        ]
    )
    stderr_lines.extend(f"tail line {i}" for i in range(120))
    stderr = "\n".join(stderr_lines)

    safe_stdout, safe_stderr = evaluator_mod._sanitize_execution_outputs_for_llm(
        stdout,
        stderr,
    )

    assert "api_key=" in safe_stdout
    assert long_token not in safe_stdout
    assert "Bearer <REDACTED_TOKEN>" in safe_stdout
    assert "<REDACTED_URL>" in safe_stdout
    assert "<HOME_PATH>" in safe_stderr
    assert "RuntimeError: boom" in safe_stderr
    assert len(safe_stdout) <= evaluator_mod.LLM_STDOUT_MAX_CHARS
    assert len(safe_stderr) <= evaluator_mod.LLM_STDERR_MAX_CHARS


def test_evaluator_passes_sanitized_logs_to_prompt(monkeypatch) -> None:
    """
    test_evaluator_passes_sanitized_logs_to_prompt を実行する。

    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
    """
    task = Task(
        title="Prompt Sanitization",
        description="Ensure prompt gets redacted logs",
        constraints=[],
        subtasks=[],
    )
    execution = evaluator_mod.LocalExecutionResult(
        stdout="token=super-secret-token-value",
        stderr=(
            "Traceback (most recent call last):\n"
            '  File "/Users/alice/private/main.py", line 1, in <module>\n'
            "RuntimeError: failed\n"
        ),
        returncode=1,
        duration_sec=0.01,
    )
    base_feedback = {
        "summary": "runtime_error_unknown",
        "likely_cause": "Unclassified runtime error",
        "suggested_fixes": ["Inspect stderr stack trace"],
        "can_self_fix": False,
        "needs_research": True,
        "return_code": 1,
        "stdout": execution.stdout,
        "stderr": execution.stderr,
        "raw": {},
    }
    captured_vars: Dict[str, Any] = {}

    def _capture_render_prompt(prompt_name: str, variables: Dict[str, Any]) -> str:
        assert prompt_name == "system_evaluator"
        captured_vars.update(variables)
        return "captured prompt"

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(evaluator_mod, "render_prompt", _capture_render_prompt)
    monkeypatch.setattr(
        evaluator_mod,
        "generate_text",
        lambda **_kwargs: json.dumps(
            {
                "likely_cause": "A runtime error occurred.",
                "suggested_fixes": ["Check the failing line"],
                "can_self_fix": True,
                "needs_research": False,
            }
        ),
    )

    evaluator_mod._analyze_failure_with_llm(task, execution, base_feedback)

    expected_stdout, expected_stderr = evaluator_mod._sanitize_execution_outputs_for_llm(
        execution.stdout,
        execution.stderr,
    )
    assert captured_vars["stdout"] == expected_stdout
    assert captured_vars["stderr"] == expected_stderr
    assert "super-secret-token-value" not in captured_vars["stdout"]
    assert "/Users/alice/private/main.py" not in captured_vars["stderr"]
