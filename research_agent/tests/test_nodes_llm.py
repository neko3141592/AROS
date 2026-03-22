from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import graph.nodes.coder as coder_mod  # noqa: E402
import graph.nodes.evaluator as evaluator_mod  # noqa: E402
import graph.nodes.planner as planner_mod  # noqa: E402
import graph.nodes.researcher as researcher_mod  # noqa: E402
from schema.task import Task  # noqa: E402
from tools.file_io import create_run_paths, save_workspace_files  # noqa: E402


def _build_state(task: Task, run_paths: Any = None) -> Dict[str, Any]:
    """
    _build_state を実行する。
    
    Args:
        task: 対象タスク。
        run_paths: 実行ディレクトリ群を保持する RunPaths。
    """
    return {
        "task": task,
        "status": "starting",
        "messages": [],
        "current_step": "init",
        "research_context": "",
        "generated_code": None,
        "generated_files": None,
        "execution_logs": None,
        "execution_stdout": None,
        "execution_stderr": None,
        "execution_return_code": None,
        "last_execution_duration_sec": None,
        "total_execution_duration_sec": 0.0,
        "retry_count": 0,
        "evaluator_feedback": None,
        "error_signature": None,
        "same_error_count": 0,
        "stop_reason": None,
        "result": None,
        "run_paths": run_paths,
        "error": None,
    }


def test_planner_creates_subtasks_from_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_planner_creates_subtasks_from_json を実行する。
    
    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
    """
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
    """
    test_planner_handles_invalid_json_as_failed を実行する。
    
    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
    """
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
    """
    test_researcher_puts_summary_into_state を実行する。
    
    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
    """
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
    """
    test_coder_saves_workspace_files を実行する。
    
    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
        tmp_path: pytestの一時ディレクトリパス。
    """
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
    monkeypatch.setattr(
        coder_mod,
        "generate_with_tools",
        lambda **_kwargs: {"content": coder_json, "tool_calls": []},
    )

    result = coder_mod.coder_node(state)

    assert result["status"] == "coding"
    assert "generated_files" in result
    assert "analysis/run.py" in result["generated_files"]
    assert run_paths.code_path.read_text(encoding="utf-8") == result["generated_code"]
    assert run_paths.code_path == run_paths.workspace_dir / "main.py"
    assert not (run_paths.run_dir / "main.py").exists()
    assert (
        run_paths.workspace_dir / "analysis" / "run.py"
    ).read_text(encoding="utf-8") == "print('analysis')\n"


def test_coder_uses_workspace_as_source_of_truth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    test_coder_uses_workspace_as_source_of_truth を実行する。
    
    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
        tmp_path: pytestの一時ディレクトリパス。
    """
    task = Task(
        title="Coder Workspace Source",
        description="Use workspace as source of truth",
        constraints=["local"],
        subtasks=[],
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    save_workspace_files(
        paths=run_paths,
        files={"main.py": "print('from workspace')\n"},
    )
    state = _build_state(task, run_paths=run_paths)
    state["research_context"] = "Keep existing workspace files."

    # 最終JSONに files が無くても、workspace 実体から評価へ進めることを確認
    monkeypatch.setattr(
        coder_mod,
        "generate_with_tools",
        lambda **_kwargs: {"content": '{"files": {}}', "tool_calls": []},
    )

    result = coder_mod.coder_node(state)

    assert result["status"] == "coding"
    assert result["current_step"] == "evaluator"
    assert result["generated_code"] == "print('from workspace')\n"
    assert result["generated_files"]["main.py"] == "print('from workspace')\n"


def test_coder_registers_run_shell_command_tool() -> None:
    """
    test_coder_registers_run_shell_command_tool を実行する。

    Args:
        なし。
    """
    tool_names = [tool["function"]["name"] for tool in coder_mod.TOOL_SCHEMAS]

    assert "run_shell_command" in tool_names
    assert "run_shell_command" in coder_mod.TOOL_FUNCTIONS


def test_coder_can_use_run_shell_command_for_readonly_exploration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    test_coder_can_use_run_shell_command_for_readonly_exploration を実行する。

    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
        tmp_path: pytestの一時ディレクトリパス。
    """
    task = Task(
        title="Coder Shell Tool",
        description="Use shell exploration before editing",
        constraints=["local"],
        subtasks=[],
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    save_workspace_files(
        paths=run_paths,
        files={
            "main.py": "print('ok')\n",
            "notes.txt": "shell-guided context\n",
        },
    )
    state = _build_state(task, run_paths=run_paths)
    state["research_context"] = "Inspect notes if needed, but edit only workspace files."

    captured_prompts: list[str] = []

    def _fake_generate_with_tools(**kwargs: Any) -> Dict[str, Any]:
        captured_prompts.append(kwargs["user_prompt"])
        if len(captured_prompts) == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "run_shell_command",
                            "arguments": json.dumps({"command": "rg shell-guided notes.txt"}),
                        },
                    }
                ],
            }

        return {
            "content": json.dumps({"files": {"main.py": "print('still ok')\n"}}),
            "tool_calls": [],
        }

    monkeypatch.setattr(coder_mod, "generate_with_tools", _fake_generate_with_tools)

    result = coder_mod.coder_node(state)

    assert result["status"] == "coding"
    assert result["current_step"] == "evaluator"
    assert result["generated_code"] == "print('still ok')\n"
    assert len(captured_prompts) == 2
    assert "# Tool Execution History" in captured_prompts[1]
    assert "run_shell_command" in captured_prompts[1]


def test_coder_includes_failure_context_in_retry_prompts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    test_coder_includes_failure_context_in_retry_prompts を実行する。

    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
        tmp_path: pytestの一時ディレクトリパス。
    """
    task = Task(
        title="Coder Retry Context",
        description="Retry with previous failure context",
        constraints=["local"],
        subtasks=[],
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    state = _build_state(task, run_paths=run_paths)
    state["research_context"] = "Fix only the previous execution failure."
    state["retry_count"] = 1
    state["execution_stderr"] = (
        "Traceback (most recent call last):\n"
        "  File \"main.py\", line 1, in <module>\n"
        "    print(missing_name)\n"
        "NameError: name 'missing_name' is not defined\n"
    )
    state["evaluator_feedback"] = {
        "summary": "name_or_type_error",
        "likely_cause": "Mismatched variable name, attribute, or type.",
        "suggested_fixes": [
            "Check for typos in variable names",
            "Verify types (e.g. str vs int)",
        ],
        "can_self_fix": True,
        "needs_research": False,
        "return_code": 1,
        "stdout": "",
        "stderr": state["execution_stderr"],
        "raw": {},
    }

    captured_prompts: list[str] = []

    def _fake_generate_with_tools(**kwargs: Any) -> Dict[str, Any]:
        captured_prompts.append(kwargs["user_prompt"])
        if len(captured_prompts) == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "list_files",
                            "arguments": json.dumps(
                                {"base_dir": ".", "recursive": True}
                            ),
                        },
                    }
                ],
            }
        return {
            "content": json.dumps({"files": {"main.py": "print('fixed')\n"}}),
            "tool_calls": [],
        }

    monkeypatch.setattr(coder_mod, "generate_with_tools", _fake_generate_with_tools)

    result = coder_mod.coder_node(state)

    assert result["status"] == "coding"
    assert result["current_step"] == "evaluator"
    assert len(captured_prompts) == 2
    assert captured_prompts[0].splitlines()[0] == "summary: name_or_type_error"
    for prompt in captured_prompts:
        assert "前回失敗の直接原因を最優先で修正" in prompt
        assert "無関係な変更は行わないでください" in prompt
        assert "summary: name_or_type_error" in prompt
        assert "read_file(main.py)" in prompt
        assert "likely_cause: Mismatched variable name, attribute, or type." in prompt
        assert "- Check for typos in variable names" in prompt
        assert state["execution_stderr"].rstrip() in prompt
    assert "# Tool Execution History" not in captured_prompts[0]
    assert "# Tool Execution History" in captured_prompts[1]


def test_retry_loop_reaches_success_after_coder_fix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    test_retry_loop_reaches_success_after_coder_fix を実行する。

    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
        tmp_path: pytestの一時ディレクトリパス。
    """
    task = Task(
        title="Retry Loop Success",
        description="Failure is fixed by coder on retry",
        constraints=["local"],
        subtasks=[],
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    save_workspace_files(
        paths=run_paths,
        files={"main.py": "print(missing_name)\n"},
    )
    base_state = _build_state(task, run_paths=run_paths)
    base_state["research_context"] = "Fix only the failing line in main.py."

    failed_eval_state = evaluator_mod.evaluator_node(base_state)

    assert failed_eval_state["status"] == "coding"
    assert failed_eval_state["current_step"] == "coder"
    assert failed_eval_state["retry_count"] == 1
    assert failed_eval_state["evaluator_feedback"]["summary"] == "name_or_type_error"

    captured_prompts: list[str] = []
    llm_call_count = 0

    def _fake_generate_with_tools(**kwargs: Any) -> Dict[str, Any]:
        nonlocal llm_call_count
        llm_call_count += 1
        prompt = kwargs["user_prompt"]
        captured_prompts.append(prompt)

        assert prompt.splitlines()[0] == "summary: name_or_type_error"
        assert "read_file(main.py)" in prompt

        if llm_call_count == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"file_path": "main.py"}),
                        },
                    }
                ],
            }

        return {
            "content": json.dumps(
                {
                    "files": {
                        "main.py": (
                            "missing_name = 'ok'\n"
                            "print(missing_name)\n"
                        )
                    }
                }
            ),
            "tool_calls": [],
        }

    monkeypatch.setattr(coder_mod, "generate_with_tools", _fake_generate_with_tools)

    coder_input_state = {**base_state, **failed_eval_state}
    coder_result = coder_mod.coder_node(coder_input_state)

    assert coder_result["status"] == "coding"
    assert coder_result["current_step"] == "evaluator"
    assert coder_result["generated_code"] == "missing_name = 'ok'\nprint(missing_name)\n"
    assert len(captured_prompts) == 2
    assert "# Tool Execution History" not in captured_prompts[0]
    assert "# Tool Execution History" in captured_prompts[1]

    success_eval_input = {**coder_input_state, **coder_result}
    success_eval_state = evaluator_mod.evaluator_node(success_eval_input)

    assert success_eval_state["status"] == "completed"
    assert success_eval_state["current_step"] == "done"
    assert success_eval_state["result"].success is True
    assert success_eval_state["execution_stdout"].strip() == "ok"
    assert success_eval_state["evaluator_feedback"]["summary"] == "success"
