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
from schema.task import SubTask, Task  # noqa: E402
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
        "execution_entrypoint": task.execution_entrypoint,
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


def test_researcher_includes_failure_context_and_subtasks_in_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_researcher_includes_failure_context_and_subtasks_in_prompt を実行する。
    """
    task = Task(
        title="Researcher Retry",
        description="Investigate a failing dependency issue",
        constraints=["local only"],
        subtasks=[],
    )
    task.subtasks = [
        SubTask(
            title="Review dependency options",
            description="Find a replacement for the missing package",
            assigned_agent="researcher",
            status="pending",
        )
    ]
    state = _build_state(task)
    state["execution_stderr"] = "ModuleNotFoundError: No module named 'missing_pkg'"
    state["evaluator_feedback"] = {
        "summary": "missing_module",
        "likely_cause": "Module 'missing_pkg' is not installed.",
        "suggested_fixes": ["Find an alternative package"],
        "can_self_fix": False,
        "needs_research": True,
        "return_code": 1,
        "stdout": "",
        "stderr": state["execution_stderr"],
        "raw": {},
    }

    monkeypatch.setattr(researcher_mod, "build_arxiv_query", lambda **_kwargs: "query")
    monkeypatch.setattr(researcher_mod, "fetch_arxiv_raw", lambda **_kwargs: ["raw"])
    monkeypatch.setattr(researcher_mod, "parse_arxiv_response", lambda _raw: ["paper"])
    monkeypatch.setattr(
        researcher_mod, "format_papers_for_llm", lambda _papers: "paper context"
    )

    captured_system_prompts: list[str] = []

    def _fake_generate_text(**kwargs: Any) -> str:
        captured_system_prompts.append(kwargs["system_prompt"])
        return "## Retry Context\n- Replace the dependency."

    monkeypatch.setattr(researcher_mod, "generate_text", _fake_generate_text)

    result = researcher_mod.researcher_node(state)

    assert result["status"] == "researching"
    assert len(captured_system_prompts) == 1
    system_prompt = captured_system_prompts[0]
    assert "missing_module" in system_prompt
    assert "Module 'missing_pkg' is not installed." in system_prompt
    assert "ModuleNotFoundError" in system_prompt
    assert "Review dependency options" in system_prompt


def test_researcher_searches_with_task_keywords_and_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    test_researcher_searches_with_task_keywords_and_fallbacks を実行する。
    """
    task = Task(
        title="Transformer Model Implementation",
        description="Implement a basic Transformer model based on 'Attention Is All You Need'.",
        constraints=["local only"],
        subtasks=[],
    )
    state = _build_state(task)
    state["execution_stderr"] = "ModuleNotFoundError: No module named 'torch'"
    state["evaluator_feedback"] = {
        "summary": "missing_module",
        "likely_cause": "Module 'torch' is not installed.",
        "suggested_fixes": ["Use a fallback implementation"],
        "can_self_fix": False,
        "needs_research": True,
        "return_code": 1,
        "stdout": "",
        "stderr": state["execution_stderr"],
        "raw": {},
    }

    captured_keywords: list[list[str]] = []
    captured_categories: list[list[str] | None] = []

    def _fake_build_arxiv_query(*, keywords: list[str], categories: list[str] | None) -> str:
        captured_keywords.append(keywords)
        captured_categories.append(categories)
        return f"query-{len(captured_keywords)}"

    def _fake_fetch_arxiv_raw(**kwargs: Any) -> list[str]:
        if kwargs["query"] in {"query-1", "query-2"}:
            return []
        return ["raw-paper"]

    def _fake_parse_arxiv_response(raw: list[str]) -> list[str]:
        if not raw:
            return []
        return ["paper"]

    monkeypatch.setattr(researcher_mod, "build_arxiv_query", _fake_build_arxiv_query)
    monkeypatch.setattr(researcher_mod, "fetch_arxiv_raw", _fake_fetch_arxiv_raw)
    monkeypatch.setattr(researcher_mod, "parse_arxiv_response", _fake_parse_arxiv_response)
    monkeypatch.setattr(
        researcher_mod, "format_papers_for_llm", lambda _papers: "paper context"
    )
    monkeypatch.setattr(
        researcher_mod,
        "generate_text",
        lambda **_kwargs: "## Retry Context\n- Use a minimal fallback implementation.",
    )

    result = researcher_mod.researcher_node(state)

    assert result["status"] == "researching"
    assert len(captured_keywords) == 3
    assert "transformer" in [term.lower() for term in captured_keywords[0]]
    assert "attention is all you need" in [
        term.lower() for term in captured_keywords[0]
    ]
    assert "missing_module" not in [term.lower() for term in captured_keywords[0]]
    assert "module 'torch' is not installed." not in [
        term.lower() for term in captured_keywords[0]
    ]
    assert captured_categories[0] == ["cs.AI", "cs.LG"]
    assert captured_categories[-1] is None


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


def test_coder_includes_subtasks_and_entrypoint_in_system_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    test_coder_includes_subtasks_and_entrypoint_in_system_prompt を実行する。
    """
    task = Task(
        title="Coder Prompt Context",
        description="Use planning context while editing",
        constraints=["local"],
        subtasks=[
            SubTask(
                title="Implement parser",
                description="Create a small parser module",
                assigned_agent="coder",
                status="pending",
            )
        ],
        execution_entrypoint="src/app.py",
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    save_workspace_files(
        paths=run_paths,
        files={"src/app.py": "print('ok')\n"},
    )
    state = _build_state(task, run_paths=run_paths)

    captured_messages: list[list[dict[str, Any]]] = []

    def _fake_generate_with_tools(**kwargs: Any) -> Dict[str, Any]:
        captured_messages.append(kwargs["messages"])
        return {"content": "DONE", "tool_calls": []}

    monkeypatch.setattr(coder_mod, "generate_with_tools", _fake_generate_with_tools)

    result = coder_mod.coder_node(state)

    assert result["status"] == "coding"
    assert len(captured_messages) == 1
    system_prompt = captured_messages[0][0]["content"]
    assert "Implement parser" in system_prompt
    assert "src/app.py" in system_prompt


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

    captured_message_batches: list[list[dict[str, Any]]] = []

    def _fake_generate_with_tools(**kwargs: Any) -> Dict[str, Any]:
        captured_message_batches.append([dict(message) for message in kwargs["messages"]])
        if len(captured_message_batches) == 1:
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
    assert len(captured_message_batches) == 2
    assert captured_message_batches[0][0]["role"] == "system"
    assert captured_message_batches[0][1]["role"] == "user"
    second_batch = captured_message_batches[1]
    assert any(message["role"] == "tool" for message in second_batch)
    assert any(
        message.get("role") == "assistant" and message.get("tool_calls")
        for message in second_batch
    )


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

    captured_message_batches: list[list[dict[str, Any]]] = []

    def _fake_generate_with_tools(**kwargs: Any) -> Dict[str, Any]:
        captured_message_batches.append([dict(message) for message in kwargs["messages"]])
        if len(captured_message_batches) == 1:
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
    assert len(captured_message_batches) == 2
    first_user_prompt = captured_message_batches[0][1]["content"]
    second_user_prompt = captured_message_batches[1][1]["content"]
    assert first_user_prompt.splitlines()[0] == "summary: name_or_type_error"
    for prompt in [first_user_prompt, second_user_prompt]:
        assert "Prioritize fixing the direct cause of the previous failure" in prompt
        assert "avoid unrelated changes" in prompt
        assert "summary: name_or_type_error" in prompt
        assert "inspect main.py before editing it" in prompt
        assert "likely_cause: Mismatched variable name, attribute, or type." in prompt
        assert "- Check for typos in variable names" in prompt
        assert state["execution_stderr"].rstrip() in prompt
    second_batch = captured_message_batches[1]
    assert not any(message["role"] == "tool" for message in captured_message_batches[0])
    assert any(message["role"] == "tool" for message in second_batch)


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

    captured_message_batches: list[list[dict[str, Any]]] = []
    llm_call_count = 0

    def _fake_generate_with_tools(**kwargs: Any) -> Dict[str, Any]:
        nonlocal llm_call_count
        llm_call_count += 1
        messages = kwargs["messages"]
        captured_message_batches.append([dict(message) for message in messages])

        prompt = messages[1]["content"]
        assert prompt.splitlines()[0] == "summary: name_or_type_error"
        assert "inspect main.py before editing it" in prompt

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
    assert len(captured_message_batches) == 2
    assert not any(
        message["role"] == "tool" for message in captured_message_batches[0]
    )
    assert any(message["role"] == "tool" for message in captured_message_batches[1])

    success_eval_input = {**coder_input_state, **coder_result}
    success_eval_state = evaluator_mod.evaluator_node(success_eval_input)

    assert success_eval_state["status"] == "completed"
    assert success_eval_state["current_step"] == "done"
    assert success_eval_state["result"].success is True
    assert success_eval_state["execution_stdout"].strip() == "ok"
    assert success_eval_state["evaluator_feedback"]["summary"] == "success"


def test_v02a_baseline_llm_flow_still_works(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    test_v02a_baseline_llm_flow_still_works を実行する。

    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
        tmp_path: pytestの一時ディレクトリパス。
    """
    task = Task(
        title="Baseline Regression Flow",
        description="Ensure planner, researcher, coder, and evaluator still complete one pass",
        constraints=["local only"],
        subtasks=[],
    )
    initial_state = _build_state(task)

    monkeypatch.setattr(
        planner_mod,
        "generate_text",
        lambda **_kwargs: json.dumps(
            [
                {
                    "title": "Research implementation details",
                    "description": "Summarize relevant implementation details",
                    "assigned_agent": "researcher",
                    "status": "pending",
                },
                {
                    "title": "Write runnable code",
                    "description": "Implement the minimal runnable script",
                    "assigned_agent": "coder",
                    "status": "pending",
                },
            ]
        ),
    )
    monkeypatch.setattr(researcher_mod, "build_arxiv_query", lambda **_kwargs: "query")
    monkeypatch.setattr(researcher_mod, "fetch_arxiv_raw", lambda **_kwargs: ["raw"])
    monkeypatch.setattr(researcher_mod, "parse_arxiv_response", lambda _raw: ["paper"])
    monkeypatch.setattr(
        researcher_mod,
        "format_papers_for_llm",
        lambda _papers: "paper context",
    )
    monkeypatch.setattr(
        researcher_mod,
        "generate_text",
        lambda **_kwargs: "## Overview\n- Build the smallest runnable baseline.",
    )
    monkeypatch.setattr(
        coder_mod,
        "generate_with_tools",
        lambda **_kwargs: {
            "content": json.dumps({"files": {"main.py": "print('baseline ok')\n"}}),
            "tool_calls": [],
        },
    )

    planner_result = planner_mod.planner_node(initial_state)
    planned_task = planner_result["task"]
    run_paths = create_run_paths(task_id=planned_task.id, base_dir=tmp_path)

    researcher_input = {**initial_state, **planner_result, "run_paths": run_paths}
    researcher_result = researcher_mod.researcher_node(researcher_input)

    coder_input = {
        **initial_state,
        **planner_result,
        **researcher_result,
        "task": planned_task,
        "run_paths": run_paths,
    }
    coder_result = coder_mod.coder_node(coder_input)

    evaluator_input = {**coder_input, **coder_result}
    evaluator_result = evaluator_mod.evaluator_node(evaluator_input)

    assert planner_result["current_step"] == "researcher"
    assert researcher_result["current_step"] == "coder"
    assert coder_result["current_step"] == "evaluator"
    assert evaluator_result["status"] == "completed"
    assert evaluator_result["result"].success is True
    assert evaluator_result["execution_stdout"].strip() == "baseline ok"


def test_coder_can_apply_minimal_diff_with_replace_string_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    test_coder_can_apply_minimal_diff_with_replace_string_tool を実行する。

    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
        tmp_path: pytestの一時ディレクトリパス。
    """
    task = Task(
        title="Minimal Diff Edit",
        description="Edit only the necessary part of main.py",
        constraints=["local"],
        subtasks=[],
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    save_workspace_files(
        paths=run_paths,
        files={
            "main.py": (
                "VALUE = 'before'\n"
                "print(VALUE)\n"
            ),
            "notes.txt": "do not change this file\n",
        },
    )
    state = _build_state(task, run_paths=run_paths)
    state["research_context"] = "Change only the printed value."

    call_count = 0

    def _fake_generate_with_tools(**_kwargs: Any) -> Dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "replace_string",
                            "arguments": json.dumps(
                                {
                                    "file_path": "main.py",
                                    "old": "'before'",
                                    "new": "'after'",
                                }
                            ),
                        },
                    }
                ],
            }

        return {
            "content": json.dumps({"files": {}}),
            "tool_calls": [],
        }

    monkeypatch.setattr(coder_mod, "generate_with_tools", _fake_generate_with_tools)

    result = coder_mod.coder_node(state)

    assert result["status"] == "coding"
    assert result["current_step"] == "evaluator"
    assert result["generated_code"] == "VALUE = 'after'\nprint(VALUE)\n"
    assert result["generated_files"]["notes.txt"] == "do not change this file\n"


def test_evaluator_to_researcher_to_coder_route_is_integrated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    test_evaluator_to_researcher_to_coder_route_is_integrated を実行する。

    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
        tmp_path: pytestの一時ディレクトリパス。
    """
    task = Task(
        title="Research Route Integration",
        description="A missing dependency should route through researcher before coder",
        constraints=["local"],
        subtasks=[],
    )
    run_paths = create_run_paths(task_id=task.id, base_dir=tmp_path)
    save_workspace_files(
        paths=run_paths,
        files={"main.py": "import not_installed_module_abcdefg\n"},
    )
    base_state = _build_state(task, run_paths=run_paths)

    failed_eval_state = evaluator_mod.evaluator_node(base_state)

    assert failed_eval_state["status"] == "researching"
    assert failed_eval_state["current_step"] == "researcher"

    monkeypatch.setattr(researcher_mod, "build_arxiv_query", lambda **_kwargs: "query")
    monkeypatch.setattr(researcher_mod, "fetch_arxiv_raw", lambda **_kwargs: ["raw"])
    monkeypatch.setattr(researcher_mod, "parse_arxiv_response", lambda _raw: ["paper"])
    monkeypatch.setattr(
        researcher_mod,
        "format_papers_for_llm",
        lambda _papers: "dependency replacement guidance",
    )
    monkeypatch.setattr(
        researcher_mod,
        "generate_text",
        lambda **_kwargs: "## Implementation Notes\n- Replace the unavailable import with local code.",
    )
    monkeypatch.setattr(
        coder_mod,
        "generate_with_tools",
        lambda **_kwargs: {
            "content": json.dumps({"files": {"main.py": "print('dependency removed')\n"}}),
            "tool_calls": [],
        },
    )

    researcher_input = {**base_state, **failed_eval_state}
    researcher_result = researcher_mod.researcher_node(researcher_input)
    coder_input = {**researcher_input, **researcher_result}
    coder_result = coder_mod.coder_node(coder_input)

    assert researcher_result["current_step"] == "coder"
    assert "Implementation Notes" in researcher_result["research_context"]
    assert coder_result["current_step"] == "evaluator"
    assert coder_result["generated_code"] == "print('dependency removed')\n"


def test_same_project_run_chain_preserves_workspace_integration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    test_same_project_run_chain_preserves_workspace_integration を実行する。

    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
        tmp_path: pytestの一時ディレクトリパス。
    """
    first_task = Task(
        title="Project Workspace Seed",
        description="Seed a project workspace",
        constraints=["local"],
        subtasks=[],
    )
    first_run_paths = create_run_paths(
        task_id=first_task.id,
        base_dir=tmp_path,
        project_id="project-chain",
    )
    save_workspace_files(
        paths=first_run_paths,
        files={
            "main.py": "print('seed from first run')\n",
            "notes/context.txt": "shared context\n",
        },
    )

    second_task = Task(
        title="Project Workspace Inherit",
        description="Use the inherited workspace as the next source of truth",
        constraints=["local"],
        subtasks=[],
    )
    second_run_paths = create_run_paths(
        task_id=second_task.id,
        base_dir=tmp_path,
        project_id="project-chain",
    )
    state = _build_state(second_task, run_paths=second_run_paths)
    state["research_context"] = "Keep the inherited workspace untouched."

    monkeypatch.setattr(
        coder_mod,
        "generate_with_tools",
        lambda **_kwargs: {"content": '{"files": {}}', "tool_calls": []},
    )

    result = coder_mod.coder_node(state)

    assert second_run_paths.parent_run_id == first_run_paths.run_id
    assert result["generated_code"] == "print('seed from first run')\n"
    assert result["generated_files"]["notes/context.txt"] == "shared context\n"


def test_different_projects_do_not_mix_workspace_integration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    test_different_projects_do_not_mix_workspace_integration を実行する。

    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
        tmp_path: pytestの一時ディレクトリパス。
    """
    project_a_seed = create_run_paths(
        task_id="task-a-seed",
        base_dir=tmp_path,
        project_id="project-a",
    )
    save_workspace_files(
        paths=project_a_seed,
        files={"main.py": "print('from project a')\n"},
    )

    project_b_seed = create_run_paths(
        task_id="task-b-seed",
        base_dir=tmp_path,
        project_id="project-b",
    )
    save_workspace_files(
        paths=project_b_seed,
        files={"main.py": "print('from project b')\n"},
    )

    project_a_next = create_run_paths(
        task_id="task-a-next",
        base_dir=tmp_path,
        project_id="project-a",
    )
    project_b_next = create_run_paths(
        task_id="task-b-next",
        base_dir=tmp_path,
        project_id="project-b",
    )

    monkeypatch.setattr(
        coder_mod,
        "generate_with_tools",
        lambda **_kwargs: {"content": '{"files": {}}', "tool_calls": []},
    )

    task_a = Task(
        title="Project A Continue",
        description="Continue only project A history",
        constraints=["local"],
        subtasks=[],
    )
    task_b = Task(
        title="Project B Continue",
        description="Continue only project B history",
        constraints=["local"],
        subtasks=[],
    )

    result_a = coder_mod.coder_node(_build_state(task_a, run_paths=project_a_next))
    result_b = coder_mod.coder_node(_build_state(task_b, run_paths=project_b_next))

    assert project_a_next.parent_run_id == project_a_seed.run_id
    assert project_b_next.parent_run_id == project_b_seed.run_id
    assert result_a["generated_code"] == "print('from project a')\n"
    assert result_b["generated_code"] == "print('from project b')\n"
