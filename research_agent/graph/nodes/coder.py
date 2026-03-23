from __future__ import annotations

import os
from typing import Any, Dict

from langchain_core.messages import AIMessage

from graph.state import AgentState
from tools.cli_logging import (
    log_llm_request,
    log_llm_response,
    log_node_end,
    log_node_start,
    log_tool_result,
    preview_text,
)
from tools.coder_helpers import (
    _build_failure_context,
    _maybe_extract_legacy_files_payload,
    _build_user_prompt,
    _build_tool_result_message,
    _build_workspace_state_payload,
    _execute_tool_call,
)
from tools.file_io import save_generated_code, save_workspace_files
from tools.llm_client import LLMClientError, generate_with_tools
from tools.prompt_manager import PromptManagerError, render_prompt
from tools.coder_tooling import TOOL_FUNCTIONS, TOOL_SCHEMAS
from tools.task_context import format_subtasks_for_prompt, resolve_execution_entrypoint
from tools.workspace_tools import (
    list_files as workspace_list_files,
    read_file as workspace_read_file,
)
from tools.model_config import get_model_name

DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_MAX_TOOL_STEPS = 8


def coder_node(state: AgentState) -> Dict[str, Any]:
    """
    Args:
        state: ノード間で受け渡す現在の状態。
    """
    task = state.get("task")
    if task is None:
        return {
            "status": "failed",
            "error": "Coder: state['task'] が存在しません。",
            "messages": [
                AIMessage(
                    content="Coder failed: task が未設定のため、コード生成を実行できませんでした。"
                )
            ],
        }

    run_paths = state.get("run_paths")
    if run_paths is None:
        return {
            "status": "failed",
            "error": "Coder: run_paths が存在しません。",
            "messages": [AIMessage(content="Coder failed: run_paths is missing.")],
        }

    try:
        log_node_start(
            "Coder",
            {
                "task_title": task.title,
                "retry_count": state.get("retry_count", 0),
                "workspace_dir": run_paths.workspace_dir,
            },
        )
        system_prompt = render_prompt(
            "system_coder",
            {
                "task_title": task.title,
                "task_description": task.description,
                "task_constraints": task.constraints,
                "task_subtasks": format_subtasks_for_prompt(task.subtasks),
                "target_entrypoint": resolve_execution_entrypoint(
                    task,
                    state.get("execution_entrypoint"),
                ),
                "research_context": state.get("research_context", ""),
            },
        )
        execution_entrypoint = resolve_execution_entrypoint(
            task,
            state.get("execution_entrypoint"),
        )
        base_user_prompt = (
            "Use the provided tools to inspect and edit the workspace until the task is complete. "
            "Prioritize fixing the direct cause of the previous failure first, and avoid unrelated changes. "
            "When you are done, stop calling tools and return a short plain-text completion message only. "
            "Do not return JSON or Markdown."
        )

        model_name = get_model_name("CODER_MODEL_NAME", DEFAULT_MODEL_NAME)
        max_tool_steps = int(
            os.getenv("MAX_TOOL_STEPS", str(DEFAULT_MAX_TOOL_STEPS)).strip()
            or str(DEFAULT_MAX_TOOL_STEPS)
        )
        if max_tool_steps < 1:
            raise ValueError("MAX_TOOL_STEPS must be >= 1.")

        failure_context = _build_failure_context(
            evaluator_feedback=state.get("evaluator_feedback"),
            execution_stderr=state.get("execution_stderr"),
            retry_count=state.get("retry_count", 0),
            entrypoint=execution_entrypoint,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": _build_user_prompt(
                    base_user_prompt,
                    failure_context=failure_context,
                ),
            },
        ]
        for step in range(max_tool_steps):
            log_llm_request(
                "Coder",
                model_name,
                system_prompt=None,
                user_prompt=messages[-1].get("content") if messages else "",
                tools=[tool["function"]["name"] for tool in TOOL_SCHEMAS],
                message_count=len(messages),
            )
            response = generate_with_tools(
                messages=messages,
                model=model_name,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.2,
                timeout=60,
            )
            tool_calls = response.get("tool_calls") or []
            content = (response.get("content") or "").strip()
            log_llm_response(
                "Coder",
                content,
                tool_calls if isinstance(tool_calls, list) else None,
            )

            if isinstance(tool_calls, list) and tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                    }
                )
                step_results: list[dict[str, Any]] = []
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        tool_result = {
                            "ok": False,
                            "name": "unknown",
                            "error": "Tool call payload must be an object.",
                        }
                        step_results.append(tool_result)
                        log_tool_result("Coder", tool_result)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": "",
                                "name": "unknown",
                                "content": _build_tool_result_message(tool_result),
                            }
                        )
                        continue
                    tool_result = _execute_tool_call(tool_call, run_paths, TOOL_FUNCTIONS)
                    step_results.append(tool_result)
                    log_tool_result("Coder", tool_result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(tool_result.get("id", "")),
                            "name": str(tool_result.get("name", "unknown")),
                            "content": _build_tool_result_message(tool_result),
                        }
                    )
                continue

            if content:
                messages.append({"role": "assistant", "content": content})
                legacy_files = _maybe_extract_legacy_files_payload(content)
                if legacy_files:
                    save_workspace_files(run_paths, legacy_files)
                try:
                    workspace_state = _build_workspace_state_payload(
                        run_paths,
                        workspace_list_files,
                        workspace_read_file,
                        save_generated_code,
                        entrypoint=execution_entrypoint,
                    )
                except ValueError as exc:
                    if step >= max_tool_steps - 1:
                        raise
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The task is not complete yet. "
                                f"{exc} Continue editing the workspace with tools. "
                                "Do not restart from scratch."
                            ),
                        }
                    )
                    continue

                generated_files = workspace_state["generated_files"]
                log_node_end(
                    "Coder",
                    {
                        "workspace_files": len(generated_files),
                        "entrypoint": execution_entrypoint,
                        "completion_message": preview_text(content),
                        "next_step": "evaluator",
                    },
                )
                return {
                    **workspace_state,
                    "execution_entrypoint": execution_entrypoint,
                    "status": "coding",
                    "current_step": "evaluator",
                    "messages": [
                        AIMessage(
                            content=(
                                "Coder: Tool Callingで編集を実行し、"
                                f"workspace を {len(generated_files)}件のファイル状態として確定しました。"
                            )
                        )
                    ],
                    "error": None,
                }

        # max_tool_steps到達時は、workspaceにあるファイルを収集して返却を試みる
        workspace_state = _build_workspace_state_payload(
            run_paths,
            workspace_list_files,
            workspace_read_file,
            save_generated_code,
            entrypoint=execution_entrypoint,
        )
        log_node_end(
            "Coder",
            {
                "workspace_files": len(workspace_state["generated_files"]),
                "entrypoint": execution_entrypoint,
                "reason": "max_tool_steps_reached",
                "next_step": "evaluator",
            },
        )
        return {
            **workspace_state,
            "execution_entrypoint": execution_entrypoint,
            "status": "coding",
            "current_step": "evaluator",
            "messages": [
                AIMessage(
                    content=(
                        "Coder: Tool Callingの最大ステップに到達しましたが、"
                        "workspace の内容をそのまま評価に渡します。"
                    )
                )
            ],
            "error": None,
        }

    except PromptManagerError as exc:
        return {
            "status": "failed",
            "error": f"Coder prompt error: {exc}",
            "messages": [
                AIMessage(
                    content=(
                        "Coder failed: システムプロンプトの読み込みまたは適用に失敗しました。"
                    )
                )
            ],
        }
    except (LLMClientError, ValueError) as exc:
        return {
            "status": "failed",
            "error": f"Coder parse/runtime error: {exc}",
            "messages": [
                AIMessage(
                    content=(
                        "Coder failed: ツール実行、会話履歴処理、"
                        "または workspace 検証に失敗しました。"
                    )
                )
            ],
        }


coder = coder_node
