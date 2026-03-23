from __future__ import annotations

import json
import os
from json import JSONDecodeError
from typing import Any, Dict

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from graph.state import AgentState
from tools.coder_helpers import (
    _build_failure_context,
    _build_user_prompt,
    _build_workspace_state_payload,
    _execute_tool_call,
    _parse_coder_output,
)
from tools.file_io import save_generated_code, save_workspace_files
from tools.llm_client import LLMClientError, generate_with_tools
from tools.prompt_manager import PromptManagerError, render_prompt
from tools.coder_tooling import TOOL_FUNCTIONS, TOOL_SCHEMAS
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
        system_prompt = render_prompt(
            "system_coder",
            {
                "task_title": task.title,
                "task_description": task.description,
                "task_constraints": task.constraints,
                "research_context": state.get("research_context", ""),
            },
        )
        base_user_prompt = (
            "Use the provided tools to inspect and edit the workspace so the task is completed."
            "Prioritize fixing the direct cause of the previous failure first, and avoid unrelated changes."
            "Call tools only when needed, and return only JSON once the work is complete."
            "The final output must always follow this format: "
            '{"files":{"main.py":"<updated python code>", "<other_changed_file>":"<updated text>"}}.'
            "Do not output explanations or Markdown."
            "When retrying after a failure, your first tool call must be read_file(main.py),"
            "and you must inspect the current main.py before using edit_file, create_file, or replace_string."
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
        )
        observations: list[str] = []
        for step in range(max_tool_steps):
            user_prompt = _build_user_prompt(
                base_user_prompt,
                observations,
                failure_context=failure_context,
            )
            response = generate_with_tools(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model_name,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.2,
                timeout=60,
            )

            print(f"LLM RESPONSE IS {response}")

            tool_calls = response.get("tool_calls") or []
            content = (response.get("content") or "").strip()

            if isinstance(tool_calls, list) and tool_calls:
                step_results: list[dict[str, Any]] = []
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        step_results.append(
                            {
                                "ok": False,
                                "error": "Tool call payload must be an object.",
                            }
                        )
                        continue
                    step_results.append(
                        _execute_tool_call(tool_call, run_paths, TOOL_FUNCTIONS)
                    )

                observations.append(
                    json.dumps(
                        {
                            "step": step + 1,
                            "tool_results": step_results,
                        },
                        ensure_ascii=False,
                    )
                )
                continue

            if content:
                parsed = _parse_coder_output(content)
                if parsed.files:
                    save_workspace_files(run_paths, parsed.files)

                workspace_state = _build_workspace_state_payload(
                    run_paths,
                    workspace_list_files,
                    workspace_read_file,
                    save_generated_code,
                )
                generated_files = workspace_state["generated_files"]
                return {
                    **workspace_state,
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
        )
        return {
            **workspace_state,
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
    except (LLMClientError, JSONDecodeError, ValidationError, ValueError) as exc:
        return {
            "status": "failed",
            "error": f"Coder parse/runtime error: {exc}",
            "messages": [
                AIMessage(
                    content=(
                        "Coder failed: LLM出力のJSONパース、ツール実行、"
                        "または結果検証に失敗しました。"
                    )
                )
            ],
        }


coder = coder_node
