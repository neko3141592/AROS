from __future__ import annotations

import json
import os
from json import JSONDecodeError
from typing import Any, Callable, Dict, Mapping

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from graph.state import AgentState
from schema.llm_outputs import CoderOutput
from tools.file_io import save_generated_code, save_workspace_files
from tools.llm_client import LLMClientError, generate_with_tools
from tools.prompt_manager import PromptManagerError, render_prompt
from tools.workspace_tools import (
    create_file as workspace_create_file,
    edit_file as workspace_edit_file,
    list_files as workspace_list_files,
    read_file as workspace_read_file,
    replace_string as workspace_replace_string,
)

DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_MAX_TOOL_STEPS = 8


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files under workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_dir": {"type": "string"},
                    "recursive": {"type": "boolean"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file under workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create or overwrite a file under workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                    "overwrite": {"type": "boolean"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace a line range in a file under workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "new_text": {"type": "string"},
                },
                "required": ["file_path", "start_line", "end_line", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_string",
            "description": "Replace string in a file under workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["file_path", "old", "new"],
            },
        },
    },
]


ToolFunction = Callable[..., Any]
TOOL_FUNCTIONS: Mapping[str, ToolFunction] = {
    "list_files": workspace_list_files,
    "read_file": workspace_read_file,
    "create_file": workspace_create_file,
    "edit_file": workspace_edit_file,
    "replace_string": workspace_replace_string,
}


def _strip_code_fence(text: str) -> str:
    """
    _strip_code_fence を実行する。
    
    Args:
        text: 処理対象のテキスト。
    """
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    lines = cleaned.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    return cleaned


def _extract_first_json_payload(text: str) -> Any:
    """
    _extract_first_json_payload を実行する。
    
    Args:
        text: 処理対象のテキスト。
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            return obj
        except JSONDecodeError:
            continue
    raise JSONDecodeError("No valid JSON payload found in text.", text, 0)


def _parse_coder_output(raw_text: str) -> CoderOutput:
    """
    _parse_coder_output を実行する。
    
    Args:
        raw_text: LLMから受け取った生テキスト。
    """
    cleaned = _strip_code_fence(raw_text)
    payload = _extract_first_json_payload(cleaned)

    if isinstance(payload, dict) and "files" not in payload:
        if all(isinstance(k, str) and isinstance(v, str) for k, v in payload.items()):
            payload = {"files": payload}

    return CoderOutput.model_validate(payload)


def _build_user_prompt(base_instruction: str, observations: list[str]) -> str:
    """
    _build_user_prompt を実行する。
    
    Args:
        base_instruction: Coder に渡す基本指示文。
        observations: ツール実行結果の履歴。
    """
    if not observations:
        return base_instruction
    return (
        f"{base_instruction}\n\n"
        "# Tool Execution History\n"
        "以下は直近のツール実行結果です。必要なら追加でツールを呼び出し、"
        "完了したら最終JSONのみ返してください。\n\n"
        + "\n\n".join(observations)
    )


def _parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    """
    _parse_tool_arguments を実行する。
    
    Args:
        raw_arguments: ツール呼び出し引数の生データ。
    """
    if raw_arguments is None or raw_arguments == "":
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        parsed = json.loads(raw_arguments)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Tool arguments must be a JSON object.")


def _execute_tool_call(tool_call: dict[str, Any], run_paths: Any) -> dict[str, Any]:
    """
    _execute_tool_call を実行する。
    
    Args:
        tool_call: LLMが返した単一のツール呼び出し情報。
        run_paths: 実行ディレクトリ群を保持する RunPaths。
    """
    call_id = str(tool_call.get("id", ""))
    fn = tool_call.get("function")
    if not isinstance(fn, dict):
        return {"id": call_id, "ok": False, "error": "Missing function payload."}

    name = fn.get("name")
    if not isinstance(name, str) or not name:
        return {"id": call_id, "ok": False, "error": "Missing function name."}

    if name not in TOOL_FUNCTIONS:
        return {"id": call_id, "name": name, "ok": False, "error": "Unknown tool name."}

    try:
        args = _parse_tool_arguments(fn.get("arguments"))
    except Exception as exc:  # noqa: BLE001
        return {
            "id": call_id,
            "name": name,
            "ok": False,
            "error": f"Invalid tool arguments: {exc}",
        }

    try:
        result = TOOL_FUNCTIONS[name](run_paths=run_paths, **args)
        return {"id": call_id, "name": name, "ok": True, "result": result}
    except Exception as exc:  # noqa: BLE001
        return {
            "id": call_id,
            "name": name,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _load_workspace_files(run_paths: Any) -> dict[str, str]:
    """
    _load_workspace_files を実行する。
    
    Args:
        run_paths: 実行ディレクトリ群を保持する RunPaths。
    """
    file_paths = workspace_list_files(run_paths=run_paths, base_dir=".", recursive=True)
    return {
        rel_path: workspace_read_file(run_paths=run_paths, file_path=rel_path)
        for rel_path in file_paths
    }


def _build_workspace_state_payload(run_paths: Any) -> dict[str, Any]:
    """
    workspace を正本として state へ返す payload を作る。
    generated_* は後方互換のために mirror として埋める。
    
    Args:
        run_paths: 実行ディレクトリ群を保持する RunPaths。
    """
    workspace_files = _load_workspace_files(run_paths)
    if "main.py" not in workspace_files:
        raise ValueError("workspace must contain main.py before evaluator step.")

    main_code = workspace_files["main.py"]
    save_generated_code(run_paths, main_code, filename="main.py")
    return {
        "generated_code": main_code,
        "generated_files": workspace_files,
    }


def coder_node(state: AgentState) -> Dict[str, Any]:
    """
    coder_node を実行する。
    
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
            "提供されたツールを使って workspace を探索・編集し、タスクを満たしてください。"
            "必要な場合のみツールを呼び出し、最終的に完了したら JSON のみ返してください。"
            "最終出力は必ず次の形式です: "
            '{"files":{"main.py":"<updated python code>", "<other_changed_file>":"<updated text>"}}。'
            "説明文やMarkdownは出力しないでください。"
        )

        model_name = (
            os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
        )
        max_tool_steps = int(
            os.getenv("MAX_TOOL_STEPS", str(DEFAULT_MAX_TOOL_STEPS)).strip()
            or str(DEFAULT_MAX_TOOL_STEPS)
        )
        if max_tool_steps < 1:
            raise ValueError("MAX_TOOL_STEPS must be >= 1.")

        observations: list[str] = []
        for step in range(max_tool_steps):
            user_prompt = _build_user_prompt(base_user_prompt, observations)
            response = generate_with_tools(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model_name,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.2,
                timeout=60,
            )

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
                    step_results.append(_execute_tool_call(tool_call, run_paths))

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

                workspace_state = _build_workspace_state_payload(run_paths)
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
        workspace_state = _build_workspace_state_payload(run_paths)
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
