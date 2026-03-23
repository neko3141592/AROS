from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any, Callable, Mapping

from schema.llm_outputs import CoderOutput

ToolFunction = Callable[..., Any]


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


def _build_failure_context(
    evaluator_feedback: Mapping[str, Any] | None,
    execution_stderr: str | None,
    retry_count: int,
) -> str:
    """
    再試行時に Coder へ渡す失敗コンテキストを組み立てる。

    Args:
        evaluator_feedback: Evaluator の構造化フィードバック。
        execution_stderr: 実行時の標準エラー出力。
        retry_count: 現在のリトライ回数。
    """
    if retry_count <= 0:
        return ""

    feedback = evaluator_feedback or {}
    summary = str(feedback.get("summary") or "unknown_failure")
    likely_cause = str(feedback.get("likely_cause") or "")
    suggested_fixes = feedback.get("suggested_fixes") or []
    fixes_text = (
        "\n".join(f"- {item}" for item in suggested_fixes)
        if suggested_fixes
        else "- (none)"
    )
    stderr_text = (execution_stderr or str(feedback.get("stderr") or "")).rstrip() or "(empty)"

    return (
        f"summary: {summary}\n"
        "# Previous Failure Context\n"
        "Prioritize fixing the direct cause of the previous failure and avoid unrelated changes.\n"
        "On retries, the first tool call must be read_file(main.py), and you must inspect the current main.py before editing.\n"
        f"likely_cause: {likely_cause}\n"
        "suggested_fixes:\n"
        f"{fixes_text}\n"
        "stderr:\n"
        f"{stderr_text}"
    )


def _build_user_prompt(
    base_instruction: str,
    observations: list[str],
    failure_context: str = "",
) -> str:
    """
    _build_user_prompt を実行する。

    Args:
        base_instruction: Coder に渡す基本指示文。
        observations: ツール実行結果の履歴。
        failure_context: 再試行時に優先すべき失敗コンテキスト。
    """
    prompt_parts: list[str] = []
    if failure_context:
        prompt_parts.append(failure_context)
    prompt_parts.append(base_instruction)
    if observations:
        prompt_parts.append(
            "# Tool Execution History\n"
            "Below are the most recent tool execution results. Call additional tools if needed, and return only the final JSON when you are done.\n\n"
            + "\n\n".join(observations)
        )
    return "\n\n".join(prompt_parts)


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


def _execute_tool_call(
    tool_call: dict[str, Any],
    run_paths: Any,
    tool_functions: Mapping[str, ToolFunction],
) -> dict[str, Any]:
    """
    _execute_tool_call を実行する。

    Args:
        tool_call: LLMが返した単一のツール呼び出し情報。
        run_paths: 実行ディレクトリ群を保持する RunPaths。
        tool_functions: 利用可能なツール関数マップ。
    """
    call_id = str(tool_call.get("id", ""))
    fn = tool_call.get("function")
    if not isinstance(fn, dict):
        return {"id": call_id, "ok": False, "error": "Missing function payload."}

    name = fn.get("name")
    if not isinstance(name, str) or not name:
        return {"id": call_id, "ok": False, "error": "Missing function name."}

    if name not in tool_functions:
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
        result = tool_functions[name](run_paths=run_paths, **args)
        return {"id": call_id, "name": name, "ok": True, "result": result}
    except Exception as exc:  # noqa: BLE001
        return {
            "id": call_id,
            "name": name,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _load_workspace_files(
    run_paths: Any,
    list_files_fn: Callable[..., list[str]],
    read_file_fn: Callable[..., str],
) -> dict[str, str]:
    """
    _load_workspace_files を実行する。

    Args:
        run_paths: 実行ディレクトリ群を保持する RunPaths。
        list_files_fn: workspace内ファイル一覧関数。
        read_file_fn: workspace内ファイル読取関数。
    """
    file_paths = list_files_fn(run_paths=run_paths, base_dir=".", recursive=True)
    return {
        rel_path: read_file_fn(run_paths=run_paths, file_path=rel_path)
        for rel_path in file_paths
    }


def _build_workspace_state_payload(
    run_paths: Any,
    list_files_fn: Callable[..., list[str]],
    read_file_fn: Callable[..., str],
    save_generated_code_fn: Callable[..., Any],
) -> dict[str, Any]:
    """
    workspace を正本として state へ返す payload を作る。
    generated_* は後方互換のために state mirror として埋める。

    Args:
        run_paths: 実行ディレクトリ群を保持する RunPaths。
        list_files_fn: workspace内ファイル一覧関数。
        read_file_fn: workspace内ファイル読取関数。
        save_generated_code_fn: 生成コード保存関数。
    """
    workspace_files = _load_workspace_files(run_paths, list_files_fn, read_file_fn)
    if "main.py" not in workspace_files:
        raise ValueError("workspace must contain main.py before evaluator step.")

    main_code = workspace_files["main.py"]
    save_generated_code_fn(run_paths, main_code, filename="main.py")
    return {
        "generated_code": main_code,
        "generated_files": workspace_files,
    }
