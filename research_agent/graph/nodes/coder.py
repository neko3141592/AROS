from __future__ import annotations

import json
import os
from json import JSONDecodeError
from typing import Any, Dict

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from graph.state import AgentState
from tools.prompt_manager import PromptManagerError, render_prompt
from tools.llm_client import LLMClientError, generate_text
from tools.file_io import save_generated_code, save_workspace_files
from schema.llm_outputs import CoderOutput

DEFAULT_MODEL_NAME = "gpt-4o-mini"


def _strip_code_fence(text: str) -> str:
    """
    LLM出力にコードフェンスが含まれる場合に除去する。
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
    テキスト中の先頭JSONオブジェクト/配列を抽出する。
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
    cleaned = _strip_code_fence(raw_text)
    payload = _extract_first_json_payload(cleaned)

    # {"files": {...}} が正規形だが、将来の差分に備えて {...} 直下も受ける。
    if isinstance(payload, dict) and "files" not in payload:
        if all(isinstance(k, str) and isinstance(v, str) for k, v in payload.items()):
            payload = {"files": payload}

    output = CoderOutput.model_validate(payload)
    if not output.files:
        raise ValueError("Coder output 'files' is empty.")
    if "main.py" not in output.files:
        raise ValueError("Coder output must include 'main.py'.")
    return output


def coder_node(state: AgentState) -> Dict[str, Any]:
    """
    Coderノード（v0.2 LLM版）。
    """

    # 1. 入力チェック
    # task がない場合は以降の処理が不可能なため、明示的に failed を返す。
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

    # 2. プロンプト生成 + LLM呼び出し
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
        user_prompt = (
            "上記の指示に従い、JSONオブジェクトのみ返してください。"
            "必ず次の形式を守ってください: "
            '{"files":{"main.py":"<python code>", "README.md":"<text>"}}。'
            "Markdownや説明文は含めないでください。"
        )

        model_name = (
            os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
        )
        llm_raw = generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model_name,
            temperature=0.2,
            timeout=60,
        )
        parsed = _parse_coder_output(llm_raw)
        generated_files = parsed.files
        generated_code = generated_files["main.py"]

        if run_paths:
            save_workspace_files(run_paths, generated_files)
            save_generated_code(run_paths, generated_code, filename="main.py")

        return {
            "generated_code": generated_code,
            "generated_files": generated_files,
            "status": "coding",
            "current_step": "evaluator",
            "messages": [
                AIMessage(
                    content=(
                        "Coder: LLMで複数ファイルを生成し、"
                        f"{len(generated_files)}件をworkspaceに保存しました。"
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
                        "Coder failed: LLM出力のJSONパースまたは検証に失敗しました。"
                    )
                )
            ],
        }


# 将来 `builder.add_node("coder", coder)` のように使えるよう別名も用意
coder = coder_node
