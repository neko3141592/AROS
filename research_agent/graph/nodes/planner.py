from __future__ import annotations

import json
import os
from json import JSONDecodeError
from typing import Any, Dict, List

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from graph.state import AgentState
from schema.llm_outputs import PlannerOutput
from schema.task import SubTask
from tools.llm_client import LLMClientError, generate_text
from tools.prompt_manager import PromptManagerError, render_prompt

DEFAULT_MODEL_NAME = "gpt-4o-mini"


def _strip_code_fence(text: str) -> str:
    """
    LLMが返してきた ```json ... ``` を除去する。
    
    Args:
        text: 処理対象のテキスト。
    """
    s = text.strip()
    lines = s.splitlines()
    if not lines:
        return s

    if not lines[0].strip().startswith("```"):
        return s

    lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines).strip()


def _extract_first_json_payload(text: str) -> Any:
    """
    LLM出力から最初の JSON オブジェクト/配列を切り出して decode する。
    
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


def _parse_planner_output(raw_text: str) -> PlannerOutput:
    """
    _parse_planner_output を実行する。
    
    Args:
        raw_text: LLMから受け取った生テキスト。
    """
    cleaned = _strip_code_fence(raw_text)
    payload = _extract_first_json_payload(cleaned)

    # prompt仕様: JSON配列のみ想定だが、将来のため object も許容
    if isinstance(payload, list):
        payload = {"subtasks": payload}
    elif not isinstance(payload, dict):
        raise ValueError("Planner output JSON must be list or object.")

    return PlannerOutput.model_validate(payload)


def _to_subtasks(output: PlannerOutput) -> List[SubTask]:
    """
    _to_subtasks を実行する。
    
    Args:
        output: output に関する値。
    """
    return [
        SubTask(
            title=item.title,
            description=item.description,
            assigned_agent=item.assigned_agent,
            status=item.status,
        )
        for item in output.subtasks
    ]


def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    役割:
    - `state["task"]` を受け取り、LLMでサブタスクへ分解する
    - 更新した `task` を State に戻す
    - 次ノード（researcher）へ進むための最小情報を返す
    
    Args:
        state: ノード間で受け渡す現在の状態。
    """

    # 1) 入力チェック
    # State中心設計なので、task がない場合はここで安全に失敗させる。
    task = state.get("task")
    if task is None:
        return {
            "status": "failed",
            "error": "Planner: state['task'] が存在しません。",
            "messages": [
                AIMessage(
                    content="Planner failed: task が未設定のため、タスク分解を実行できませんでした。"
                )
            ],
        }

    # 2) プロンプトの生成とLLM呼び出し
    try:
        system_prompt = render_prompt(
            "system_planner",
            {
                "task_title": task.title,
                "task_description": task.description,
                "task_constraints": task.constraints,
            },
        )

        user_prompt = (
            "上記の指示に従い、サブタスク配列をJSONで返してください。"
            "説明文やMarkdownは含めないでください。"
        )

        model_name = (
            os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
        )
        llm_raw = generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model_name,
            temperature=0.2,
            timeout=30,
        )

        parsed = _parse_planner_output(llm_raw)
        subtasks = _to_subtasks(parsed)

        planned_task = task.model_copy(deep=True)
        planned_task.subtasks = subtasks

        return {
            "task": planned_task,
            "status": "planning",
            "current_step": "researcher",
            "messages": [
                AIMessage(
                    content=(
                        "Planner: LLM分解を実行し、"
                        f"{len(planned_task.subtasks)} 件のサブタスクを作成しました。"
                    )
                )
            ],
            "error": None,
        }

    except (JSONDecodeError, ValidationError, ValueError) as exc:
        return {
            "status": "failed",
            "error": f"Planner parse error: {exc}",
            "messages": [
                AIMessage(
                    content="Planner failed: LLM出力のJSONパースに失敗しました。"
                )
            ],
        }

    except (PromptManagerError, LLMClientError) as exc:
        return {
            "status": "failed",
            "error": f"Planner runtime error: {exc}",
            "messages": [
                AIMessage(
                    content="Planner failed: プロンプト生成またはLLM呼び出しに失敗しました。"
                )
            ],
        }


# 将来 `builder.add_node(\"planner\", planner)` のように使えるよう別名も用意
planner = planner_node
