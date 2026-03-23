from __future__ import annotations

from json import JSONDecodeError
from typing import Any, Dict

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from graph.state import AgentState
from tools.cli_logging import log_llm_request, log_llm_response, log_node_end, log_node_start
from tools.llm_client import LLMClientError, generate_text
from tools.model_config import get_model_name
from tools.planner_helpers import (
    _parse_planner_output,
    _to_subtasks,
)
from tools.prompt_manager import PromptManagerError, render_prompt

DEFAULT_MODEL_NAME = "gpt-4o-mini"


def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    役割:
    - `state["task"]` を受け取り、LLMでサブタスクへ分解する
    - 更新した `task` を State に戻す
    - 次ノード（researcher）へ進むための最小情報を返す

    Args:
        state: ノード間で受け渡す現在の状態。
    """

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

    try:
        log_node_start(
            "Planner",
            {
                "task_title": task.title,
                "constraints": len(task.constraints),
                "existing_subtasks": len(task.subtasks),
            },
        )
        system_prompt = render_prompt(
            "system_planner",
            {
                "task_title": task.title,
                "task_description": task.description,
                "task_constraints": task.constraints,
            },
        )

        user_prompt = (
            "Follow the instructions above and return the subtask array as JSON."
            "Do not include explanations or Markdown."
        )

        model_name = get_model_name("PLANNER_MODEL_NAME", DEFAULT_MODEL_NAME)
        log_llm_request(
            "Planner",
            model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        llm_raw = generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model_name,
            temperature=0.2,
            timeout=30,
        )
        log_llm_response("Planner", llm_raw)

        parsed = _parse_planner_output(llm_raw)
        subtasks = _to_subtasks(parsed)

        planned_task = task.model_copy(deep=True)
        planned_task.subtasks = subtasks

        log_node_end(
            "Planner",
            {
                "subtask_count": len(planned_task.subtasks),
                "next_step": "researcher",
            },
        )
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


# 将来 `builder.add_node("planner", planner)` のように使えるよう別名も用意
planner = planner_node
