from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AIMessage

from graph.state import AgentState
from schema.task import SubTask, Task


def _build_mock_subtasks(task: Task) -> List[SubTask]:
    """
    v0.1 用のモック分解ロジック。
    実運用のLLM分解ではなく、固定の3ステップを返す。
    """
    return [
        SubTask(
            title="サブタスク1: 先行研究の要点整理",
            description=(
                f"「{task.title}」に関連する背景情報を3点に要約し、"
                "次工程で使える前提知識を整理する。"
            ),
            assigned_agent="researcher",
            status="pending",
        ),
        SubTask(
            title="サブタスク2: 実験コードのひな形作成",
            description=(
                "要約した情報をもとに、最小実行可能なPythonコードを作成する。"
            ),
            assigned_agent="coder",
            status="pending",
        ),
        SubTask(
            title="サブタスク3: 実行結果の評価",
            description=(
                "生成コードの実行結果を確認し、成功/失敗と改善ポイントをまとめる。"
            ),
            assigned_agent="evaluator",
            status="pending",
        ),
    ]


def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Plannerノード（v0.1モック版）。

    役割:
    - `state["task"]` を受け取り、固定サブタスクを3つ作る
    - 更新した `task` を State に戻す
    - 次ノード（researcher）へ進むための最小情報を返す
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

    # 2) タスク分解（モック）
    # 元の task を直接破壊しないよう、深いコピーを作ってから更新する。
    planned_task = task.model_copy(deep=True)
    planned_task.subtasks = _build_mock_subtasks(planned_task)

    # 3) ノードの出力を返却
    # LangGraph 側で既存Stateにマージされ、次ノードに引き渡される。
    return {
        "task": planned_task,
        "status": "planning",
        "current_step": "researcher",
        "messages": [
            AIMessage(
                content=(
                    "Planner: モック分解を実行し、"
                    f"{len(planned_task.subtasks)} 件のサブタスクを作成しました。"
                )
            )
        ],
        "error": None,
    }


# 将来 `builder.add_node(\"planner\", planner)` のように使えるよう別名も用意
planner = planner_node
