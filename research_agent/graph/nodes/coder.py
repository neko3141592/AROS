from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage

from graph.state import AgentState


def _build_mock_code(task_title: str) -> str:
    """
    v0.1 用の固定コード生成ロジック。
    仕様どおり「Hello World」を返す。
    """
    return (
        f"# Mock code for task: {task_title}\n"
        "def main() -> None:\n"
        "    print(\"Hello World\")\n"
        "\n"
        "if __name__ == \"__main__\":\n"
        "    main()\n"
    )


def coder_node(state: AgentState) -> Dict[str, Any]:
    """
    Coderノード（v0.1モック版）。

    役割:
    - 現在の task 情報から固定コードを作る
    - `generated_code` に保存して次ノード（evaluator）へ渡す
    """

    # 1) 入力チェック
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

    # 2) モックコード生成
    # v0.1 では実LLM生成を行わず、固定の Hello World コードを生成する。
    generated_code = _build_mock_code(task_title=task.title)

    # 3) State更新内容を返却
    # LangGraph が既存Stateにマージし、次ノードで利用可能になる。
    return {
        "generated_code": generated_code,
        "status": "coding",
        "current_step": "evaluator",
        "messages": [
            AIMessage(
                content=(
                    "Coder: v0.1モックとして Hello World コードを生成しました。"
                )
            )
        ],
        "error": None,
    }


# 将来 `builder.add_node("coder", coder)` のように使えるよう別名も用意
coder = coder_node
