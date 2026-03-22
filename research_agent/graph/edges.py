from __future__ import annotations

from typing import Literal

from graph.state import AgentState


def should_continue(state: AgentState) -> Literal["done", "coder", "researcher"]:
    """
    Evaluatorノードの後に実行される条件付きエッジ。
    
    判定基準:
    - 実験が成功した、あるいはリトライ上限に達した場合は 'done' (終了) へ。
    - 実験が失敗し、自己修正可能なら 'coder' (再実装) へ。
    - 実験が失敗し、追加調査が必要なら 'researcher' へ。
    
    Args:
        state: ノード間で受け渡す現在の状態。
    """
    
    current_step = state.get("current_step")
    status = state.get("status")

    if current_step == "researcher" or status == "researching":
        print(
            "--- [Edge: should_continue] 実験失敗。"
            f"Researcherへ戻ります（現在リトライ数: {state.get('retry_count')}） ---"
        )
        return "researcher"

    if current_step == "coder" or status == "coding":
        print(
            "--- [Edge: should_continue] 実験失敗。"
            f"Coderへ戻ります（現在リトライ数: {state.get('retry_count')}） ---"
        )
        return "coder"

    # 成功 (completed) または 上限到達 (failed) の場合は終了
    print(
        "--- [Edge: should_continue] 実行終了判定"
        f"（ステータス: {status}, current_step: {current_step}） ---"
    )
    return "done"
