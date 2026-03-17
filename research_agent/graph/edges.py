from __future__ import annotations

from typing import Literal

from graph.state import AgentState


def should_continue(state: AgentState) -> Literal["coder", "done"]:
    """
    Evaluatorノードの後に実行される条件付きエッジ。
    
    判定基準:
    - 実験が成功した、あるいはリトライ上限に達した場合は 'done' (終了) へ。
    - 実験が失敗し、かつリトライが可能（retry_countが規定値以下）な場合は 'coder' (再実装) へ。
    """
    
    # 1) evaluatorノードによって設定された次のステップ（next_step）を確認
    # v0.1では簡易的に、state内のstatusやcurrent_stepを見て判定する
    status = state.get("status")
    
    if status == "coding":
        # 失敗したがリトライ回数に余裕がある場合は Coder に戻る
        print(f"--- [Edge: should_continue] 実験失敗。Coderへ戻ります（現在リトライ数: {state.get('retry_count')}） ---")
        return "coder"
    
    # 成功 (completed) または 上限到達 (failed) の場合は終了
    print(f"--- [Edge: should_continue] 実行終了判定（ステータス: {status}） ---")
    return "done"
