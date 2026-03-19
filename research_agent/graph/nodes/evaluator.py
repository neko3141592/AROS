from __future__ import annotations

import random
from typing import Any, Dict

from langchain_core.messages import AIMessage
from schema.task import ExperimentResult
from tools.file_io import write_execution_log, write_meta
from graph.state import AgentState


def evaluator_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluatorノード（v0.1モック版）。

    役割:
    - Coderが生成したコードの実行結果（v0.1ではダミー）を評価する。
    - 成功か失敗かを判定し、失敗の場合はリトライを促す。
    - リトライ回数（retry_count）を管理し、上限を超えたら強制終了させる。
    """
    print("--- [Node: Evaluator] 実験結果を評価しています... ---")

    # 1) 入力確認
    task = state.get("task")
    generated_code = state.get("generated_code")
    generated_files = state.get("generated_files") or {}
    retry_count = state.get("retry_count", 0)

    if not generated_code and isinstance(generated_files, dict):
        generated_code = generated_files.get("main.py")

    if not task or not generated_code:
        return {
            "status": "failed",
            "error": "Evaluator: 評価対象のタスクまたはコードが存在しません。",
            "messages": [AIMessage(content="Evaluator failed: Missing task or code.")],
        }

    # 2) 実行シミュレーション (v0.1 モック)
    # 本来はここで RunPod 等で実行したログを解析する。
    # v0.1 ではランダム、または固定の成功/失敗をシミュレート。
    
    # 成功率 70% のシミュレーション
    is_success = random.random() < 0.7
    
    mock_logs = (
        "Epoch 1: loss=0.5, acc=0.8\n"
        "Epoch 2: loss=0.3, acc=0.9\n"
        "Optimization finished successfully." if is_success else "Traceback: ZeroDivisionError at line 42"
    )

    # 3) ExperimentResult の作成
    result = ExperimentResult(
        task_id=task.id,
        success=is_success,
        metrics={"accuracy": 0.9} if is_success else {},
        logs=mock_logs,
        error_message=None if is_success else "Simulation error for testing retry loop."
    )

    # 4. 実行ログの保存
    run_paths = state.get("run_paths")

    if run_paths:
        write_execution_log(run_paths, mock_logs, append=True)
        
    # 5. メタ情報の保存
    if run_paths:
        if isinstance(generated_files, dict) and generated_files:
            file_list = sorted(generated_files.keys())
        else:
            file_list = ["main.py"]
        write_meta(run_paths, task.id, file_list)

    # 5. ステータスとリトライの判定
    # 成功したか、リトライ上限（例：3回）に達したか
    if is_success:
        new_status = "completed"
        next_step = "done"
        message_content = "Evaluator: 実験は正常に完了しました。成功としてマークします。"
    elif retry_count >= 2:  # 0, 1, 2 の 3回目で終了
        new_status = "failed"
        next_step = "done"
        message_content = f"Evaluator: リトライ上限（{retry_count + 1}回）に達しました。実行を停止します。"
    else:
        new_status = "coding"  # 失敗したがリトライ可能なら Coder に戻す
        next_step = "coder"
        message_content = f"Evaluator: 実行エラーを検知（試行 {retry_count + 1}回目）。Coder に修正を依頼します。"

    # 5) Stateの更新
    return {
        "result": result,
        "execution_logs": mock_logs,
        "status": new_status,
        "current_step": next_step,
        "retry_count": retry_count + 1,
        "messages": [AIMessage(content=message_content)],
    }


# エイリアス設定
evaluator = evaluator_node
