from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage
from schema.task import ExperimentResult
from tools.file_io import write_execution_log, write_meta
from tools.local_executor import run_workspace_python
from graph.state import AgentState


def evaluator_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluatorノード（v0.2 ローカル実行版）。

    役割:
    - Coderが生成したコードを workspace で実行して評価する。
    - 成功か失敗かを判定し、失敗の場合はリトライを促す。
    - リトライ回数（retry_count）を管理し、上限を超えたら強制終了させる。
    """
    print("--- [Node: Evaluator] 実験結果を評価しています... ---")

    # 1) 入力確認
    task = state.get("task")
    generated_files = state.get("generated_files") or {}
    run_paths = state.get("run_paths")
    retry_count = state.get("retry_count", 0)

    if not task:
        return {
            "status": "failed",
            "error": "Evaluator: 評価対象の task が存在しません。",
            "messages": [AIMessage(content="Evaluator failed: Missing task.")],
        }

    if not run_paths:
        return {
            "status": "failed",
            "error": "Evaluator: run_paths が存在しません。",
            "messages": [AIMessage(content="Evaluator failed: Missing run_paths.")],
        }

    # 2) ローカル実行
    execution = run_workspace_python(
        run_paths=run_paths,
        entrypoint="main.py",
        timeout_sec=60,
    )
    is_success = execution.returncode == 0

    execution_log = (
        "=== Local Execution ===\n"
        "Command: python main.py\n"
        f"Return Code: {execution.returncode}\n\n"
        "=== STDOUT ===\n"
        f"{execution.stdout.rstrip()}\n\n"
        "=== STDERR ===\n"
        f"{execution.stderr.rstrip()}\n"
    )

    # 3) ExperimentResult の作成
    if is_success:
        error_message = None
    else:
        error_message = execution.stderr.strip()
        if not error_message:
            error_message = f"Process exited with code {execution.returncode}."

    result = ExperimentResult(
        task_id=task.id,
        success=is_success,
        metrics={},
        logs=execution_log,
        error_message=error_message,
    )

    # 4. 実行ログの保存
    write_execution_log(run_paths, execution_log, append=True)

    # 5. メタ情報の保存
    if isinstance(generated_files, dict) and generated_files:
        file_list = sorted(generated_files.keys())
    else:
        # generated_files が空でも workspace の実ファイルを優先して記録する
        file_list = sorted(
            str(path.relative_to(run_paths.workspace_dir))
            for path in run_paths.workspace_dir.rglob("*")
            if path.is_file()
        )
        if not file_list and (run_paths.workspace_dir / "main.py").exists():
            file_list = ["main.py"]
    write_meta(run_paths, task.id, file_list)

    # 6. ステータスとリトライの判定
    # 成功したか、リトライ上限（例：3回）に達したか
    if is_success:
        new_status = "completed"
        next_step = "done"
        next_retry_count = retry_count
        message_content = "Evaluator: 実験は正常に完了しました。成功としてマークします。"
    elif retry_count >= 2:  # 0, 1, 2 の 3回目で終了
        new_status = "failed"
        next_step = "done"
        next_retry_count = retry_count + 1
        message_content = f"Evaluator: リトライ上限（{retry_count + 1}回）に達しました。実行を停止します。"
    else:
        new_status = "coding"  # 失敗したがリトライ可能なら Coder に戻す
        next_step = "coder"
        next_retry_count = retry_count + 1
        message_content = f"Evaluator: 実行エラーを検知（試行 {retry_count + 1}回目）。Coder に修正を依頼します。"

    # 7) Stateの更新
    return {
        "result": result,
        "execution_logs": execution_log,
        "execution_stdout": execution.stdout,
        "execution_stderr": execution.stderr,
        "execution_return_code": execution.returncode,
        "status": new_status,
        "current_step": next_step,
        "retry_count": next_retry_count,
        "error": None if is_success else error_message,
        "messages": [AIMessage(content=message_content)],
    }


# エイリアス設定
evaluator = evaluator_node
