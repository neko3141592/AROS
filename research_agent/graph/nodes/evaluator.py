from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage
from dataclasses import dataclass
from schema.task import ExperimentResult
from tools.file_io import write_execution_log, write_meta
from tools.local_executor import LocalExecutionResult, run_workspace_python
from graph.state import AgentState, EvaluatorFeedback

import hashlib
import re

@dataclass
class FailureClassification:
    summary: str
    likely_cause: str
    suggested_fixes: list[str]
    can_self_fix: bool
    needs_research: bool


def _collect_workspace_file_list(run_paths: Any) -> list[str]:
    """
    _collect_workspace_file_list を実行する。
    
    Args:
        run_paths: 実行ディレクトリ群を保持する RunPaths。
    """
    return sorted(
        str(path.relative_to(run_paths.workspace_dir).as_posix())
        for path in run_paths.workspace_dir.rglob("*")
        if path.is_file()
    )

def _extract_exception_line(stderr: str) -> str:
    """
    Tracebackの最終例外行を返す。見つからない場合は空白を返す
    """
    for line in reversed(stderr.splitlines()):
        line = line.strip()
        if line and not line.startswith(" ") and ":" in line:
            return line
    return ""

def _classify_failure(stderr: str, returncode: int) -> FailureClassification:
    """
    stderrとreturncodeから失敗を分類し、FailureClassificationを返す。
    将来的にruntime_error_unknown の発生率が高くなったら、LLMエスカレーションを検討。
    """
    exception_line = _extract_exception_line(stderr)

    if returncode == 124 or "timeout" in stderr.lower():
        return FailureClassification(
            summary="timeout",
            likely_cause="Execution exceeded the time limit.",
            suggested_fixes=["Check for infinite loops", "Reduce workload", "Increase timeout threshold"],
            can_self_fix=True,
            needs_research=False,
        )

    def starts(prefix: str) -> bool:
        return exception_line.startswith(prefix)

    if starts("ModuleNotFoundError"):
        module = exception_line.split("'")[1] if "'" in exception_line else "unknown"
        return FailureClassification(
            summary="missing_module",
            likely_cause=f"Module '{module}' is not installed.",
            suggested_fixes=[f"pip install {module}", "Check virtual environment", "Check for typos in module name"],
            can_self_fix=False,
            needs_research=True,
        )

    if starts("SyntaxError") or starts("IndentationError"):
        return FailureClassification(
            summary="syntax_error",
            likely_cause="Invalid syntax or indentation.",
            suggested_fixes=["Unify indentation (4 spaces recommended)", "Check matching brackets and quotes"],
            can_self_fix=True,
            needs_research=False,
        )

    if starts("NameError") or starts("AttributeError") or starts("TypeError"):
        return FailureClassification(
            summary="name_or_type_error",
            likely_cause="Mismatched variable name, attribute, or type.",
            suggested_fixes=["Check for typos in variable names", "Verify types (e.g. str vs int)", "Check object is not None"],
            can_self_fix=True,
            needs_research=False,
        )

    return FailureClassification(
        summary="runtime_error_unknown",
        likely_cause=f"Unclassified runtime error: {exception_line or 'no details'}",
        suggested_fixes=["Inspect stderr stack trace", "Reproduce and isolate the issue"],
        can_self_fix=False,
        needs_research=True,
    )

def _build_error_signature(stderr: str, returncode: int) -> str:
    """
    例外種別・代表メッセージ・returncodeを正規化してSHA256ハッシュを生成する。
    同一エラーの反復検知に使用。可変値（パス・行番号）は除去して揺れを吸収する。
    """
    exception_line = _extract_exception_line(stderr)

    normalized = exception_line
    # /path/to/file.py 形式のパスを除去
    normalized = re.sub(r'[\w/\\.-]+\.py', '<file>', normalized)
    # 行番号 (line 42) を除去
    normalized = re.sub(r'line \d+', 'line <N>', normalized)
    # 数字の連続を除去（ポート番号・アドレス等）
    normalized = re.sub(r'\b\d+\b', '<N>', normalized)
    # クォート内の文字列（パス・変数名等）を除去
    normalized = re.sub(r"'[^']*'", "'<V>'", normalized)
    # 余分な空白を正規化
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    raw = f"{normalized}|rc={returncode}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _build_evaluator_feedback(execution: LocalExecutionResult) -> EvaluatorFeedback:
    """
    実行結果からCoderへ渡す構造化フィードバックを組み立てる。

    Args:
        execution: ローカル実行の結果。
    """
    if execution.returncode == 0:
        return {
            "summary": "success",
            "likely_cause": "Execution completed successfully.",
            "suggested_fixes": [],
            "can_self_fix": False,
            "needs_research": False,
            "return_code": execution.returncode,
            "stdout": execution.stdout,
            "stderr": execution.stderr,
            "raw": {},
        }

    classification = _classify_failure(execution.stderr, execution.returncode)
    exception_line = _extract_exception_line(execution.stderr)
    return {
        "summary": classification.summary,
        "likely_cause": classification.likely_cause,
        "suggested_fixes": classification.suggested_fixes,
        "can_self_fix": classification.can_self_fix,
        "needs_research": classification.needs_research,
        "return_code": execution.returncode,
        "stdout": execution.stdout,
        "stderr": execution.stderr,
        "raw": {
            "exception_line": exception_line,
            "returncode": execution.returncode,
        },
    }


def _build_feedback_log_section(feedback: EvaluatorFeedback) -> str:
    """
    execution_log に追記する FEEDBACK セクションを生成する。

    Args:
        feedback: Evaluator の構造化フィードバック。
    """
    fixes = feedback.get("suggested_fixes") or []
    if fixes:
        fixes_text = "\n".join(f"- {item}" for item in fixes)
    else:
        fixes_text = "- (none)"

    return (
        "=== FEEDBACK ===\n"
        f"summary: {feedback.get('summary', '')}\n"
        f"likely_cause: {feedback.get('likely_cause', '')}\n"
        f"can_self_fix: {feedback.get('can_self_fix', False)}\n"
        f"needs_research: {feedback.get('needs_research', False)}\n"
        "suggested_fixes:\n"
        f"{fixes_text}\n"
    )


def evaluator_node(state: AgentState) -> Dict[str, Any]:
    """
    Evaluatorノード（v0.2 ローカル実行版）。
    
    役割:
    - Coderが生成したコードを workspace で実行して評価する。
    - 成功か失敗かを判定し、失敗の場合はリトライを促す。
    - リトライ回数（retry_count）を管理し、上限を超えたら強制終了させる。
    
    Args:
        state: ノード間で受け渡す現在の状態。
    """
    print("--- [Node: Evaluator] 実験結果を評価しています... ---")

    # 1) 入力確認
    task = state.get("task")
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
    feedback = _build_evaluator_feedback(execution)

    if is_success:
        current_error_signature = None
        next_same_error_count = 0
    else:
        current_error_signature = _build_error_signature(
            execution.stderr,
            execution.returncode,
        )
        prev_error_signature = state.get("error_signature")
        prev_same_error_count = state.get("same_error_count", 0)
        if prev_error_signature == current_error_signature:
            next_same_error_count = prev_same_error_count + 1
        else:
            next_same_error_count = 1

    execution_log = (
        "=== Local Execution ===\n"
        "Command: python main.py\n"
        f"Return Code: {execution.returncode}\n\n"
        "=== STDOUT ===\n"
        f"{execution.stdout.rstrip()}\n\n"
        "=== STDERR ===\n"
        f"{execution.stderr.rstrip()}\n\n"
        f"{_build_feedback_log_section(feedback)}"
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
    # state上の generated_files ではなく、workspace 実体を正本として記録する
    file_list = _collect_workspace_file_list(run_paths)
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
        message_content = (
            f"{feedback['summary']}: "
            f"リトライ上限（{retry_count + 1}回）に達しました。実行を停止します。"
        )
    elif feedback.get("needs_research") or not feedback.get("can_self_fix", False):
        new_status = "researching"
        next_step = "researcher"
        next_retry_count = retry_count + 1
        message_content = (
            f"{feedback['summary']}: "
            f"試行 {retry_count + 1}回目で失敗。Researcher に追加調査を依頼します。"
        )
    else:
        new_status = "coding"  # 失敗したがリトライ可能なら Coder に戻す
        next_step = "coder"
        next_retry_count = retry_count + 1
        message_content = (
            f"{feedback['summary']}: "
            f"試行 {retry_count + 1}回目で失敗。Coder に修正を依頼します。"
        )

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
        "evaluator_feedback": feedback,
        "error_signature": current_error_signature,
        "same_error_count": next_same_error_count,
        "stop_reason": None,
        "error": None if is_success else error_message,
        "messages": [AIMessage(content=message_content)],
    }


# エイリアス設定
evaluator = evaluator_node
