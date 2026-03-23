from __future__ import annotations

import os
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Dict

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from graph.state import AgentState, EvaluatorFeedback
from schema.llm_outputs import EvaluatorAnalysisOutput
from schema.task import ExperimentResult, Task
from tools.evaluator_helpers import (
    LLM_STDERR_MAX_CHARS,
    LLM_STDOUT_MAX_CHARS,
    _build_error_signature,
    _classify_failure,
    _extract_exception_line,
    _parse_evaluator_analysis_output,
    _read_positive_float_env,
    _sanitize_execution_outputs_for_llm,
    _should_use_evaluator_llm_analysis,
    calculate_next_timeout,
)
from tools.file_io import write_execution_log, write_meta
from tools.llm_client import LLMClientError, generate_text
from tools.local_executor import LocalExecutionResult, run_workspace_python
from tools.model_config import get_model_name
from tools.prompt_manager import PromptManagerError, render_prompt

DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_EXECUTION_TIMEOUT_SEC = 60.0
DEFAULT_MAX_TOTAL_EXECUTION_TIME_SEC = 180.0
DEFAULT_LLM_ANALYSIS_TIMEOUT_SEC = 20.0
# retry_count は 0 始まりで、3回目の失敗で停止する。
MAX_SAME_ERROR_COUNT = 3
MAX_RETRY_COUNT = 2


@dataclass
class EvaluatorDecision:
    status: str
    next_step: str
    stop_reason: str | None
    message: str


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


def _merge_feedback_with_llm_analysis(
    base_feedback: EvaluatorFeedback,
    analysis: EvaluatorAnalysisOutput,
) -> EvaluatorFeedback:
    """
    LLM 解析結果でベースフィードバックを補強する。

    Args:
        base_feedback: ヒューリスティック由来のベースフィードバック。
        analysis: LLM 解析結果。
    """
    summary = str(base_feedback.get("summary") or "")
    base_can_self_fix = bool(base_feedback.get("can_self_fix", False))
    base_needs_research = bool(base_feedback.get("needs_research", False))

    # For well-understood heuristic classes, keep routing control flags stable.
    if summary in {
        "missing_module",
        "syntax_error",
        "name_or_type_error",
        "timeout",
    }:
        can_self_fix = base_can_self_fix
        needs_research = base_needs_research
    else:
        # Otherwise only allow the LLM to move toward a more conservative route.
        can_self_fix = base_can_self_fix and analysis.can_self_fix
        needs_research = base_needs_research or analysis.needs_research

    return {
        **base_feedback,
        "likely_cause": analysis.likely_cause,
        "suggested_fixes": analysis.suggested_fixes,
        "can_self_fix": can_self_fix,
        "needs_research": needs_research,
    }


def _analyze_failure_with_llm(
    task: Task,
    execution: LocalExecutionResult,
    base_feedback: EvaluatorFeedback,
) -> EvaluatorFeedback:
    """
    失敗解析を LLM で補強し、失敗時はベースフィードバックへフォールバックする。

    Args:
        task: 現在のタスク。
        execution: ローカル実行結果。
        base_feedback: ヒューリスティック分類結果。
    """
    if not _should_use_evaluator_llm_analysis():
        return base_feedback

    sanitized_stdout, sanitized_stderr = _sanitize_execution_outputs_for_llm(
        execution.stdout,
        execution.stderr,
    )

    try:
        system_prompt = render_prompt(
            "system_evaluator",
            {
                "task_title": task.title,
                "task_description": task.description,
                "return_code": execution.returncode,
                "stdout": sanitized_stdout,
                "stderr": sanitized_stderr,
                "base_summary": base_feedback.get("summary", ""),
                "base_likely_cause": base_feedback.get("likely_cause", ""),
                "base_suggested_fixes": base_feedback.get("suggested_fixes", []),
            },
        )
        evaluator_model = get_model_name(
            "EVALUATOR_MODEL_NAME",
            DEFAULT_MODEL_NAME,
        )
        raw_text = generate_text(
            system_prompt=system_prompt,
            user_prompt="Return only the JSON object.",
            model=evaluator_model,
            temperature=0.0,
            timeout=DEFAULT_LLM_ANALYSIS_TIMEOUT_SEC,
        )
        parsed = _parse_evaluator_analysis_output(raw_text)
        return _merge_feedback_with_llm_analysis(base_feedback, parsed)
    except (
        PromptManagerError,
        LLMClientError,
        JSONDecodeError,
        ValidationError,
        ValueError,
    ):
        return base_feedback


def _build_evaluator_feedback(
    task: Task,
    execution: LocalExecutionResult,
) -> EvaluatorFeedback:
    """
    実行結果からCoderへ渡す構造化フィードバックを組み立てる。

    Args:
        task: 現在のタスク。
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
    feedback: EvaluatorFeedback = {
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
    return _analyze_failure_with_llm(
        task=task,
        execution=execution,
        base_feedback=feedback,
    )


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


def _decide_next_action(
    feedback: EvaluatorFeedback,
    retry_count: int,
    same_error_count: int,
    total_execution_duration_sec: float,
    max_total_execution_time_sec: float,
) -> EvaluatorDecision:
    if feedback["summary"] == "success":
        return EvaluatorDecision(
            "completed",
            "done",
            None,
            "Evaluator: 実験は正常に完了しました。",
        )

    if total_execution_duration_sec >= max_total_execution_time_sec:
        return EvaluatorDecision(
            "failed",
            "done",
            "total_timeout",
            (
                f"{feedback['summary']}: 累積実行時間が "
                f"{max_total_execution_time_sec:.1f}s を超えたため停止します。"
            ),
        )

    if same_error_count >= MAX_SAME_ERROR_COUNT:
        return EvaluatorDecision(
            "failed",
            "done",
            "repeated_error",
            f"{feedback['summary']}: 同一エラーが {same_error_count} 回連続で発生したため停止します。",
        )

    if retry_count >= MAX_RETRY_COUNT:
        return EvaluatorDecision(
            "failed",
            "done",
            "max_retry",
            f"{feedback['summary']}: リトライ上限（{retry_count + 1}回）に達しました。",
        )

    if feedback.get("needs_research") or not feedback.get("can_self_fix", False):
        return EvaluatorDecision(
            "researching",
            "researcher",
            None,
            f"{feedback['summary']}: 試行 {retry_count + 1}回目で失敗。Researcher に依頼します。",
        )

    return EvaluatorDecision(
        "coding",
        "coder",
        None,
        f"{feedback['summary']}: 試行 {retry_count + 1}回目で失敗。Coder に修正を依頼します。",
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
    previous_total_execution_duration_sec = float(
        state.get("total_execution_duration_sec") or 0.0
    )

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

    execution_timeout_sec = _read_positive_float_env(
        "EXECUTION_TIMEOUT_SEC",
        DEFAULT_EXECUTION_TIMEOUT_SEC,
    )
    max_total_execution_time_sec = _read_positive_float_env(
        "MAX_TOTAL_EXECUTION_TIME_SEC",
        DEFAULT_MAX_TOTAL_EXECUTION_TIME_SEC,
    )
    next_execution_timeout_sec = calculate_next_timeout(
        per_try_timeout=execution_timeout_sec,
        total_max_timeout=max_total_execution_time_sec,
        current_total_used=previous_total_execution_duration_sec,
    )

    if next_execution_timeout_sec <= 0:
        timeout_message = (
            "timeout: 累積実行時間の残り予算がないため、実行せずに停止します。"
        )
        feedback: EvaluatorFeedback = {
            "summary": "timeout",
            "likely_cause": (
                "Total execution budget was exhausted before starting the next run."
            ),
            "suggested_fixes": [
                "Reduce workload per trial",
                "Increase MAX_TOTAL_EXECUTION_TIME_SEC if appropriate",
            ],
            "can_self_fix": False,
            "needs_research": False,
            "return_code": -1,
            "stdout": "",
            "stderr": "Skipped execution because no total timeout budget remained.",
            "raw": {
                "max_total_execution_time_sec": max_total_execution_time_sec,
                "previous_total_execution_duration_sec": (
                    previous_total_execution_duration_sec
                ),
                "applied_timeout_sec": next_execution_timeout_sec,
            },
        }
        execution_log = (
            "=== Local Execution ===\n"
            "Command: python main.py\n"
            "Skipped: total timeout budget exhausted before execution.\n"
            f"Per Try Timeout Sec: {execution_timeout_sec:.6f}\n"
            f"Total Timeout Sec: {max_total_execution_time_sec:.6f}\n"
            f"Total Used Sec: {previous_total_execution_duration_sec:.6f}\n"
            f"Applied Timeout Sec: {next_execution_timeout_sec:.6f}\n\n"
            f"{_build_feedback_log_section(feedback)}"
        )
        result = ExperimentResult(
            task_id=task.id,
            success=False,
            metrics={
                "execution_duration_sec": 0.0,
                "total_execution_duration_sec": previous_total_execution_duration_sec,
            },
            logs=execution_log,
            error_message=timeout_message,
            stop_reason="total_timeout",
        )
        write_execution_log(run_paths, execution_log, append=True)

        file_list = _collect_workspace_file_list(run_paths)
        if not file_list and (run_paths.workspace_dir / "main.py").exists():
            file_list = ["main.py"]
        write_meta(run_paths, task.id, file_list)

        return {
            "result": result,
            "execution_logs": execution_log,
            "execution_stdout": "",
            "execution_stderr": feedback["stderr"],
            "execution_return_code": None,
            "last_execution_duration_sec": 0.0,
            "total_execution_duration_sec": previous_total_execution_duration_sec,
            "status": "failed",
            "current_step": "done",
            "retry_count": retry_count,
            "evaluator_feedback": feedback,
            "error_signature": state.get("error_signature"),
            "same_error_count": state.get("same_error_count", 0),
            "stop_reason": "total_timeout",
            "error": timeout_message,
            "messages": [AIMessage(content=timeout_message)],
        }

    # 2) ローカル実行
    execution = run_workspace_python(
        run_paths=run_paths,
        entrypoint="main.py",
        timeout_sec=next_execution_timeout_sec,
    )
    is_success = execution.returncode == 0
    feedback = _build_evaluator_feedback(task=task, execution=execution)
    next_total_execution_duration_sec = (
        previous_total_execution_duration_sec + execution.duration_sec
    )

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
        f"Duration Sec: {execution.duration_sec:.6f}\n"
        f"Total Duration Sec: {next_total_execution_duration_sec:.6f}\n\n"
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
        metrics={
            "execution_duration_sec": execution.duration_sec,
            "total_execution_duration_sec": next_total_execution_duration_sec,
        },
        logs=execution_log,
        error_message=error_message,
        stop_reason=None,
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
    decision = _decide_next_action(
        feedback,
        retry_count,
        next_same_error_count,
        next_total_execution_duration_sec,
        max_total_execution_time_sec,
    )
    next_retry_count = retry_count if is_success else retry_count + 1
    result = result.model_copy(update={"stop_reason": decision.stop_reason})

    # 7) Stateの更新
    return {
        "result": result,
        "execution_logs": execution_log,
        "execution_stdout": execution.stdout,
        "execution_stderr": execution.stderr,
        "execution_return_code": execution.returncode,
        "last_execution_duration_sec": execution.duration_sec,
        "total_execution_duration_sec": next_total_execution_duration_sec,
        "status": decision.status,
        "current_step": decision.next_step,
        "retry_count": next_retry_count,
        "evaluator_feedback": feedback,
        "error_signature": current_error_signature,
        "same_error_count": next_same_error_count,
        "stop_reason": decision.stop_reason,
        "error": None if is_success else error_message,
        "messages": [AIMessage(content=decision.message)],
    }


# エイリアス設定
evaluator = evaluator_node
