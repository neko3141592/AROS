from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Dict

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from graph.state import AgentState, EvaluatorFeedback
from schema.llm_outputs import EvaluatorAnalysisOutput
from schema.task import ExperimentResult, Task
from tools.file_io import write_execution_log, write_meta
from tools.llm_client import LLMClientError, generate_text
from tools.local_executor import LocalExecutionResult, run_workspace_python
from tools.prompt_manager import PromptManagerError, render_prompt

DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_EXECUTION_TIMEOUT_SEC = 60.0
DEFAULT_MAX_TOTAL_EXECUTION_TIME_SEC = 180.0
DEFAULT_LLM_ANALYSIS_TIMEOUT_SEC = 20.0
# retry_count は 0 始まりで、3回目の失敗で停止する。
MAX_SAME_ERROR_COUNT = 3
MAX_RETRY_COUNT = 2

@dataclass
class FailureClassification:
    summary: str
    likely_cause: str
    suggested_fixes: list[str]
    can_self_fix: bool
    needs_research: bool

@dataclass
class EvaluatorDecision:
    status: str
    next_step: str
    stop_reason: str | None
    message: str


def _read_positive_float_env(name: str, default: float) -> float:
    """
    正の浮動小数点設定値を環境変数から読み取る。

    Args:
        name: 環境変数名。
        default: 未設定時の既定値。
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default

    value = float(raw)
    if value <= 0:
        raise ValueError(f"{name} must be > 0.")
    return value


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


def _strip_code_fence(text: str) -> str:
    """
    LLM が返すコードフェンスを除去する。

    Args:
        text: 処理対象テキスト。
    """
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    lines = cleaned.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    return cleaned


def _extract_first_json_payload(text: str) -> Any:
    """
    文字列中の最初の JSON payload を取り出す。

    Args:
        text: 処理対象テキスト。
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


def _parse_evaluator_analysis_output(raw_text: str) -> EvaluatorAnalysisOutput:
    """
    Evaluator 用 LLM 解析出力をパースする。

    Args:
        raw_text: LLM の生テキスト。
    """
    cleaned = _strip_code_fence(raw_text)
    payload = _extract_first_json_payload(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("Evaluator analysis output must be a JSON object.")
    return EvaluatorAnalysisOutput.model_validate(payload)


def _should_use_evaluator_llm_analysis() -> bool:
    """
    Evaluator の LLM 解析を有効化すべきか判定する。
    """
    flag = (os.getenv("ENABLE_EVALUATOR_LLM_ANALYSIS") or "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    return bool(os.getenv("OPENAI_API_KEY"))


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
    return {
        **base_feedback,
        "likely_cause": analysis.likely_cause,
        "suggested_fixes": analysis.suggested_fixes,
        "can_self_fix": analysis.can_self_fix,
        "needs_research": analysis.needs_research,
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

    try:
        system_prompt = render_prompt(
            "system_evaluator",
            {
                "task_title": task.title,
                "task_description": task.description,
                "return_code": execution.returncode,
                "stdout": execution.stdout or "(empty)",
                "stderr": execution.stderr or "(empty)",
                "base_summary": base_feedback.get("summary", ""),
                "base_likely_cause": base_feedback.get("likely_cause", ""),
                "base_suggested_fixes": base_feedback.get("suggested_fixes", []),
            },
        )
        model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
        raw_text = generate_text(
            system_prompt=system_prompt,
            user_prompt="Return only the JSON object.",
            model=model_name,
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

    # 2) ローカル実行
    execution = run_workspace_python(
        run_paths=run_paths,
        entrypoint="main.py",
        timeout_sec=execution_timeout_sec,
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
