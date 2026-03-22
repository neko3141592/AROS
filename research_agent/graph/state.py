from typing import Any, TypedDict, Annotated, Dict, List, Optional
import operator
from langchain_core.messages import BaseMessage
from schema.task import Task, ExperimentResult
from tools.file_io import RunPaths


class EvaluatorFeedback(TypedDict, total=False):
    """
    EvaluatorからCoderへ渡す自己修正用のフィードバック。
    """

    summary: str
    likely_cause: str
    suggested_fixes: List[str]
    can_self_fix: bool
    needs_research: bool
    return_code: int
    stdout: str
    stderr: str
    raw: Dict[str, Any]


class AgentState(TypedDict):
    """
    LangGraphの各ノード間で受け渡される状態（State）オブジェクト。
    グラフの実行中、常に最新の状態がここに保持される。
    シリアライズ可能なデータ（テキストやPydanticモデル）のみを保持すること。
    """
    
    # --- 1. 入出力・メタデータ ---
    task: Task                           # 現在実行中の親タスク
    status: str                          # 全体ステータス (planning, coding, executing, completed, failed)
    
    # --- 2. メッセージ履歴（会話・思考ログ） ---
    # operator.add を指定することで、既存のリストに新しいメッセージが追記される
    messages: Annotated[List[BaseMessage], operator.add]
    
    # --- 3. 実行コンテキスト ---
    current_step: str                    # 現在実行中のステップ名や説明
    research_context: str                # Researcherが収集した情報の要約など
    
    # --- 4. 生成物・実験データ ---
    # 正本は run_paths.workspace_dir 上の実ファイル。
    # generated_* は後方互換のための state mirror として保持する。
    generated_code: Optional[str]
    generated_files: Optional[Dict[str, str]]
    execution_logs: Optional[str]        # 実験の実行ログ
    execution_stdout: Optional[str]      # 実行時の標準出力
    execution_stderr: Optional[str]      # 実行時の標準エラー出力
    execution_return_code: Optional[int] # 実行時の終了コード
    last_execution_duration_sec: Optional[float]   # 直近試行の実行時間
    total_execution_duration_sec: Optional[float]  # 累積実行時間
    
    # --- 5. ファイル情報 ---
    run_paths: Optional[RunPaths]        # runごとの保存先情報（workspace_dir を含む）

    # --- 6. 結果と自己修復制御 ---
    retry_count: int                     # エラー発生時のリトライ回数
    evaluator_feedback: Optional[EvaluatorFeedback]  # Evaluatorの構造化フィードバック
    error_signature: Optional[str]       # 同一エラー判定用のフィンガープリント
    same_error_count: int                # 同一エラーの連続発生回数
    stop_reason: Optional[str]           # 停止理由（max_retry, repeated_errorなど）
    result: Optional[ExperimentResult]
    error: Optional[str]


def create_initial_state(task: Task) -> AgentState:
    """
    実行開始時に使う初期状態を生成する。
    
    Args:
        task: 対象タスク。
    """

    return {
        "task": task,
        "messages": [],
        "current_step": "planner",
        "research_context": "",
        "generated_code": None,
        "generated_files": None,
        "execution_logs": None,
        "execution_stdout": None,
        "execution_stderr": None,
        "execution_return_code": None,
        "last_execution_duration_sec": None,
        "total_execution_duration_sec": 0.0,
        "run_paths": None,
        "retry_count": 0,
        "evaluator_feedback": None,
        "error_signature": None,
        "same_error_count": 0,
        "stop_reason": None,
        "status": "pending",
        "result": None,
        "error": None,
    }
