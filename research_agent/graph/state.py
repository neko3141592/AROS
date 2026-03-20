from typing import TypedDict, Annotated, Dict, List, Optional
import operator
from langchain_core.messages import BaseMessage
from schema.task import Task, ExperimentResult
from tools.file_io import RunPaths

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
    # generated_* は後方互換のための mirror（参照元は workspace）として保持する。
    generated_code: Optional[str]
    generated_files: Optional[Dict[str, str]]
    execution_logs: Optional[str]        # 実験の実行ログ
    execution_stdout: Optional[str]      # 実行時の標準出力
    execution_stderr: Optional[str]      # 実行時の標準エラー出力
    execution_return_code: Optional[int] # 実行時の終了コード
    
    # --- 6. ファイル情報 ---
    run_paths: Optional[RunPaths]        # runごとの保存先情報（workspace_dir を含む）

    # --- 5. 結果と自己修復制御 ---
    retry_count: int                     # エラー発生時のリトライ回数
    result: Optional[ExperimentResult]
    error: Optional[str]


def create_initial_state(task: Task) -> AgentState:
    """
    実行開始時に使う初期状態を生成する。
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
        "run_paths": None,
        "retry_count": 0,
        "status": "pending",
        "result": None,
        "error": None,
    }
