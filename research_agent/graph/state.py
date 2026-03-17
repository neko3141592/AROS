from typing import TypedDict, Annotated, List, Optional
import operator
from langchain_core.messages import BaseMessage
from research_agent.schema.task import Task, ExperimentResult

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
    generated_code: Optional[str]        # Coderが生成したPythonコード
    execution_logs: Optional[str]        # 実験の実行ログ
    
    # --- 5. 結果と自己修復制御 ---
    retry_count: int                     # エラー発生時のリトライ回数
    result: Optional[ExperimentResult]   # 最終的な実験結果

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
        "code": None,
        "logs": [],
        "status": "pending",
        "result": None,
        "error": None,
    }
