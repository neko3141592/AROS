from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import uuid

class SubTask(BaseModel):
    """
    実行可能な最小単位のタスクモデル（再帰構造を持たない1階層のみ）
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="サブタスクの一意なID")
    title: str = Field(..., description="サブタスクのタイトル")
    description: str = Field(..., description="サブタスクの具体的な手順")
    assigned_agent: str = Field(..., description="担当エージェント (researcher, coder, etc.)")
    status: str = Field("pending", description="ステータス (pending, completed, failed)")

class Task(BaseModel):
    """
    ユーザーからの初期入力を管理する親タスクモデル。
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="タスクの一意なID")
    title: str = Field(..., description="タスクの簡潔なタイトル")
    description: str = Field(..., description="タスクの詳細な説明・目的")
    constraints: List[str] = Field(default_factory=list, description="制約事項（例: PyTorchを使用, 実行時間は10分以内など）")
    subtasks: List[SubTask] = Field(default_factory=list, description="分解された実行ステップ（サブタスク）のリスト")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ExperimentResult(BaseModel):
    """
    1回の実験（コード実行）の最終結果を格納するモデル
    """
    task_id: str = Field(..., description="紐づくタスクのID")
    success: bool = Field(..., description="実験がエラーなく正常に完了し、目的を達成したか")
    metrics: Dict[str, float] = Field(default_factory=dict, description="評価指標の辞書 (例: {'accuracy': 0.95, 'loss': 0.1})")
    logs: str = Field(default_factory=str, description="標準出力・標準エラー出力の結果")
    error_message: Optional[str] = Field(None, description="失敗時のエラーメッセージ")
