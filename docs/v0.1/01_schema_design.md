# AROS v0.1: スキーマと状態管理の詳細構造設計

LangGraphで構築するAutonomous Research Loop (ARL) において、「絶対に崩壊しない」ための最も重要な部分がここ（Schema & State）です。LangGraphの各ノードは、必ずこの `AgentState` のみをインターフェースとしてデータの受け渡しを行います。

## 1. `schema/task.py`: データモデルの定義

Pydanticを用いて、タスク入力と実験結果の厳密な型定義を行います。これにより、LLMの生成結果がスキーマに違反した場合に即座にエラーとして検知（バリデーション）できます。

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import uuid

class Task(BaseModel):
    """
    ユーザーからの初期入力を管理する親タスクモデル
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="タスクの一意なID")
    title: str = Field(..., description="タスクの簡潔なタイトル")
    description: str = Field(..., description="タスクの詳細な説明・目的")
    constraints: List[str] = Field(default_factory=list, description="制約事項（例: PyTorchを使用, 実行時間は10分以内など）")
    subtasks: List["SubTask"] = Field(default_factory=list, description="分解された実行ステップ（サブタスク）のリスト")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SubTask(BaseModel):
    """
    実行可能な最小単位のタスクモデル（再帰構造を持たない1階層のみ）
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="サブタスクの一意なID")
    title: str = Field(..., description="サブタスクのタイトル")
    description: str = Field(..., description="サブタスクの具体的な手順")
    assigned_agent: str = Field(..., description="担当エージェント (researcher, coder, etc.)")
    status: str = Field("pending", description="ステータス (pending, completed, failed)")

class ExperimentResult(BaseModel):
    """
    1回の実験（コード実行）の最終結果を格納するモデル
    """
    task_id: str = Field(..., description="紐づくタスクのID")
    success: bool = Field(..., description="実験がエラーなく正常に完了し、目的を達成したか")
    metrics: Dict[str, float] = Field(default_factory=dict, description="評価指標の辞書 (例: {'accuracy': 0.95, 'loss': 0.1})")
    logs: str = Field(default_factory=str, description="標準出力・標準エラー出力の結果")
    error_message: Optional[str] = Field(None, description="失敗時のエラーメッセージ")
```

## 2. `graph/state.py`: AgentStateの定義

LangGraphのグラフ全体で引き回される「状態」の定義です。型ヒントを厳密に定義することで、各ノード（Planner, Coder等）が「どのデータを読み取れ、どのデータを更新すべきか」が明確になり、スパゲッティ化（状態の崩壊）を防ぎます。

LangGraphの標準的なアプローチとして `TypedDict` と `Annotated`（reducer機能）を使用します。

```python
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
import operator
from schema.task import Task, ExperimentResult

class AgentState(TypedDict):
    """
    LangGraphの各ノード間で受け渡される状態（State）オブジェクト。
    グラフの実行中、常に最新の状態がここに保持される。
    """
    
    # --- 1. 入出力・メタデータ ---
    task: Task                           # 現在実行中のタスク
    status: str                          # 現在の全体ステータス (例: "planning", "coding", "executing", "completed", "failed")
    
    # --- 2. メッセージ履歴（会話・思考ログ） ---
    # Annotatedとoperator.addを使うことで、リストが上書きされず追加(append)されるようになる
    messages: Annotated[List[BaseMessage], operator.add]
    
    # --- 3. 実行コンテキスト ---
    current_step: str                    # 現在の処理フェーズの人間向け説明
    research_context: str                # Researcherが集めた論文情報などのテキスト
    
    # --- 4. 生成物・実験データ ---
    generated_code: Optional[str]        # Coderが生成した実行対象のPythonコード
    execution_logs: Optional[str]        # Evaluator/実行エンジンが返したコードの実行ログ
    
    # --- 5. 結果と自己修復制御 ---
    retry_count: int                     # エラー発生時の再試行回数（無限ループ防止用）
    result: Optional[ExperimentResult]   # 実験の最終結果（正常完了時のみセット）
```

### 【設計のポイント・絶対に守るべきこと】
1. **`messages` の Reducer (`operator.add`)**: AIの推論プロセス（思考）はすべて `messages` にどんどん追記（Append）していく設計にします。上書き（Overwrite）すると過去の文脈がロストし、AIがパニックになります。
2. **`retry_count` の導入**: エラーが出た際に Coder に戻すエッジ（Self-Correction）を作りますが、`retry_count` が一定回数（例: 3回）を超えたら強制終了（Failed）させる条件分岐を `edges.py` で必ず実装します（クラウド破産・無限ループ防止）。
3. **副作用（I/O）の隔離**: この `AgentState` に格納される情報はすべて純粋なテキストやデータモデルのみとします。RunPodの接続セッションオブジェクトやファイルハンドラなどの「シリアライズできないインスタンス」は決して State に入れてはいけません（DBへの状態の永続化ができなくなるため）。
