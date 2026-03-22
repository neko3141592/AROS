from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


class PlannerSubTaskOutput(BaseModel):
    """
    Planner が LLM から受け取る1件分のサブタスク出力モデル。
    """

    title: str = Field(..., min_length=1, description="サブタスクのタイトル")
    description: str = Field(..., min_length=1, description="サブタスクの実行内容")
    assigned_agent: Literal["researcher", "coder", "evaluator"] = Field(
        ..., description="担当エージェント"
    )
    status: Literal["pending", "completed", "failed"] = Field(
        default="pending", description="サブタスクの状態"
    )


class PlannerOutput(BaseModel):
    """
    Planner ノードの LLM 出力全体モデル。
    """
    
    subtasks: List[PlannerSubTaskOutput] = Field(
        ..., min_length=1, description="分解されたサブタスク配列"
    )


class CoderOutput(BaseModel):
    """
    Coder ノードの LLM 出力モデル。
    files は {ファイルパス: ファイル本文} を表す。
    """

    files: Dict[str, str] = Field(
        default_factory=dict, description="生成ファイル群（path -> content）"
    )


class EvaluatorAnalysisOutput(BaseModel):
    """
    Evaluator が失敗解析用 LLM から受け取る補助出力モデル。
    """

    likely_cause: str = Field(..., min_length=1, description="推定される主因")
    suggested_fixes: List[str] = Field(
        default_factory=list, description="具体的な修正候補"
    )
    can_self_fix: bool = Field(..., description="Coder が自己修正可能か")
    needs_research: bool = Field(..., description="追加調査が必要か")
