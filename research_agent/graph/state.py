from typing import List, Literal, Optional, TypedDict

from schema.task import ExperimentResult, Task

AgentStep = Literal["planner", "researcher", "coder", "evaluator", "done"]
AgentStatus = Literal["pending", "running", "completed", "failed"]


class AgentState(TypedDict):
    """
    LangGraph 全ノードで共有する状態。
    ノード間のデータ受け渡しは必ずこの型を経由する。
    """

    task: Task
    messages: List[str]
    current_step: AgentStep
    code: Optional[str]
    logs: List[str]
    status: AgentStatus
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
