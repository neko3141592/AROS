import sys
from typing import Any, Dict
from langchain_core.messages import HumanMessage
from research_agent.graph import research_graph
from research_agent.schema.task import Task

def run_aros(task_title: str, task_description: str):
    """
    AROS v0.1 実行エントリーポイント。
    初期状態を構築し、LangGraphを起動する。
    """
    print("=== AROS (Autonomous Research Loop) v0.1 Starting ===")
    
    # 1. 初期タスクの構築
    # 本来はユーザー入力から自動生成されるが、v0.1では手動で定義
    initial_task = Task(
        title=task_title,
        description=task_description,
        constraints=["v0.1 Mock Execution", "PyTorch (Dummy)"],
        subtasks=[]  # Plannerが埋める
    )

    # 2. 初期状態 (Initial State) の設定
    initial_state: Dict[str, Any] = {
        "task": initial_task,
        "status": "starting",
        "messages": [HumanMessage(content=f"Request: {task_title}")],
        "current_step": "init",
        "research_context": "",
        "generated_code": None,
        "execution_logs": None,
        "retry_count": 0,
        "result": None
    }

    # 3. グラフの実行 (LangGraph Invoke)
    # これにより Planner -> Researcher -> Coder -> Evaluator (-> Coder ...) の順で動く
    print(f"Executing task: {task_title}...")
    final_state = research_graph.invoke(initial_state)

    # 4. 結果の表示
    print("\n=== Execution Summary ===")
    print(f"Status: {final_state['status']}")
    print(f"Final Step: {final_state['current_step']}")
    print(f"Retry Count: {final_state['retry_count']}")
    
    if final_state["result"]:
        result = final_state["result"]
        print(f"Success: {result.success}")
        if result.success:
            print("--- Generated Code Snippet ---")
            print(final_state["generated_code"][:200] + "...")
        else:
            print(f"Error: {result.error_message}")

    print("\n=== Message History ===")
    for msg in final_state["messages"]:
        role = "AI" if not isinstance(msg, HumanMessage) else "User"
        print(f"[{role}]: {msg.content}")

if __name__ == "__main__":
    # デフォルトのタスク内容。引数から取ることも可能。
    title = "Transformer Model Implementation"
    desc = "Implement a basic Transformer model based on 'Attention Is All You Need'."
    
    if len(sys.argv) > 2:
        title = sys.argv[1]
        desc = sys.argv[2]
        
    run_aros(title, desc)
