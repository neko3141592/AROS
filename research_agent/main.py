import sys
import os
import json
from datetime import datetime
from typing import Any, Dict
from tools.file_io import create_run_paths
import os
import sys

# パッケージのルート(research_agent)をパスに追加して、絶対インポートを可能にする
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage, AIMessage

# research_agent 直下で実行されることを前提にインポート
from graph import research_graph
from schema.task import Task

def json_serial(obj):
    """JSON serialization helper for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def save_state_to_json(state: Dict[str, Any], task_id: str):
    """
    AgentState をシリアライズして storage/ に保存する。
    """
    # 1. PydanticモデルやMessageオブジェクトをシリアライズ可能な形式に変換
    serializable_state = state.copy()
    serializable_state["task"] = state["task"].model_dump()
    if state["result"]:
        serializable_state["result"] = state["result"].model_dump()
    
    # メッセージの変換
    serializable_state["messages"] = [
        {"role": "AI" if isinstance(m, AIMessage) else "User", "content": m.content}
        for m in state["messages"]
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"storage/runs/state_{task_id}_{timestamp}.json"
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serializable_state, f, indent=4, ensure_ascii=False, default=json_serial)
    
    print(f"--- State saved to: {file_path} ---")

def run_aros(task_title: str, task_description: str, run_paths: RunPaths):
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

    # 2. 保存パスの取得
    run_paths = create_run_paths(task_id=initial_task.id)

    # 3. 初期状態 (Initial State) の設定
    initial_state: Dict[str, Any] = {
        "task": initial_task,
        "status": "starting",
        "messages": [HumanMessage(content=f"Request: {task_title}")],
        "current_step": "init",
        "research_context": "",
        "generated_code": None,
        "execution_logs": None,
        "retry_count": 0,
        "result": None,
        "run_paths": run_paths
    }

    # 3. グラフの実行 (LangGraph Invoke)
    # これにより Planner -> Researcher -> Coder -> Evaluator (-> Coder ...) の順で動く
    print(f"Executing task: {task_title}...")
    final_state = research_graph.invoke(initial_state)

    # 4. 結果の表示
    print("\n=== Execution Summary ===")
    print(f"Status: {final_state['status']}")
    
    # 状態を JSON に保存
    save_state_to_json(final_state, task_id=initial_task.id)

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
