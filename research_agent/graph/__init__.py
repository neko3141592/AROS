from __future__ import annotations

from langgraph.graph import StateGraph, END

# 各コンポーネントのインポート
try:
    from graph.state import AgentState
    from graph.nodes.planner import planner_node
    from graph.nodes.researcher import researcher_node
    from graph.nodes.coder import coder_node
    from graph.nodes.evaluator import evaluator_node
    from graph.edges import should_continue
except ModuleNotFoundError:
    from research_agent.graph.state import AgentState
    from research_agent.graph.nodes.planner import planner_node
    from research_agent.graph.nodes.researcher import researcher_node
    from research_agent.graph.nodes.coder import coder_node
    from research_agent.graph.nodes.evaluator import evaluator_node
    from research_agent.graph.edges import should_continue

def create_research_graph():
    """
    AROS v0.1: 自律型研究エージェントのグラフ構造を構築・コンパイルする。
    
    フロー:
    Planner -> Researcher -> Coder -> Evaluator
                                      | (リトライ判定)
                                      v
                                    Coder / END
    """
    
    # 1) StateGraph の初期化
    workflow = StateGraph(AgentState)
    
    # 2) ノードの登録
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("evaluator", evaluator_node)
    
    # 3) 基本エッジ（固定ルート）の定義
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "coder")
    workflow.add_edge("coder", "evaluator")
    
    # 4) 条件付きエッジ（リサイクル・終了判定）の定義
    # Evaluator ノードが終わった後、should_continue 関数の戻り値によって分岐する
    workflow.add_conditional_edges(
        "evaluator",
        should_continue,
        {
            "coder": "coder",   # 失敗時は Coder に戻る
            "done": END         # 完了時はグラフを終了する
        }
    )
    
    # 5) グラフのコンパイル
    app = workflow.compile()
    
    return app

# 利用しやすいようにコンパイル済みインスタンスを公開しておく
# v0.4 で外部DB（Checkpoint）を導入する際は、ここで設定を行う
research_graph = create_research_graph()
