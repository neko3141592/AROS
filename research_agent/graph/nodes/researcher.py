from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage

try:
    from graph.state import AgentState

except ModuleNotFoundError:
    from research_agent.graph.state import AgentState


def _build_mock_research_context(task_title: str) -> str:
    """
    固定文章返却ロジック(v0.1用)。
    実際の論文検索結果を模したダミー情報を生成する。
    """

    return (
        f"# Mock Research Context for: {task_title}\n\n"
        "## Selected Paper: 'Attention Is All You Need' (v0.1 Mock)\n"
        "- **Authors**: Ashish Vaswani, et al.\n"
        "- **Summary**: This paper proposes the Transformer architecture, "
        "relying entirely on attention mechanisms to draw global dependencies "
        "between input and output.\n"
        "- **Key Insight**: Multi-head self-attention allows for significantly "
        "more parallelization than convolutional or recurrent layers.\n\n"
        "## Implementation Notes\n"
        "- Scaling factor for dot-product attention is 1/sqrt(d_k).\n"
        "- Positional encoding is required to maintain sequence order information."
    )


def researcher_node(state: AgentState) -> Dict[str, Any]:
    """
    Researcherノード（v0.1モック版）。

    役割:
    - 指定されたタスクに関連する論文情報を「調査・要約」する。
    - v0.1の方針に基づき、全文ではなく「実装に必要なエッセンス」のみを抽出してStateに格納する。
    """
    print("--- [Node: Researcher] 論文を調査・要約しています... ---")

    # 1) 入力確認
    task = state.get("task")
    if not task:
        return {
            "status": "failed",
            "error": "Researcher: task が存在しません。",
            "messages": [AIMessage(content="Researcher failed: task is missing.")],
        }

    # 2) 調査と要約 (v0.1 モック)
    # 本来はここで API実行 -> LLM要約 が入るが、
    # 設計指針に従い「実装に特化した要約済みテキスト」を取得する。
    summary_context = _build_mock_research_context(task_title=task.title)

    # 3) Stateの更新を返却
    # 要約だけを渡すことで、後続の Coder がコンテキスト上限に触れるのを防ぐ。
    return {
        "research_context": summary_context,
        "status": "researching",
        "current_step": "coder",
        "messages": [
            AIMessage(
                content=(
                    "Researcher: 関連論文の調査を完了し、"
                    "実装に必要な要約（Context）を抽出しました。"
                )
            )
        ],
        "error": None,
    }


# エイリアス設定
researcher = researcher_node
