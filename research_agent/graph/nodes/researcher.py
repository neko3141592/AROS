from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage

from graph.state import AgentState
from tools.llm_client import LLMClientError, generate_text
from tools.model_config import get_model_name
from tools.paper_search import (
    ArxivSearchError,
    build_arxiv_query,
    fetch_arxiv_raw,
    format_papers_for_llm,
    parse_arxiv_response,
)
from tools.prompt_manager import PromptManagerError, render_prompt
from tools.researcher_helpers import (
    _build_fallback_research_context,
    _generate_research_summary,
)

DEFAULT_MODEL_NAME = "gpt-4o-mini"


def researcher_node(state: AgentState) -> Dict[str, Any]:
    """
    Researcherノード（v0.1モック版）。

    役割:
    - 指定されたタスクに関連する論文情報を「調査・要約」する。
    - v0.1の方針に基づき、全文ではなく「実装に必要なエッセンス」のみを抽出してStateに格納する。

    Args:
        state: ノード間で受け渡す現在の状態。
    """
    print("--- [Node: Researcher] 論文を調査・要約しています... ---")

    task = state.get("task")
    if not task:
        return {
            "status": "failed",
            "error": "Researcher: task が存在しません。",
            "messages": [AIMessage(content="Researcher failed: task is missing.")],
        }

    try:
        model_name = get_model_name("RESEARCHER_MODEL_NAME", DEFAULT_MODEL_NAME)
        summary_context, paper_count = _generate_research_summary(
            task_title=task.title,
            task_description=task.description,
            model_name=model_name,
            build_arxiv_query_fn=build_arxiv_query,
            fetch_arxiv_raw_fn=fetch_arxiv_raw,
            parse_arxiv_response_fn=parse_arxiv_response,
            format_papers_for_llm_fn=format_papers_for_llm,
            render_prompt_fn=render_prompt,
            generate_text_fn=generate_text,
        )

        message_content = (
            "Researcher: arXiv検索を実行し、"
            f"{paper_count}件の論文を実装向けコンテキストに整形しました。"
        )

    except (ArxivSearchError, ValueError, PromptManagerError, LLMClientError) as exc:
        summary_context = _build_fallback_research_context(
            task_title=task.title,
            reason=str(exc),
        )
        message_content = (
            "Researcher: arXiv検索に失敗したため、"
            "フォールバックの調査コンテキストを返しました。"
        )

    return {
        "research_context": summary_context,
        "status": "researching",
        "current_step": "coder",
        "messages": [AIMessage(content=message_content)],
        "error": None,
    }


# エイリアス設定
researcher = researcher_node
