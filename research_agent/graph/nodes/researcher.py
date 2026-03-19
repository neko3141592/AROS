from __future__ import annotations

import os
from typing import Any, Dict

from langchain_core.messages import AIMessage
from tools.llm_client import LLMClientError, generate_text
from tools.prompt_manager import PromptManagerError, render_prompt

from graph.state import AgentState
from tools.paper_search import (
    ArxivSearchError,
    build_arxiv_query,
    fetch_arxiv_raw,
    format_papers_for_llm,
    parse_arxiv_response,
)

DEFAULT_MODEL_NAME = "gpt-4o-mini"


def _build_fallback_research_context(task_title: str, reason: str) -> str:
    """
    固定文章返却ロジック(v0.1用)。
    実際の論文検索結果を模したダミー情報を生成する。
    """

    return (
        f"# Research Context (Fallback) for: {task_title}\n\n"
        "arXiv search was unavailable, so this fallback summary is used.\n"
        f"- Reason: {reason}\n"
        "- Suggested direction: Start from a minimal baseline and validate with a small synthetic dataset.\n"
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

    # 2) 調査と要約 (v0.2: arXiv連携)
    # API障害時でもワークフロー全体が止まらないよう、フォールバックを返す。
    try:
        query = build_arxiv_query(
            keywords=[task.title, task.description],
            categories=["cs.AI", "cs.LG"],
        )
        raw_results = fetch_arxiv_raw(query=query, max_results=5, sort_by="relevance")
        papers = parse_arxiv_response(raw_results)
        formatted_context = format_papers_for_llm(papers)

        system_prompt = render_prompt(
            "system_researcher",
            {
                "task_title": task.title,
                "task_description": task.description,
                "search_results": formatted_context,
            }
        )

        user_prompt = (
            "上記の指示に従い、Markdown形式の実装コンテキストのみを返してください。"
            "前置きやコードフェンスは不要です。"
        )

        model_name = (
            os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
        )
        summary_context = generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model_name,
            temperature=0.2,
            timeout=30,
        )

        message_content = (
            "Researcher: arXiv検索を実行し、"
            f"{len(papers)}件の論文を実装向けコンテキストに整形しました。"
        )

    except (ArxivSearchError, ValueError, PromptManagerError, LLMClientError) as exc:
        summary_context = _build_fallback_research_context(task_title=task.title, reason=str(exc))
        message_content = (
            "Researcher: arXiv検索に失敗したため、"
            "フォールバックの調査コンテキストを返しました。"
        )

    # 3) Stateの更新を返却
    # 要約だけを渡すことで、後続の Coder がコンテキスト上限に触れるのを防ぐ。
    return {
        "research_context": summary_context,
        "status": "researching",
        "current_step": "coder",
        "messages": [
            AIMessage(
                content=message_content
            )
        ],
        "error": None,
    }


# エイリアス設定
researcher = researcher_node
