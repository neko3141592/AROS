from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage

from graph.state import AgentState
from tools.cli_logging import log_kv, log_node_end, log_node_start, preview_list, preview_text
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
from tools.task_context import format_subtasks_for_prompt, summarize_failure_for_prompt

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
    task = state.get("task")
    if not task:
        return {
            "status": "failed",
            "error": "Researcher: task が存在しません。",
            "messages": [AIMessage(content="Researcher failed: task is missing.")],
        }

    try:
        failure = summarize_failure_for_prompt(
            state.get("evaluator_feedback"),
            state.get("execution_stderr"),
        )
        log_node_start(
            "Researcher",
            {
                "task_title": task.title,
                "subtasks": len(task.subtasks),
                "failure_summary": failure["summary"] or "(none)",
                "failure_stderr": preview_text(failure["stderr"]),
            },
        )
        model_name = get_model_name("RESEARCHER_MODEL_NAME", DEFAULT_MODEL_NAME)

        def _log_search_attempt(
            keywords: list[str],
            categories: list[str] | None,
            paper_count: int,
        ) -> None:
            log_kv("Researcher Query Keywords", preview_list(keywords))
            log_kv("Researcher Query Categories", preview_list(categories or []))
            log_kv("Researcher Query Hits", paper_count)

        summary_context, paper_count = _generate_research_summary(
            task_title=task.title,
            task_description=task.description,
            task_constraints=task.constraints,
            task_subtasks=format_subtasks_for_prompt(task.subtasks),
            previous_failure_summary=failure["summary"],
            previous_failure_likely_cause=failure["likely_cause"],
            previous_failure_stderr=failure["stderr"],
            model_name=model_name,
            build_arxiv_query_fn=build_arxiv_query,
            fetch_arxiv_raw_fn=fetch_arxiv_raw,
            parse_arxiv_response_fn=parse_arxiv_response,
            format_papers_for_llm_fn=format_papers_for_llm,
            render_prompt_fn=render_prompt,
            generate_text_fn=generate_text,
            log_search_attempt_fn=_log_search_attempt,
        )

        message_content = (
            "Researcher: arXiv検索を実行し、"
            f"{paper_count}件の論文を実装向けコンテキストに整形しました。"
        )
        log_node_end(
            "Researcher",
            {
                "paper_count": paper_count,
                "context_preview": preview_text(summary_context),
                "next_step": "coder",
            },
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
        log_node_end(
            "Researcher",
            {
                "fallback_reason": preview_text(str(exc)),
                "context_preview": preview_text(summary_context),
                "next_step": "coder",
            },
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
