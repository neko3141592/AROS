from __future__ import annotations

from typing import Any, Callable


def _build_fallback_research_context(task_title: str, reason: str) -> str:
    """
    固定文章返却ロジック(v0.1用)。
    実際の論文検索結果を模したダミー情報を生成する。

    Args:
        task_title: タスクのタイトル。
        reason: フォールバック理由や補足情報。
    """

    return (
        f"# Research Context (Fallback) for: {task_title}\n\n"
        "arXiv search was unavailable, so this fallback summary is used.\n"
        f"- Reason: {reason}\n"
        "- Suggested direction: Start from a minimal baseline and validate with a small synthetic dataset.\n"
    )


def _generate_research_summary(
    task_title: str,
    task_description: str,
    model_name: str,
    build_arxiv_query_fn: Callable[..., str],
    fetch_arxiv_raw_fn: Callable[..., list[Any]],
    parse_arxiv_response_fn: Callable[..., list[Any]],
    format_papers_for_llm_fn: Callable[..., str],
    render_prompt_fn: Callable[..., str],
    generate_text_fn: Callable[..., str],
) -> tuple[str, int]:
    """
    arXiv検索とLLM要約の本処理を実行し、要約テキストと論文件数を返す。

    Args:
        task_title: タスクタイトル。
        task_description: タスク説明。
        model_name: 要約に使うモデル名。
        build_arxiv_query_fn: クエリ生成関数。
        fetch_arxiv_raw_fn: arXiv生データ取得関数。
        parse_arxiv_response_fn: arXivレスポンス整形関数。
        format_papers_for_llm_fn: LLM向け文脈整形関数。
        render_prompt_fn: プロンプト描画関数。
        generate_text_fn: LLMテキスト生成関数。
    """
    query = build_arxiv_query_fn(
        keywords=[task_title, task_description],
        categories=["cs.AI", "cs.LG"],
    )
    raw_results = fetch_arxiv_raw_fn(query=query, max_results=5, sort_by="relevance")
    papers = parse_arxiv_response_fn(raw_results)
    formatted_context = format_papers_for_llm_fn(papers)

    system_prompt = render_prompt_fn(
        "system_researcher",
        {
            "task_title": task_title,
            "task_description": task_description,
            "search_results": formatted_context,
        },
    )

    user_prompt = (
        "Follow the instructions above and return only the implementation context in Markdown."
        "Do not add preamble text or code fences."
    )

    summary_context = generate_text_fn(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model_name,
        temperature=0.2,
        timeout=30,
    )

    return summary_context, len(papers)
