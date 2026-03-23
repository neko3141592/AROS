from __future__ import annotations

import re
from typing import Any, Callable


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "basic",
    "by",
    "for",
    "from",
    "how",
    "if",
    "implement",
    "implementation",
    "in",
    "into",
    "is",
    "it",
    "model",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "use",
    "using",
    "with",
}


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


def _unique_terms(terms: list[str]) -> list[str]:
    """
    順序を保ったまま重複語を除く。
    """
    seen: set[str] = set()
    result: list[str] = []
    for term in terms:
        normalized = term.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(term.strip())
    return result


def _extract_search_terms(task_title: str, task_description: str) -> list[str]:
    """
    task title / description から arXiv 検索向けの短い技術語を抽出する。
    """
    raw_terms: list[str] = []

    quoted_phrases = re.findall(r"'([^']+)'|\"([^\"]+)\"", task_description)
    for left, right in quoted_phrases:
        phrase = (left or right).strip()
        if phrase:
            raw_terms.append(phrase)

    token_source = f"{task_title} {task_description}".lower()
    token_candidates = re.findall(r"[a-zA-Z][a-zA-Z0-9.+_-]{2,}", token_source)
    for token in token_candidates:
        if token in _STOPWORDS:
            continue
        raw_terms.append(token)

    preferred_terms: list[str] = []
    for term in raw_terms:
        lowered = term.lower()
        if "attention is all you need" in lowered:
            preferred_terms.append("attention is all you need")
        elif "transformer" in lowered:
            preferred_terms.append("transformer")
        elif "bert" in lowered:
            preferred_terms.append("bert")
        elif "diffusion" in lowered:
            preferred_terms.append("diffusion")
        elif "llama" in lowered:
            preferred_terms.append("llama")
        elif "gpt" in lowered:
            preferred_terms.append("gpt")
        elif lowered not in _STOPWORDS:
            preferred_terms.append(term)

    return _unique_terms(preferred_terms)


def _build_search_plan(
    task_title: str,
    task_description: str,
) -> list[tuple[list[str], list[str] | None]]:
    """
    0件時に段階的に広げる検索プランを返す。
    """
    search_terms = _extract_search_terms(task_title, task_description)
    if not search_terms:
        search_terms = [task_title.strip()]

    narrow_terms = search_terms[:3]
    broad_terms = search_terms[:1]

    plans = [
        (narrow_terms, ["cs.AI", "cs.LG"]),
        (broad_terms, ["cs.AI", "cs.LG"]),
        (broad_terms, None),
    ]

    unique_plans: list[tuple[list[str], list[str] | None]] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...] | None]] = set()
    for keywords, categories in plans:
        normalized = (
            tuple(keyword.lower() for keyword in keywords),
            tuple(categories) if categories is not None else None,
        )
        if not keywords or normalized in seen:
            continue
        seen.add(normalized)
        unique_plans.append((keywords, categories))

    return unique_plans


def _generate_research_summary(
    task_title: str,
    task_description: str,
    task_constraints: list[str],
    task_subtasks: list[str],
    previous_failure_summary: str,
    previous_failure_likely_cause: str,
    previous_failure_stderr: str,
    model_name: str,
    build_arxiv_query_fn: Callable[..., str],
    fetch_arxiv_raw_fn: Callable[..., list[Any]],
    parse_arxiv_response_fn: Callable[..., list[Any]],
    format_papers_for_llm_fn: Callable[..., str],
    render_prompt_fn: Callable[..., str],
    generate_text_fn: Callable[..., str],
    log_search_attempt_fn: Callable[[list[str], list[str] | None, int], None] | None = None,
) -> tuple[str, int]:
    """
    arXiv検索とLLM要約の本処理を実行し、要約テキストと論文件数を返す。

    Args:
        task_title: タスクタイトル。
        task_description: タスク説明。
        task_constraints: タスク制約。
        task_subtasks: タスク分解結果。
        previous_failure_summary: 直前失敗の要約。
        previous_failure_likely_cause: 直前失敗の原因推定。
        previous_failure_stderr: 直前失敗の stderr。
        model_name: 要約に使うモデル名。
        build_arxiv_query_fn: クエリ生成関数。
        fetch_arxiv_raw_fn: arXiv生データ取得関数。
        parse_arxiv_response_fn: arXivレスポンス整形関数。
        format_papers_for_llm_fn: LLM向け文脈整形関数。
        render_prompt_fn: プロンプト描画関数。
        generate_text_fn: LLMテキスト生成関数。
    """
    papers: list[Any] = []
    for keywords, categories in _build_search_plan(task_title, task_description):
        query = build_arxiv_query_fn(
            keywords=keywords,
            categories=categories,
        )
        raw_results = fetch_arxiv_raw_fn(query=query, max_results=5, sort_by="relevance")
        papers = parse_arxiv_response_fn(raw_results)
        if log_search_attempt_fn is not None:
            log_search_attempt_fn(keywords, categories, len(papers))
        if papers:
            break

    formatted_context = format_papers_for_llm_fn(papers)

    system_prompt = render_prompt_fn(
        "system_researcher",
        {
            "task_title": task_title,
            "task_description": task_description,
            "task_constraints": task_constraints,
            "task_subtasks": task_subtasks,
            "previous_failure_summary": previous_failure_summary,
            "previous_failure_likely_cause": previous_failure_likely_cause,
            "previous_failure_stderr": previous_failure_stderr,
            "search_results": formatted_context,
        },
    )

    user_prompt = (
        "Follow the instructions above and return only the implementation context in Markdown."
        " If a previous failure is provided, focus the research on unblocking that failure instead of restating the original task."
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
