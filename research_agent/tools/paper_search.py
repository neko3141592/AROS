from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

try:
    import arxiv
except ImportError:  # pragma: no cover - tested via dependency error path
    arxiv = None  # type: ignore[assignment]


class ArxivSearchError(Exception):
    """arXiv 検索処理で発生するエラー。"""


@dataclass(frozen=True)
class Paper:
    """
    LLM に渡しやすい論文情報の正規化モデル。
    """

    paper_id: str
    title: str
    summary: str
    authors: list[str]
    published: str
    updated: str
    primary_category: str
    categories: list[str]
    pdf_url: str
    entry_url: str


def _escape_term(term: str) -> str:
    return term.replace('"', '\\"').strip()


def _iso_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value or "")


def build_arxiv_query(
    keywords: Sequence[str], categories: Sequence[str] | None = None
) -> str:
    """
    arXiv 用の search_query 文字列を構築する。
    """
    normalized_keywords = [_escape_term(k) for k in keywords if k and k.strip()]
    if not normalized_keywords:
        raise ValueError("keywords must contain at least one non-empty term.")

    keyword_query = " AND ".join([f'all:"{k}"' for k in normalized_keywords])

    normalized_categories = [c.strip() for c in (categories or []) if c and c.strip()]
    if not normalized_categories:
        return keyword_query

    category_query = " OR ".join([f"cat:{c}" for c in normalized_categories])
    return f"({keyword_query}) AND ({category_query})"


def fetch_arxiv_raw(
    query: str,
    max_results: int = 5,
    sort_by: str = "relevance",
) -> list[Any]:
    """
    arXiv API を呼び出して生の検索結果を返す。
    """
    if arxiv is None:
        raise ArxivSearchError(
            "arxiv package is not installed. Add 'arxiv' to requirements and install dependencies."
        )

    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
    }
    if sort_by not in sort_map:
        valid = ", ".join(sort_map.keys())
        raise ValueError(f"Invalid sort_by: {sort_by}. Choose from {valid}.")

    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_map[sort_by],
            sort_order=arxiv.SortOrder.Descending,
        )
        client = arxiv.Client(
            page_size=min(max_results, 100),
            delay_seconds=3.0,
            num_retries=2,
        )
        return list(client.results(search))
    except Exception as exc:
        raise ArxivSearchError(f"Failed to fetch papers from arXiv: {exc}") from exc


def parse_arxiv_response(raw_results: Sequence[Any]) -> list[Paper]:
    """
    arxiv.Result の配列を Paper モデルへ変換する。
    """
    papers: list[Paper] = []

    for item in raw_results:
        get_short_id = getattr(item, "get_short_id", None)
        paper_id = (
            str(get_short_id()) if callable(get_short_id) else str(getattr(item, "entry_id", ""))
        )
        summary = " ".join(str(getattr(item, "summary", "")).split())
        title = " ".join(str(getattr(item, "title", "")).split())
        authors = [str(getattr(author, "name", "")).strip() for author in getattr(item, "authors", [])]
        authors = [name for name in authors if name]

        papers.append(
            Paper(
                paper_id=paper_id,
                title=title,
                summary=summary,
                authors=authors,
                published=_iso_date(getattr(item, "published", "")),
                updated=_iso_date(getattr(item, "updated", "")),
                primary_category=str(getattr(item, "primary_category", "") or ""),
                categories=list(getattr(item, "categories", []) or []),
                pdf_url=str(getattr(item, "pdf_url", "") or ""),
                entry_url=str(getattr(item, "entry_id", "") or ""),
            )
        )

    return papers


def format_papers_for_llm(papers: Sequence[Paper], max_summary_chars: int = 500) -> str:
    """
    検索結果を LLM プロンプトに貼り付けやすいテキストへ整形する。
    """
    if not papers:
        return "No arXiv papers were found for the given query."

    lines = ["# ArXiv Search Results", ""]

    for idx, paper in enumerate(papers, start=1):
        summary = paper.summary[:max_summary_chars].strip()
        if len(paper.summary) > max_summary_chars:
            summary += " ..."

        lines.extend(
            [
                f"## {idx}. {paper.title}",
                f"- Paper ID: {paper.paper_id}",
                f"- Authors: {', '.join(paper.authors) if paper.authors else 'N/A'}",
                f"- Published: {paper.published}",
                f"- Updated: {paper.updated}",
                f"- Primary Category: {paper.primary_category or 'N/A'}",
                f"- Categories: {', '.join(paper.categories) if paper.categories else 'N/A'}",
                f"- PDF: {paper.pdf_url or 'N/A'}",
                f"- Entry: {paper.entry_url or 'N/A'}",
                f"- Summary: {summary or 'N/A'}",
                "",
            ]
        )

    return "\n".join(lines).strip()

