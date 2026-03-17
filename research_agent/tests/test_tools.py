from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.file_io import (  # noqa: E402
    create_run_paths,
    read_execution_log,
    read_generated_code,
    save_generated_code,
    save_workspace_files,
    write_execution_log,
)
import tools.paper_search as paper_search  # noqa: E402


def test_create_run_paths_creates_directories(tmp_path: Path) -> None:
    paths = create_run_paths(task_id="task-001", base_dir=tmp_path)

    assert paths.run_dir.exists()
    assert paths.run_dir.is_dir()
    assert paths.workspace_dir.exists()
    assert paths.workspace_dir.is_dir()
    assert paths.run_id.startswith("task-001_")


def test_save_and_read_generated_code_roundtrip(tmp_path: Path) -> None:
    paths = create_run_paths(task_id="task-002", base_dir=tmp_path)
    code = "print('hello')\n"

    save_generated_code(paths, code)
    loaded = read_generated_code(paths)

    assert loaded == code
    assert paths.code_path.exists()


def test_write_execution_log_appends(tmp_path: Path) -> None:
    paths = create_run_paths(task_id="task-003", base_dir=tmp_path)

    write_execution_log(paths, "line1", append=True)
    write_execution_log(paths, "line2", append=True)

    log_text = read_execution_log(paths)
    assert "line1\n" in log_text
    assert "line2\n" in log_text
    assert log_text.index("line1") < log_text.index("line2")


def test_save_workspace_files_supports_nested_paths(tmp_path: Path) -> None:
    paths = create_run_paths(task_id="task-004", base_dir=tmp_path)
    files = {
        "run.py": "print('run')\n",
        "analysis/analyze.py": "print('analyze')\n",
    }

    saved_paths = save_workspace_files(paths, files)

    assert len(saved_paths) == 2
    assert (paths.workspace_dir / "run.py").read_text(encoding="utf-8") == "print('run')\n"
    assert (
        paths.workspace_dir / "analysis" / "analyze.py"
    ).read_text(encoding="utf-8") == "print('analyze')\n"


def test_save_workspace_files_rejects_parent_traversal(tmp_path: Path) -> None:
    paths = create_run_paths(task_id="task-005", base_dir=tmp_path)

    with pytest.raises(ValueError):
        save_workspace_files(paths, {"../evil.py": "print('bad')\n"})


def test_build_arxiv_query_includes_keywords_and_categories() -> None:
    query = paper_search.build_arxiv_query(
        keywords=["transformer", "attention"],
        categories=["cs.LG", "cs.AI"],
    )
    assert 'all:"transformer"' in query
    assert 'all:"attention"' in query
    assert "cat:cs.LG" in query
    assert "cat:cs.AI" in query


def test_fetch_arxiv_raw_uses_arxiv_client(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class DummySearch:
        def __init__(self, query: str, max_results: int, sort_by: object, sort_order: object):
            captured["query"] = query
            captured["max_results"] = max_results
            captured["sort_by"] = sort_by
            captured["sort_order"] = sort_order

    class DummyClient:
        def __init__(self, page_size: int, delay_seconds: float, num_retries: int):
            captured["page_size"] = page_size
            captured["delay_seconds"] = delay_seconds
            captured["num_retries"] = num_retries

        def results(self, search: DummySearch):
            captured["search_received"] = search
            return iter(["r1", "r2"])

    fake_arxiv = SimpleNamespace(
        Search=DummySearch,
        Client=DummyClient,
        SortCriterion=SimpleNamespace(
            Relevance="RELEVANCE",
            SubmittedDate="SUBMITTED",
            LastUpdatedDate="UPDATED",
        ),
        SortOrder=SimpleNamespace(Descending="DESC"),
    )

    monkeypatch.setattr(paper_search, "arxiv", fake_arxiv)
    results = paper_search.fetch_arxiv_raw(
        query='all:"transformer"',
        max_results=2,
        sort_by="relevance",
    )

    assert results == ["r1", "r2"]
    assert captured["query"] == 'all:"transformer"'
    assert captured["max_results"] == 2
    assert captured["page_size"] == 2


def test_parse_and_format_arxiv_response() -> None:
    class DummyAuthor:
        def __init__(self, name: str):
            self.name = name

    class DummyResult:
        def __init__(self):
            self.title = "  Transformer Paper  "
            self.summary = "  This   is\n a summary. "
            self.authors = [DummyAuthor("Alice"), DummyAuthor("Bob")]
            self.published = datetime(2024, 1, 2, 3, 4, 5)
            self.updated = datetime(2024, 2, 3, 4, 5, 6)
            self.primary_category = "cs.LG"
            self.categories = ["cs.LG", "cs.AI"]
            self.pdf_url = "https://arxiv.org/pdf/1234.5678.pdf"
            self.entry_id = "http://arxiv.org/abs/1234.5678v1"

        def get_short_id(self) -> str:
            return "1234.5678v1"

    papers = paper_search.parse_arxiv_response([DummyResult()])
    assert len(papers) == 1
    assert papers[0].paper_id == "1234.5678v1"
    assert papers[0].title == "Transformer Paper"
    assert papers[0].summary == "This is a summary."
    assert papers[0].authors == ["Alice", "Bob"]

    formatted = paper_search.format_papers_for_llm(papers, max_summary_chars=100)
    assert "# ArXiv Search Results" in formatted
    assert "Transformer Paper" in formatted
    assert "Alice, Bob" in formatted
    assert "https://arxiv.org/pdf/1234.5678.pdf" in formatted
