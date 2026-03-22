from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.file_io import (  # noqa: E402
    _copy_workspace_snapshot,
    _get_runs_root,
    _resolve_parent_workspace,
    _sanitize_id,
    create_run_paths,
    read_execution_log,
    read_generated_code,
    save_generated_code,
    save_workspace_files,
    write_meta,
    write_execution_log,
)
from tools.local_executor import run_workspace_python  # noqa: E402
from tools.workspace_tools import (  # noqa: E402
    create_file as workspace_create_file,
    edit_file as workspace_edit_file,
    list_files as workspace_list_files,
    read_file as workspace_read_file,
    replace_string as workspace_replace_string,
    run_shell_command as workspace_run_shell_command,
)
import tools.paper_search as paper_search  # noqa: E402


def test_create_run_paths_creates_directories(tmp_path: Path) -> None:
    """
    test_create_run_paths_creates_directories を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    paths = create_run_paths(task_id="task-001", base_dir=tmp_path)

    assert paths.run_dir.exists()
    assert paths.run_dir.is_dir()
    assert paths.workspace_dir.exists()
    assert paths.workspace_dir.is_dir()
    assert paths.code_path == paths.workspace_dir / "main.py"
    assert paths.run_id.startswith("task-001_")
    assert paths.project_id is None
    assert paths.parent_run_id is None
    assert paths.workspace_source_run_id is None


def test_create_run_paths_with_project_id_uses_project_hierarchy(tmp_path: Path) -> None:
    """
    test_create_run_paths_with_project_id_uses_project_hierarchy を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    paths = create_run_paths(
        task_id="task-project-001",
        base_dir=tmp_path,
        project_id="my project",
    )

    expected_runs_root = (tmp_path / "projects" / "my_project" / "runs").resolve()
    assert paths.run_dir.parent == expected_runs_root
    assert paths.project_id == "my_project"
    assert paths.parent_run_id is None
    assert paths.workspace_source_run_id is None


def test_create_run_paths_auto_inherits_latest_workspace_for_project_id(
    tmp_path: Path,
) -> None:
    """
    test_create_run_paths_auto_inherits_latest_workspace_for_project_id を実行する。

    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    first = create_run_paths(
        task_id="task-project-seed",
        base_dir=tmp_path,
        project_id="project-z",
    )
    save_workspace_files(
        first,
        {
            "main.py": "print('seed')\n",
            "notes/context.txt": "keep this context\n",
        },
    )

    second = create_run_paths(
        task_id="task-project-next",
        base_dir=tmp_path,
        project_id="project-z",
    )

    assert second.parent_run_id == first.run_id
    assert second.workspace_source_run_id == first.run_id
    assert second.workspace_dir != first.workspace_dir
    assert (second.workspace_dir / "main.py").read_text(encoding="utf-8") == "print('seed')\n"
    assert (
        second.workspace_dir / "notes" / "context.txt"
    ).read_text(encoding="utf-8") == "keep this context\n"


def test_create_run_paths_prefers_explicit_parent_over_latest_project_run(
    tmp_path: Path,
) -> None:
    """
    test_create_run_paths_prefers_explicit_parent_over_latest_project_run を実行する。

    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    first = create_run_paths(
        task_id="task-project-first",
        base_dir=tmp_path,
        project_id="project-priority",
    )
    save_workspace_files(first, {"main.py": "print('first')\n"})

    second = create_run_paths(
        task_id="task-project-second",
        base_dir=tmp_path,
        project_id="project-priority",
    )
    save_workspace_files(second, {"main.py": "print('second')\n"})

    third = create_run_paths(
        task_id="task-project-third",
        base_dir=tmp_path,
        project_id="project-priority",
        parent_run_id=first.run_id,
    )

    assert third.parent_run_id == first.run_id
    assert third.workspace_source_run_id == first.run_id
    assert (third.workspace_dir / "main.py").read_text(encoding="utf-8") == "print('first')\n"


def test_create_run_paths_inherits_workspace_from_parent_run(tmp_path: Path) -> None:
    """
    test_create_run_paths_inherits_workspace_from_parent_run を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    parent = create_run_paths(
        task_id="task-parent",
        base_dir=tmp_path,
        project_id="project-x",
    )
    save_workspace_files(
        parent,
        {
            "main.py": "print('parent')\n",
            "src/module.py": "VALUE = 1\n",
        },
    )

    child = create_run_paths(
        task_id="task-child",
        base_dir=tmp_path,
        project_id="project-x",
        parent_run_id=parent.run_id,
    )

    assert child.parent_run_id == parent.run_id
    assert child.workspace_source_run_id == parent.run_id
    assert child.workspace_dir != parent.workspace_dir
    assert (child.workspace_dir / "main.py").read_text(encoding="utf-8") == "print('parent')\n"
    assert (
        child.workspace_dir / "src" / "module.py"
    ).read_text(encoding="utf-8") == "VALUE = 1\n"


def test_create_run_paths_invalid_parent_falls_back_to_new_workspace(tmp_path: Path) -> None:
    """
    test_create_run_paths_invalid_parent_falls_back_to_new_workspace を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    paths = create_run_paths(
        task_id="task-fallback",
        base_dir=tmp_path,
        project_id="project-y",
        parent_run_id="missing-run-id",
    )

    assert paths.parent_run_id == "missing-run-id"
    assert paths.workspace_source_run_id is None
    assert paths.workspace_dir.exists()
    assert paths.workspace_dir.is_dir()
    assert list(paths.workspace_dir.rglob("*")) == []


def test_sanitize_id_replaces_unsafe_patterns() -> None:
    """
    test_sanitize_id_replaces_unsafe_patterns を実行する。
    
    Args:
        なし。
    """
    assert _sanitize_id("  proj/../alpha beta  ") == "proj_alpha_beta"


def test_get_runs_root_with_project_id(tmp_path: Path) -> None:
    """
    test_get_runs_root_with_project_id を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    runs_root = _get_runs_root(tmp_path, project_id="my project")
    assert runs_root == (tmp_path / "projects" / "my_project" / "runs").resolve()


def test_resolve_parent_workspace_under_same_root(tmp_path: Path) -> None:
    """
    test_resolve_parent_workspace_under_same_root を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    runs_root = tmp_path / "runs"
    parent_run_dir = runs_root / "run_001"
    parent_workspace = parent_run_dir / "workspace"
    parent_workspace.mkdir(parents=True)
    (parent_workspace / "main.py").write_text("print('ok')\n", encoding="utf-8")

    resolved = _resolve_parent_workspace(runs_root, "run_001")
    assert resolved == parent_workspace.resolve()


def test_resolve_parent_workspace_rejects_missing_parent(tmp_path: Path) -> None:
    """
    test_resolve_parent_workspace_rejects_missing_parent を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    with pytest.raises(FileNotFoundError):
        _resolve_parent_workspace(tmp_path / "runs", "missing_run")


def test_copy_workspace_snapshot_copies_files(tmp_path: Path) -> None:
    """
    test_copy_workspace_snapshot_copies_files を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    src = tmp_path / "src_workspace"
    dst = tmp_path / "dst_workspace"
    src.mkdir(parents=True)
    (src / "main.py").write_text("print('snapshot')\n", encoding="utf-8")

    copied = _copy_workspace_snapshot(src, dst)

    assert copied == dst.resolve()
    assert (dst / "main.py").read_text(encoding="utf-8") == "print('snapshot')\n"


def test_copy_workspace_snapshot_rejects_existing_destination(tmp_path: Path) -> None:
    """
    test_copy_workspace_snapshot_rejects_existing_destination を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    src = tmp_path / "src_workspace"
    dst = tmp_path / "dst_workspace"
    src.mkdir(parents=True)
    dst.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        _copy_workspace_snapshot(src, dst)


def test_save_and_read_generated_code_roundtrip(tmp_path: Path) -> None:
    """
    test_save_and_read_generated_code_roundtrip を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    paths = create_run_paths(task_id="task-002", base_dir=tmp_path)
    code = "print('hello')\n"

    save_generated_code(paths, code)
    loaded = read_generated_code(paths)

    assert loaded == code
    assert paths.code_path.exists()
    assert paths.code_path == paths.workspace_dir / "main.py"
    assert not (paths.run_dir / "main.py").exists()


def test_write_execution_log_appends(tmp_path: Path) -> None:
    """
    test_write_execution_log_appends を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    paths = create_run_paths(task_id="task-003", base_dir=tmp_path)

    write_execution_log(paths, "line1", append=True)
    write_execution_log(paths, "line2", append=True)

    log_text = read_execution_log(paths)
    assert "line1\n" in log_text
    assert "line2\n" in log_text
    assert log_text.index("line1") < log_text.index("line2")


def test_save_workspace_files_supports_nested_paths(tmp_path: Path) -> None:
    """
    test_save_workspace_files_supports_nested_paths を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
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
    """
    test_save_workspace_files_rejects_parent_traversal を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    paths = create_run_paths(task_id="task-005", base_dir=tmp_path)

    with pytest.raises(ValueError):
        save_workspace_files(paths, {"../evil.py": "print('bad')\n"})


def test_workspace_create_and_read_file_roundtrip(tmp_path: Path) -> None:
    """
    test_workspace_create_and_read_file_roundtrip を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    paths = create_run_paths(task_id="task-005b", base_dir=tmp_path)

    rel = workspace_create_file(paths, "notes/readme.txt", "hello workspace\n")
    loaded = workspace_read_file(paths, rel)

    assert rel == "notes/readme.txt"
    assert loaded == "hello workspace\n"


def test_workspace_create_file_rejects_overwrite_by_default(tmp_path: Path) -> None:
    """
    test_workspace_create_file_rejects_overwrite_by_default を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    paths = create_run_paths(task_id="task-005c", base_dir=tmp_path)
    workspace_create_file(paths, "main.py", "print('v1')\n")

    with pytest.raises(FileExistsError):
        workspace_create_file(paths, "main.py", "print('v2')\n")


def test_workspace_edit_file_replaces_line_range(tmp_path: Path) -> None:
    """
    test_workspace_edit_file_replaces_line_range を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    paths = create_run_paths(task_id="task-005d", base_dir=tmp_path)
    workspace_create_file(paths, "sample.txt", "A\nB\nC\n")

    rel = workspace_edit_file(paths, "sample.txt", start_line=2, end_line=3, new_text="X\nY\n")
    loaded = workspace_read_file(paths, "sample.txt")

    assert rel == "sample.txt"
    assert loaded == "A\nX\nY\n"


def test_workspace_replace_string_supports_count(tmp_path: Path) -> None:
    """
    test_workspace_replace_string_supports_count を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    paths = create_run_paths(task_id="task-005e", base_dir=tmp_path)
    workspace_create_file(paths, "sample.txt", "foo foo foo\n")

    rel = workspace_replace_string(paths, "sample.txt", old="foo", new="bar", count=2)
    loaded = workspace_read_file(paths, "sample.txt")

    assert rel == "sample.txt"
    assert loaded == "bar bar foo\n"


def test_workspace_tools_reject_parent_traversal(tmp_path: Path) -> None:
    """
    test_workspace_tools_reject_parent_traversal を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    paths = create_run_paths(task_id="task-005f", base_dir=tmp_path)

    with pytest.raises(ValueError):
        workspace_list_files(paths, base_dir="../")


def test_run_shell_command_supports_allowlisted_readonly_commands(
    tmp_path: Path,
) -> None:
    """
    test_run_shell_command_supports_allowlisted_readonly_commands を実行する。

    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    paths = create_run_paths(task_id="task-005g", base_dir=tmp_path)
    workspace_create_file(paths, "notes.txt", "alpha\nbeta\n")

    output = workspace_run_shell_command(paths, "rg beta notes.txt")

    assert "beta" in output


@pytest.mark.parametrize(
    ("command", "expected_message"),
    [
        ("cat ../secret.txt", "denied"),
        ("cat /etc/passwd", "denied"),
        ("rg beta notes.txt | cat", "denied"),
        ("cat notes.txt > out.txt", "denied"),
    ],
)
def test_run_shell_command_rejects_unsafe_commands(
    tmp_path: Path,
    command: str,
    expected_message: str,
) -> None:
    """
    test_run_shell_command_rejects_unsafe_commands を実行する。

    Args:
        tmp_path: pytestの一時ディレクトリパス。
        command: 実行対象コマンド。
        expected_message: 期待するエラーメッセージ断片。
    """
    paths = create_run_paths(task_id="task-005h", base_dir=tmp_path)
    workspace_create_file(paths, "notes.txt", "alpha\nbeta\n")

    with pytest.raises(ValueError, match=expected_message):
        workspace_run_shell_command(paths, command)


def test_write_meta_persists_file_list(tmp_path: Path) -> None:
    """
    test_write_meta_persists_file_list を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    parent = create_run_paths(
        task_id="task-006-parent",
        base_dir=tmp_path,
        project_id="project-meta",
    )
    save_workspace_files(parent, {"main.py": "print('meta parent')\n"})
    paths = create_run_paths(
        task_id="task-006",
        base_dir=tmp_path,
        project_id="project-meta",
        parent_run_id=parent.run_id,
    )

    write_meta(paths, task_id="task-006", files=["main.py", "analysis/run.py"])
    payload = json.loads(paths.meta_path.read_text(encoding="utf-8"))

    assert payload["task_id"] == "task-006"
    assert payload["files"] == ["main.py", "analysis/run.py"]
    assert payload["project_id"] == "project-meta"
    assert payload["parent_run_id"] == parent.run_id
    assert payload["workspace_source_run_id"] == parent.run_id


def test_run_workspace_python_captures_outputs_and_returncode(tmp_path: Path) -> None:
    """
    test_run_workspace_python_captures_outputs_and_returncode を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    paths = create_run_paths(task_id="task-007", base_dir=tmp_path)
    save_workspace_files(
        paths,
        {
            "main.py": (
                "import sys\n"
                "print('hello stdout')\n"
                "print('hello stderr', file=sys.stderr)\n"
                "raise SystemExit(3)\n"
            )
        },
    )

    result = run_workspace_python(paths)

    assert "hello stdout" in result.stdout
    assert "hello stderr" in result.stderr
    assert result.returncode == 3
    assert result.duration_sec >= 0.0


def test_run_workspace_python_returns_timeout_code(tmp_path: Path) -> None:
    """
    test_run_workspace_python_returns_timeout_code を実行する。
    
    Args:
        tmp_path: pytestの一時ディレクトリパス。
    """
    paths = create_run_paths(task_id="task-008", base_dir=tmp_path)
    save_workspace_files(
        paths,
        {"main.py": "import time\ntime.sleep(1.0)\nprint('done')\n"},
    )

    result = run_workspace_python(paths, timeout_sec=0.01)

    assert result.returncode == 124
    assert "timed out" in result.stderr.lower()
    assert result.duration_sec >= 0.0


def test_build_arxiv_query_includes_keywords_and_categories() -> None:
    """
    test_build_arxiv_query_includes_keywords_and_categories を実行する。
    
    Args:
        なし。
    """
    query = paper_search.build_arxiv_query(
        keywords=["transformer", "attention"],
        categories=["cs.LG", "cs.AI"],
    )
    assert 'all:"transformer"' in query
    assert 'all:"attention"' in query
    assert "cat:cs.LG" in query
    assert "cat:cs.AI" in query


def test_fetch_arxiv_raw_uses_arxiv_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    test_fetch_arxiv_raw_uses_arxiv_client を実行する。
    
    Args:
        monkeypatch: pytestの monkeypatch フィクスチャ。
    """
    captured: dict[str, object] = {}

    class DummySearch:
        def __init__(self, query: str, max_results: int, sort_by: object, sort_order: object):
            """
            __init__ を実行する。
            
            Args:
                query: 検索クエリ文字列。
                max_results: 取得する最大件数。
                sort_by: ソート基準。
                sort_order: ソート順。
            """
            captured["query"] = query
            captured["max_results"] = max_results
            captured["sort_by"] = sort_by
            captured["sort_order"] = sort_order

    class DummyClient:
        def __init__(self, page_size: int, delay_seconds: float, num_retries: int):
            """
            __init__ を実行する。
            
            Args:
                page_size: ページサイズ。
                delay_seconds: リクエスト間待機秒数。
                num_retries: リトライ回数。
            """
            captured["page_size"] = page_size
            captured["delay_seconds"] = delay_seconds
            captured["num_retries"] = num_retries

        def results(self, search: DummySearch):
            """
            results を実行する。
            
            Args:
                search: 検索オブジェクト。
            """
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
    """
    test_parse_and_format_arxiv_response を実行する。
    
    Args:
        なし。
    """
    class DummyAuthor:
        def __init__(self, name: str):
            """
            __init__ を実行する。
            
            Args:
                name: 名称文字列。
            """
            self.name = name

    class DummyResult:
        def __init__(self):
            """
            __init__ を実行する。
            
            Args:
                なし。
            """
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
            """
            get_short_id を実行する。
            
            Args:
                なし。
            """
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
