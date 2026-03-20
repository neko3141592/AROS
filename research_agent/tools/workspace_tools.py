from __future__ import annotations

from pathlib import Path

from tools.file_io import RunPaths


def _resolve_workspace_path(run_paths: RunPaths, rel_path: str) -> Path:
    """
    workspace 配下の安全な絶対パスへ変換する。

    制約:
    - 絶対パスを拒否
    - ".." を含むパスを拒否
    - resolve 後に workspace 配下であることを検証
    """
    raw = Path(rel_path)
    if raw.is_absolute():
        raise ValueError(f"Absolute path is not allowed: {rel_path}")
    if ".." in raw.parts:
        raise ValueError(f"Parent traversal is not allowed: {rel_path}")

    workspace_root = run_paths.workspace_dir.resolve()
    resolved = (workspace_root / raw).resolve()

    try:
        resolved.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace: {rel_path}") from exc

    return resolved


def _write_text_atomic(path: Path, content: str) -> None:
    """
    テキストをアトミックに保存する。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def _to_workspace_relative(run_paths: RunPaths, path: Path) -> str:
    return str(path.relative_to(run_paths.workspace_dir).as_posix())


def list_files(run_paths: RunPaths, base_dir: str = ".", recursive: bool = True) -> list[str]:
    target_dir = _resolve_workspace_path(run_paths, base_dir)
    if not target_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {base_dir}")

    if recursive:
        candidates = target_dir.rglob("*")
    else:
        candidates = target_dir.iterdir()

    files = [
        str(path.relative_to(run_paths.workspace_dir).as_posix())
        for path in candidates
        if path.is_file()
    ]
    return sorted(files)


def read_file(run_paths: RunPaths, file_path: str) -> str:
    target = _resolve_workspace_path(run_paths, file_path)
    if not target.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if target.is_dir():
        raise IsADirectoryError(f"Path is a directory: {file_path}")
    return target.read_text(encoding="utf-8")


def create_file(
    run_paths: RunPaths,
    file_path: str,
    content: str,
    overwrite: bool = False,
) -> str:
    target = _resolve_workspace_path(run_paths, file_path)

    if target.exists():
        if target.is_dir():
            raise IsADirectoryError(f"Path is a directory: {file_path}")
        if not overwrite:
            raise FileExistsError(
                f"File already exists (set overwrite=True to replace): {file_path}"
            )

    _write_text_atomic(target, content)
    return _to_workspace_relative(run_paths, target)


def edit_file(
    run_paths: RunPaths,
    file_path: str,
    start_line: int,
    end_line: int,
    new_text: str,
) -> str:
    if start_line < 1:
        raise ValueError("start_line must be >= 1.")
    if end_line < start_line:
        raise ValueError("end_line must be >= start_line.")

    target = _resolve_workspace_path(run_paths, file_path)
    if not target.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if target.is_dir():
        raise IsADirectoryError(f"Path is a directory: {file_path}")

    original = target.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    if not lines:
        raise ValueError("Cannot edit by line range: file is empty.")

    total_lines = len(lines)
    if start_line > total_lines or end_line > total_lines:
        raise ValueError(
            f"Line range out of bounds: {start_line}-{end_line} (total={total_lines})"
        )

    replacement_lines = new_text.splitlines(keepends=True)
    start_idx = start_line - 1
    end_idx_exclusive = end_line
    updated_lines = lines[:start_idx] + replacement_lines + lines[end_idx_exclusive:]
    updated = "".join(updated_lines)

    _write_text_atomic(target, updated)
    return _to_workspace_relative(run_paths, target)


def replace_string(
    run_paths: RunPaths,
    file_path: str,
    old: str,
    new: str,
    count: int = -1,
) -> str:
    if old == "":
        raise ValueError("old must not be empty.")
    if count < -1:
        raise ValueError("count must be -1 or >= 0.")

    target = _resolve_workspace_path(run_paths, file_path)
    if not target.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if target.is_dir():
        raise IsADirectoryError(f"Path is a directory: {file_path}")

    original = target.read_text(encoding="utf-8")
    occurrence_count = original.count(old)
    if occurrence_count == 0:
        raise ValueError(f"String not found in file: {old!r}")

    if count == -1:
        updated = original.replace(old, new)
    else:
        updated = original.replace(old, new, count)
        if count == 0:
            raise ValueError("count=0 results in no replacement.")

    if updated == original:
        raise ValueError("No replacement applied.")

    _write_text_atomic(target, updated)
    return _to_workspace_relative(run_paths, target)
