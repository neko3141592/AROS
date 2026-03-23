from __future__ import annotations
import json
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

# デフォルトの保存先を定義
DEFAULT_BASE_DIR = Path(__file__).resolve().parents[1] / "storage" / "runs"


def _sanitize_id(value: str) -> str:
    """
    ID文字列をファイルシステム安全な形に正規化する。
    
    Args:
        value: 正規化対象のID文字列。
    
        Returns:
        ファイルシステム上で安全に使えるID文字列。
    """
    if not isinstance(value, str):
        raise TypeError("id must be a string.")

    cleaned = value.strip()
    if not cleaned:
        raise ValueError("id must not be empty.")

    cleaned = cleaned.replace("/", "_").replace("\\", "_")
    cleaned = cleaned.replace("..", "_")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")

    if not cleaned:
        raise ValueError("id became empty after sanitization.")
    return cleaned


def _get_runs_root(base_dir: Path | None, project_id: str | None) -> Path:
    """
    run保存先のルートを返す。
    
    Args:
        base_dir: ベースディレクトリ。None の場合は既定ルートを使う。
        project_id: プロジェクトID。指定時は project/runs 階層を返す。
    
        Returns:
        run を格納するルートディレクトリ。
    """
    if project_id is None:
        return (base_dir or DEFAULT_BASE_DIR).resolve()

    safe_project_id = _sanitize_id(project_id)
    if base_dir is None:
        storage_root = DEFAULT_BASE_DIR.parent
        return (storage_root / "projects" / safe_project_id / "runs").resolve()
    return (base_dir / "projects" / safe_project_id / "runs").resolve()


def _resolve_parent_workspace(runs_root: Path, parent_run_id: str) -> Path:
    """
    parent_run_id から親runの workspace を解決する。
    同一 runs_root 配下のみ許可する。
    
    Args:
        runs_root: 親runを探索する runs ルート。
        parent_run_id: 継承元の親run ID。
    
        Returns:
        親runの workspace ディレクトリパス。
    """
    root = runs_root.resolve()
    safe_parent_id = _sanitize_id(parent_run_id)
    parent_run_dir = (root / safe_parent_id).resolve()

    try:
        parent_run_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("parent_run_id must resolve under runs_root.") from exc

    if not parent_run_dir.exists() or not parent_run_dir.is_dir():
        raise FileNotFoundError(f"Parent run not found: {safe_parent_id}")

    parent_workspace = (parent_run_dir / "workspace").resolve()
    if not parent_workspace.exists() or not parent_workspace.is_dir():
        raise FileNotFoundError(
            f"Parent workspace not found for run: {safe_parent_id}"
        )
    return parent_workspace


def _find_latest_run_id(runs_root: Path, exclude_run_id: str | None = None) -> str | None:
    """
    runs_root 配下から最新の run_id を返す。

    Args:
        runs_root: run ディレクトリを格納するルート。
        exclude_run_id: 除外したい run_id。

    Returns:
        最新 run の run_id。見つからなければ None。
    """
    root = runs_root.resolve()
    candidates: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if exclude_run_id and child.name == exclude_run_id:
            continue
        if not (child / "workspace").is_dir():
            continue
        candidates.append(child)

    if not candidates:
        return None

    latest = max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
    return latest.name


def _copy_workspace_snapshot(src: Path, dst: Path) -> Path:
    """
    workspace をスナップショットとしてコピーする。
    既存dstは上書きせずエラーにする。
    
    Args:
        src: コピー元 workspace ディレクトリ。
        dst: コピー先 workspace ディレクトリ。
    
        Returns:
        作成されたコピー先ディレクトリパス。
    """
    src_dir = src.resolve()
    dst_dir = dst.resolve()

    if not src_dir.exists() or not src_dir.is_dir():
        raise FileNotFoundError(f"Source workspace not found: {src_dir}")
    if dst_dir.exists():
        raise FileExistsError(f"Destination already exists: {dst_dir}")

    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, dst_dir)
    return dst_dir

@dataclass(frozen=True)
class RunPaths:
    """
    1回のリサーチ実行に関連する全てのファイルパスを管理するデータクラス。
    """
    run_id: str         # 実行の一意なID
    run_dir: Path       # この実行のための専用ディレクトリ
    code_path: Path     # 生成されたメインコード(main.py)の正本パス（workspace/main.py）
    log_path: Path      # 実行ログ(execution.log)のパス
    workspace_dir: Path # 作業用一時ファイルなどが置かれるディレクトリ
    meta_path: Path     # 実行のメタ情報(meta.json)のパス
    project_id: str | None
    parent_run_id: str | None
    workspace_source_run_id: str | None


def create_run_paths(
    task_id: str,
    base_dir: Path | None = None,
    project_id: str | None = None,
    parent_run_id: str | None = None,
) -> RunPaths:
    """
    新しい実行用ディレクトリを作成し、関連するパスを生成して返す。
    
    Args:
        task_id: 親タスクのID。
        base_dir: ベースとなるディレクトリ。未指定時は storage/runs/ を使用。
        project_id: プロジェクトID。指定時は project/runs 階層に保存する。
        parent_run_id: 親run ID。指定時は親workspaceをコピーして引き継ぐ。
    
        Returns:
        実行ディレクトリ一式を指す RunPaths。
    """
    safe_project_id = _sanitize_id(project_id) if project_id else None
    explicit_parent_run_id = _sanitize_id(parent_run_id) if parent_run_id else None

    runs_root = _get_runs_root(base_dir, safe_project_id)
    runs_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    safe_task_id = _sanitize_id(task_id)
    
    # 実行ごとにユニークなディレクトリ名を作成 (例: task1_20240317_abcdef12)
    run_id = f"{safe_task_id}_{ts}_{short}"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_parent_run_id = explicit_parent_run_id
    if resolved_parent_run_id is None and safe_project_id is not None:
        resolved_parent_run_id = _find_latest_run_id(
            runs_root=runs_root,
            exclude_run_id=run_id,
        )

    # 作業領域(workspace)の作成
    workspace_dir = run_dir / "workspace"
    workspace_source_run_id: str | None = None
    if resolved_parent_run_id:
        try:
            parent_workspace = _resolve_parent_workspace(
                runs_root=runs_root, parent_run_id=resolved_parent_run_id
            )
            _copy_workspace_snapshot(parent_workspace, workspace_dir)
            workspace_source_run_id = parent_workspace.parent.name
        except (FileNotFoundError, ValueError):
            # 親runが不正/不存在の場合は新規workspaceへフォールバック（後方互換）
            workspace_dir.mkdir(parents=True, exist_ok=True)
    else:
        workspace_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        code_path=workspace_dir / "main.py",
        log_path=run_dir / "execution.log",
        workspace_dir=workspace_dir,
        meta_path=run_dir / "meta.json",
        project_id=safe_project_id,
        parent_run_id=resolved_parent_run_id,
        workspace_source_run_id=workspace_source_run_id,
    )

def _write_text_atomic(path: Path, content: str) -> None:
    """
    ファイルをアトミック（原子的に）に書き込む。
    一時ファイルに書き込んでからリネームすることで、書き込み途中の破損を防ぐ。
    
    Args:
        path: 書き込み先ファイルパス。
        content: 書き込む文字列。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)

def _validate_relpath(relpath: str) -> Path:
    """
    相対パスが安全であることを検証する（ディレクトリトラバーサル防止）。
    
    Args:
        relpath: 検証対象の相対パス文字列。
    
        Returns:
        検証済みの Path オブジェクト。
    """
    p = Path(relpath)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Unsafe relative path: {relpath}")
    return p

def save_workspace_files(paths: RunPaths, files: Mapping[str, str]) -> list[Path]:
    """
    複数のファイルを workspace ディレクトリ内に一括保存する。
    
    Args:
        paths: RunPaths オブジェクト。
        files: {相対パス: ファイル内容} のマッピング。
    
        Returns:
        保存したファイルの絶対パス一覧。
    """
    saved: list[Path] = []
    for rel, content in files.items():
        rel_path = _validate_relpath(rel)
        out = paths.workspace_dir / rel_path
        _write_text_atomic(out, content)
        saved.append(out)
    return saved

def save_generated_code(paths: RunPaths, code: str, filename: str = "main.py") -> Path:
    """
    生成されたコードを指定されたファイル名で保存する。
    
    Args:
        paths: RunPaths オブジェクト。
        code: 保存するコード文字列。
        filename: 保存先ファイル名。
    
        Returns:
        保存先ファイルパス。
    """
    rel_path = _validate_relpath(filename)
    target = paths.workspace_dir / rel_path
    _write_text_atomic(target, code)
    return target

def write_execution_log(paths: RunPaths, text: str, append: bool = True) -> Path:
    """
    実行ログ（標準出力やエラーなど）をファイルに記録する。
    
    Args:
        paths: RunPaths オブジェクト。
        text: 書き込むログ文字列。
        append: 追記モードで書くかどうか。
    
        Returns:
        実行ログファイルパス。
    """
    paths.log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with paths.log_path.open(mode, encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
    return paths.log_path

def read_execution_log(paths: RunPaths) -> str:
    """
    保存されている実行ログを読み込む。
    
    Args:
        paths: RunPaths オブジェクト。
    
        Returns:
        実行ログ全文。未作成なら空文字。
    """
    if not paths.log_path.exists():
        return ""
    return paths.log_path.read_text(encoding="utf-8")

def read_generated_code(paths: RunPaths, filename: str = "main.py") -> str:
    """
    保存されている生成コードを読み込む。
    
    Args:
        paths: RunPaths オブジェクト。
        filename: 読み込むファイル名。
    
        Returns:
        コード文字列。未作成なら空文字。
    """
    target = paths.code_path if filename == "main.py" else paths.run_dir / filename
    if not target.exists():
        return ""
    return target.read_text(encoding="utf-8")

def write_meta(paths: RunPaths, task_id: str, files: list[str]) -> Path:
    """
    実行のメタ情報（タスクID、作成日時、関連ファイル等）をJSONとして保存する。
    
    Args:
        paths: RunPaths オブジェクト。
        task_id: 対象タスクID。
        files: 関連ファイルの相対パス一覧。
    
        Returns:
        meta.json の保存先パス。
    """
    payload = {
        "run_id": paths.run_id,
        "project_id": paths.project_id,
        "parent_run_id": paths.parent_run_id,
        "workspace_source_run_id": paths.workspace_source_run_id,
        "task_id": task_id,
        "created_at": datetime.now().isoformat(),
        "files": files,
    }
    _write_text_atomic(paths.meta_path, json.dumps(payload, ensure_ascii=False, indent=2))
    return paths.meta_path
