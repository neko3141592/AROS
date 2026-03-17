from __future__ import annotations
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

# デフォルトの保存先を定義
DEFAULT_BASE_DIR = Path(__file__).resolve().parents[1] / "storage" / "runs"

@dataclass(frozen=True)
class RunPaths:
    """
    1回のリサーチ実行に関連する全てのファイルパスを管理するデータクラス。
    """
    run_id: str         # 実行の一意なID
    run_dir: Path       # この実行のための専用ディレクトリ
    code_path: Path     # 生成されたメインコード(main.py)のパス
    log_path: Path      # 実行ログ(execution.log)のパス
    workspace_dir: Path # 作業用一時ファイルなどが置かれるディレクトリ
    meta_path: Path     # 実行のメタ情報(meta.json)のパス


def create_run_paths(task_id: str, base_dir: Path | None = None) -> RunPaths:
    """
    新しい実行用ディレクトリを作成し、関連するパスを生成して返す。
    
    Args:
        task_id: 親タスクのID
        base_dir: ベースとなるディレクトリ。未指定時は storage/runs/ を使用。
    """
    base = base_dir or DEFAULT_BASE_DIR
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:8]
    # ファイル名に使用できない文字を置換
    safe_task_id = task_id.replace("/", "_").replace(" ", "_")
    
    # 実行ごとにユニークなディレクトリ名を作成 (例: task1_20240317_abcdef12)
    run_id = f"{safe_task_id}_{ts}_{short}"
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 作業領域(workspace)の作成
    workspace_dir = run_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        code_path=run_dir / "main.py",
        log_path=run_dir / "execution.log",
        workspace_dir=workspace_dir,
        meta_path=run_dir / "meta.json",
    )

def _write_text_atomic(path: Path, content: str) -> None:
    """
    ファイルをアトミック（原子的に）に書き込む。
    一時ファイルに書き込んでからリネームすることで、書き込み途中の破損を防ぐ。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)

def _validate_relpath(relpath: str) -> Path:
    """
    相対パスが安全であることを検証する（ディレクトリトラバーサル防止）。
    """
    p = Path(relpath)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Unsafe relative path: {relpath}")
    return p

def save_workspace_files(paths: RunPaths, files: Mapping[str, str]) -> list[Path]:
    """
    複数のファイルを workspace ディレクトリ内に一括保存する。
    
    Args:
        paths: RunPaths オブジェクト
        files: {相対パス: ファイル内容} のマッピング
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
    """
    target = paths.code_path if filename == "main.py" else paths.run_dir / filename
    _write_text_atomic(target, code)
    return target

def write_execution_log(paths: RunPaths, text: str, append: bool = True) -> Path:
    """
    実行ログ（標準出力やエラーなど）をファイルに記録する。
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
    """
    if not paths.log_path.exists():
        return ""
    return paths.log_path.read_text(encoding="utf-8")

def read_generated_code(paths: RunPaths, filename: str = "main.py") -> str:
    """
    保存されている生成コードを読み込む。
    """
    target = paths.code_path if filename == "main.py" else paths.run_dir / filename
    if not target.exists():
        return ""
    return target.read_text(encoding="utf-8")

def write_meta(paths: RunPaths, task_id: str, files: list[str]) -> Path:
    """
    実行のメタ情報（タスクID、作成日時、関連ファイル等）をJSONとして保存する。
    """
    payload = {
        "run_id": paths.run_id,
        "task_id": task_id,
        "created_at": datetime.now().isoformat(),
        "files": files,
    }
    _write_text_atomic(paths.meta_path, json.dumps(payload, ensure_ascii=False, indent=2))
    return paths.meta_path