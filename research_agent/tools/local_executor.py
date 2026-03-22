from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from tools.file_io import RunPaths


@dataclass(frozen=True)
class LocalExecutionResult:
    """
    Workspace 上の Python 実行結果を保持する。
    """

    stdout: str
    stderr: str
    returncode: int
    duration_sec: float


def _validate_relative_entrypoint(entrypoint: str) -> Path:
    """
    実行対象が workspace 配下の相対パスであることを検証する。
    
    Args:
        entrypoint: 実行するエントリーポイントの相対パス。
    """
    p = Path(entrypoint)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Unsafe entrypoint path: {entrypoint}")
    return p


def _coerce_output(value: str | bytes | None) -> str:
    """
    _coerce_output を実行する。
    
    Args:
        value: 変換対象の値。
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_workspace_python(
    run_paths: RunPaths, entrypoint: str = "main.py", timeout_sec: float = 60
) -> LocalExecutionResult:
    """
    workspace_dir 上の Python エントリポイントを実行し、標準出力/標準エラー/終了コードを返す。
    
    Args:
        run_paths: 実行ディレクトリ群を保持する RunPaths。
        entrypoint: 実行するエントリーポイントの相対パス。
        timeout_sec: 処理タイムアウト秒数。
    """
    if timeout_sec <= 0:
        raise ValueError("timeout_sec must be > 0.")

    entry_rel = _validate_relative_entrypoint(entrypoint)
    target = run_paths.workspace_dir / entry_rel
    if not target.exists():
        return LocalExecutionResult(
            stdout="",
            stderr=f"Entrypoint not found: {entry_rel.as_posix()}",
            returncode=127,
            duration_sec=0.0,
        )

    cmd = [sys.executable, entry_rel.as_posix()]
    start = time.monotonic()
    try:
        completed = subprocess.run(
            cmd,
            cwd=run_paths.workspace_dir,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return LocalExecutionResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
            duration_sec=max(0.0, time.monotonic() - start),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _coerce_output(exc.stdout)
        stderr = _coerce_output(exc.stderr)
        timeout_message = f"Execution timed out after {timeout_sec} seconds."
        if stderr:
            stderr = f"{stderr.rstrip()}\n{timeout_message}"
        else:
            stderr = timeout_message

        return LocalExecutionResult(
            stdout=stdout,
            stderr=stderr,
            returncode=124,
            duration_sec=max(0.0, time.monotonic() - start),
        )
