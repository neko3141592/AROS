from __future__ import annotations

import os
from typing import Protocol

from tools.file_io import RunPaths
from tools.local_executor import LocalExecutionResult, run_workspace_python


class ExecutionEngine(Protocol):
    """
    Evaluator が利用する実行エンジンの共通インターフェース。
    """

    backend_name: str

    def run(
        self,
        run_paths: RunPaths,
        entrypoint: str,
        timeout_sec: float,
    ) -> LocalExecutionResult:
        """
        workspace 上で entrypoint を実行し、結果を返す。
        """


class LocalExecutionEngine:
    """
    既存のローカル Python 実行をラップする ExecutionEngine 実装。
    """

    backend_name = "local"

    def run(
        self,
        run_paths: RunPaths,
        entrypoint: str,
        timeout_sec: float,
    ) -> LocalExecutionResult:
        return run_workspace_python(
            run_paths=run_paths,
            entrypoint=entrypoint,
            timeout_sec=timeout_sec,
        )


def get_execution_engine(backend: str | None = None) -> ExecutionEngine:
    """
    実行バックエンド名から ExecutionEngine を生成する。

    Args:
        backend: 実行バックエンド名。未指定時は環境変数 EXECUTION_BACKEND を参照。
    """
    resolved = (backend or os.getenv("EXECUTION_BACKEND", "local")).strip().lower()
    if not resolved:
        resolved = "local"

    if resolved == "local":
        return LocalExecutionEngine()

    if resolved == "runpod":
        raise NotImplementedError(
            "EXECUTION_BACKEND=runpod is not implemented yet. "
            "Use EXECUTION_BACKEND=local for now."
        )

    raise ValueError(f"Unsupported EXECUTION_BACKEND: {resolved}")

