from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.execution_engine import LocalExecutionEngine, get_execution_engine  # noqa: E402


def test_get_execution_engine_defaults_to_local(monkeypatch) -> None:
    """
    test_get_execution_engine_defaults_to_local を実行する。
    """
    monkeypatch.delenv("EXECUTION_BACKEND", raising=False)
    engine = get_execution_engine()
    assert isinstance(engine, LocalExecutionEngine)
    assert engine.backend_name == "local"


def test_get_execution_engine_raises_for_runpod_until_implemented(monkeypatch) -> None:
    """
    test_get_execution_engine_raises_for_runpod_until_implemented を実行する。
    """
    monkeypatch.setenv("EXECUTION_BACKEND", "runpod")
    try:
        get_execution_engine()
        assert False, "Expected NotImplementedError"
    except NotImplementedError:
        assert True


def test_get_execution_engine_raises_for_unknown_backend(monkeypatch) -> None:
    """
    test_get_execution_engine_raises_for_unknown_backend を実行する。
    """
    monkeypatch.setenv("EXECUTION_BACKEND", "unknown_backend")
    try:
        get_execution_engine()
        assert False, "Expected ValueError"
    except ValueError:
        assert True

