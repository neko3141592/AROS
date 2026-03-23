from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Iterable, Mapping


def _is_enabled() -> bool:
    raw = os.getenv("AROS_CLI_LOG", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _emit(message: str) -> None:
    if not _is_enabled():
        return
    print(f"[{_timestamp()}] {message}")


def preview_text(text: Any, limit: int = 160) -> str:
    """
    長すぎるテキストを CLI 表示向けに短く整形する。
    """
    raw = str(text or "").replace("\n", "\\n").strip()
    if not raw:
        return "(empty)"
    if len(raw) <= limit:
        return raw
    return f"{raw[:limit]}... [truncated {len(raw) - limit} chars]"


def preview_list(values: Iterable[Any], limit: int = 5) -> str:
    """
    リスト値を CLI 表示向けに短く整形する。
    """
    items = [str(value) for value in values]
    if not items:
        return "(none)"
    if len(items) <= limit:
        return ", ".join(items)
    shown = ", ".join(items[:limit])
    return f"{shown}, ... (+{len(items) - limit} more)"


def log_banner(title: str) -> None:
    _emit(f"=== {title} ===")


def log_kv(label: str, value: Any) -> None:
    _emit(f"{label}: {value}")


def log_node_start(node_name: str, details: Mapping[str, Any] | None = None) -> None:
    _emit(f"[{node_name}] start")
    if not details:
        return
    for key, value in details.items():
        _emit(f"[{node_name}] {key}: {value}")


def log_node_end(node_name: str, details: Mapping[str, Any] | None = None) -> None:
    _emit(f"[{node_name}] end")
    if not details:
        return
    for key, value in details.items():
        _emit(f"[{node_name}] {key}: {value}")


def log_llm_request(
    node_name: str,
    model_name: str,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    tools: Iterable[str] | None = None,
    message_count: int | None = None,
) -> None:
    _emit(f"[{node_name}] LLM request -> model={model_name}")
    if message_count is not None:
        _emit(f"[{node_name}] message_count={message_count}")
    if tools is not None:
        _emit(f"[{node_name}] tools={preview_list(tools)}")
    if system_prompt is not None:
        _emit(f"[{node_name}] system_prompt={preview_text(system_prompt)}")
    if user_prompt is not None:
        _emit(f"[{node_name}] user_prompt={preview_text(user_prompt)}")


def log_llm_response(
    node_name: str,
    content: str | None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> None:
    tool_calls = tool_calls or []
    tool_names = [
        str((call.get("function") or {}).get("name") or "unknown")
        for call in tool_calls
        if isinstance(call, dict)
    ]
    _emit(
        f"[{node_name}] LLM response <- content={preview_text(content)} "
        f"tool_calls={preview_list(tool_names)}"
    )


def log_tool_result(node_name: str, tool_result: Mapping[str, Any]) -> None:
    tool_name = str(tool_result.get("name") or "unknown")
    ok = bool(tool_result.get("ok"))
    if ok:
        payload = preview_text(tool_result.get("result"))
        _emit(f"[{node_name}] tool={tool_name} ok result={payload}")
        return
    payload = preview_text(tool_result.get("error"))
    _emit(f"[{node_name}] tool={tool_name} failed error={payload}")


def log_execution(
    entrypoint: str,
    timeout_sec: float,
    returncode: int | None = None,
    duration_sec: float | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    skipped: bool = False,
) -> None:
    if skipped:
        _emit(
            f"[Evaluator] execution skipped entrypoint={entrypoint} "
            f"timeout={timeout_sec:.2f}s"
        )
        return

    _emit(
        f"[Evaluator] execution finished entrypoint={entrypoint} "
        f"timeout={timeout_sec:.2f}s returncode={returncode} "
        f"duration={duration_sec:.3f}s"
    )
    _emit(f"[Evaluator] stdout={preview_text(stdout)}")
    _emit(f"[Evaluator] stderr={preview_text(stderr)}")


def log_route(source: str, target: str, reason: str) -> None:
    _emit(f"[{source}] route -> {target} ({reason})")
