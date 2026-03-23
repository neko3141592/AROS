from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_EXECUTION_ENTRYPOINT = "main.py"


def resolve_execution_entrypoint(
    task: Any | None,
    state_entrypoint: str | None = None,
    default: str = DEFAULT_EXECUTION_ENTRYPOINT,
) -> str:
    """
    Task / state から実行対象の相対エントリーポイントを解決する。
    """
    raw_value = state_entrypoint
    if raw_value is None and task is not None:
        raw_value = getattr(task, "execution_entrypoint", None)

    entrypoint = str(raw_value or default).strip() or default
    entrypoint_path = Path(entrypoint)
    if entrypoint_path.is_absolute() or ".." in entrypoint_path.parts:
        raise ValueError(f"Unsafe execution entrypoint: {entrypoint}")
    return entrypoint_path.as_posix()


def format_subtasks_for_prompt(subtasks: Sequence[Any] | None) -> list[str]:
    """
    プロンプト埋め込み向けにサブタスク一覧を整形する。
    """
    if not subtasks:
        return []

    formatted: list[str] = []
    for item in subtasks:
        title = str(getattr(item, "title", "") or "").strip()
        description = str(getattr(item, "description", "") or "").strip()
        assigned_agent = str(getattr(item, "assigned_agent", "") or "").strip()

        prefix = f"[{assigned_agent}] " if assigned_agent else ""
        if title and description:
            formatted.append(f"{prefix}{title}: {description}")
        elif title:
            formatted.append(f"{prefix}{title}")
        elif description:
            formatted.append(f"{prefix}{description}")

    return formatted


def summarize_failure_for_prompt(
    evaluator_feedback: Mapping[str, Any] | None,
    execution_stderr: str | None,
) -> dict[str, str]:
    """
    Researcher / Coder が参照しやすい失敗要約を返す。
    """
    feedback = evaluator_feedback or {}
    return {
        "summary": str(feedback.get("summary") or ""),
        "likely_cause": str(feedback.get("likely_cause") or ""),
        "stderr": str(execution_stderr or feedback.get("stderr") or "").strip(),
    }
