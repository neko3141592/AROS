from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any

from schema.llm_outputs import PlannerOutput
from schema.task import SubTask


def _strip_code_fence(text: str) -> str:
    """
    LLMが返してきた ```json ... ``` を除去する。

    Args:
        text: 処理対象のテキスト。
    """
    s = text.strip()
    lines = s.splitlines()
    if not lines:
        return s

    if not lines[0].strip().startswith("```"):
        return s

    lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines).strip()


def _extract_first_json_payload(text: str) -> Any:
    """
    LLM出力から最初の JSON オブジェクト/配列を切り出して decode する。

    Args:
        text: 処理対象のテキスト。
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            return obj
        except JSONDecodeError:
            continue
    raise JSONDecodeError("No valid JSON payload found in text.", text, 0)


def _parse_planner_output(raw_text: str) -> PlannerOutput:
    """
    _parse_planner_output を実行する。

    Args:
        raw_text: LLMから受け取った生テキスト。
    """
    cleaned = _strip_code_fence(raw_text)
    payload = _extract_first_json_payload(cleaned)

    if isinstance(payload, list):
        payload = {"subtasks": payload}
    elif not isinstance(payload, dict):
        raise ValueError("Planner output JSON must be list or object.")

    return PlannerOutput.model_validate(payload)


def _to_subtasks(output: PlannerOutput) -> list[SubTask]:
    """
    _to_subtasks を実行する。

    Args:
        output: output に関する値。
    """
    return [
        SubTask(
            title=item.title,
            description=item.description,
            assigned_agent=item.assigned_agent,
            status=item.status,
        )
        for item in output.subtasks
    ]
