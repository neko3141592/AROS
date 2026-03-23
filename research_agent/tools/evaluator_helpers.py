from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any

from schema.llm_outputs import EvaluatorAnalysisOutput

LLM_STDOUT_MAX_CHARS = 2500
LLM_STDERR_MAX_CHARS = 4500
LLM_LOG_MAX_LINE_CHARS = 400
LLM_STDERR_CONTEXT_LINES = 2
LLM_LOG_HEAD_LINES = 20
LLM_LOG_TAIL_LINES = 20

_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password|passwd)\b(\s*[:=]\s*)(['\"]?)([^'\"\s]+)\3"
)
_BEARER_TOKEN_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")
_GENERIC_LONG_TOKEN_RE = re.compile(
    r"\b(?=[A-Za-z0-9_\-]{24,}\b)(?=[A-Za-z0-9_\-]*[A-Za-z])(?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9_\-]+\b"
)
_DSN_CREDENTIAL_RE = re.compile(
    r"\b([a-z][a-z0-9+.\-]*://)([^/\s:@]+):([^@\s]+)@",
    re.IGNORECASE,
)
_LONG_URL_RE = re.compile(r"https?://[^\s\"')\]]{40,}", re.IGNORECASE)
_HOME_PATH_RE = re.compile(r"(?:/Users|/home)/[^/\s:]+(?:/[^\s:]+)*")
_NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_EXCEPTION_LINE_RE = re.compile(r"^\s*[A-Za-z_][\w.]*?(?:Error|Exception|Warning)\s*:")


@dataclass
class FailureClassification:
    summary: str
    likely_cause: str
    suggested_fixes: list[str]
    can_self_fix: bool
    needs_research: bool


def _read_positive_float_env(name: str, default: float) -> float:
    """
    正の浮動小数点設定値を環境変数から読み取る。

    Args:
        name: 環境変数名。
        default: 未設定時の既定値。
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default

    value = float(raw)
    if value <= 0:
        raise ValueError(f"{name} must be > 0.")
    return value


def calculate_next_timeout(
    per_try_timeout: float,
    total_max_timeout: float,
    current_total_used: float,
) -> float:
    remaining_budget = max(0, total_max_timeout - current_total_used)
    next_timeout = min(per_try_timeout, remaining_budget)
    return next_timeout


def _extract_exception_line(stderr: str) -> str:
    """
    Tracebackの最終例外行を返す。見つからない場合は空白を返す
    """
    for line in reversed(stderr.splitlines()):
        line = line.strip()
        if line and not line.startswith(" ") and ":" in line:
            return line
    return ""


def _classify_failure(stderr: str, returncode: int) -> FailureClassification:
    """
    stderrとreturncodeから失敗を分類し、FailureClassificationを返す。
    将来的にruntime_error_unknown の発生率が高くなったら、LLMエスカレーションを検討。
    """
    exception_line = _extract_exception_line(stderr)

    if returncode == 124 or "timeout" in stderr.lower():
        return FailureClassification(
            summary="timeout",
            likely_cause="Execution exceeded the time limit.",
            suggested_fixes=[
                "Check for infinite loops",
                "Reduce workload",
                "Increase timeout threshold",
            ],
            can_self_fix=True,
            needs_research=False,
        )

    def starts(prefix: str) -> bool:
        return exception_line.startswith(prefix)

    if starts("ModuleNotFoundError"):
        module = exception_line.split("'")[1] if "'" in exception_line else "unknown"
        return FailureClassification(
            summary="missing_module",
            likely_cause=f"Module '{module}' is not installed.",
            suggested_fixes=[
                f"pip install {module}",
                "Check virtual environment",
                "Check for typos in module name",
            ],
            can_self_fix=False,
            needs_research=True,
        )

    if starts("SyntaxError") or starts("IndentationError"):
        return FailureClassification(
            summary="syntax_error",
            likely_cause="Invalid syntax or indentation.",
            suggested_fixes=[
                "Unify indentation (4 spaces recommended)",
                "Check matching brackets and quotes",
            ],
            can_self_fix=True,
            needs_research=False,
        )

    if starts("NameError") or starts("AttributeError") or starts("TypeError"):
        return FailureClassification(
            summary="name_or_type_error",
            likely_cause="Mismatched variable name, attribute, or type.",
            suggested_fixes=[
                "Check for typos in variable names",
                "Verify types (e.g. str vs int)",
                "Check object is not None",
            ],
            can_self_fix=True,
            needs_research=False,
        )

    return FailureClassification(
        summary="runtime_error_unknown",
        likely_cause=f"Unclassified runtime error: {exception_line or 'no details'}",
        suggested_fixes=["Inspect stderr stack trace", "Reproduce and isolate the issue"],
        can_self_fix=False,
        needs_research=True,
    )


def _build_error_signature(stderr: str, returncode: int) -> str:
    """
    例外種別・代表メッセージ・returncodeを正規化してSHA256ハッシュを生成する。
    同一エラーの反復検知に使用。可変値（パス・行番号）は除去して揺れを吸収する。
    """
    exception_line = _extract_exception_line(stderr)

    normalized = exception_line
    normalized = re.sub(r"[\w/\\.-]+\.py", "<file>", normalized)
    normalized = re.sub(r"line \d+", "line <N>", normalized)
    normalized = re.sub(r"\b\d+\b", "<N>", normalized)
    normalized = re.sub(r"'[^']*'", "'<V>'", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    raw = f"{normalized}|rc={returncode}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _truncate_text_middle(text: str, max_chars: int) -> str:
    """
    文字列が長すぎる場合、中央を省略して最大文字数へ収める。
    """
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = "\n... <truncated> ...\n"
    if max_chars <= len(marker):
        return text[:max_chars]
    head_len = (max_chars - len(marker)) // 2
    tail_len = max_chars - len(marker) - head_len
    return f"{text[:head_len]}{marker}{text[-tail_len:]}"


def _truncate_long_lines(text: str, max_line_chars: int) -> str:
    """
    1行ごとの長さを制限し、巨大行で文脈が崩れるのを防ぐ。
    """
    if not text:
        return ""
    if max_line_chars <= 0:
        return ""

    truncated_lines: list[str] = []
    for line in text.splitlines():
        if len(line) <= max_line_chars:
            truncated_lines.append(line)
            continue
        truncated_lines.append(f"{line[:max_line_chars]} ... <line truncated>")
    return "\n".join(truncated_lines)


def _mask_sensitive_log_text(text: str) -> str:
    """
    ログ中の機密情報っぽい値をマスキングする。
    """
    if not text:
        return ""

    masked = _NON_PRINTABLE_RE.sub("?", text)

    def _mask_assignment(match: re.Match[str]) -> str:
        key = match.group(1)
        sep = match.group(2)
        quote = match.group(3) or ""
        return f"{key}{sep}{quote}<REDACTED>{quote}"

    masked = _SECRET_ASSIGNMENT_RE.sub(_mask_assignment, masked)
    masked = _BEARER_TOKEN_RE.sub("Bearer <REDACTED_TOKEN>", masked)
    masked = _OPENAI_KEY_RE.sub("<REDACTED_API_KEY>", masked)
    masked = _GENERIC_LONG_TOKEN_RE.sub("<REDACTED_TOKEN>", masked)
    masked = _DSN_CREDENTIAL_RE.sub(r"\1<REDACTED_USER>:<REDACTED_PASS>@", masked)
    masked = _LONG_URL_RE.sub("<REDACTED_URL>", masked)
    masked = _HOME_PATH_RE.sub("<HOME_PATH>", masked)
    return masked


def _summarize_head_tail(
    text: str,
    max_chars: int,
    head_lines: int = LLM_LOG_HEAD_LINES,
    tail_lines: int = LLM_LOG_TAIL_LINES,
) -> str:
    """
    長文ログを先頭/末尾中心に要約する。
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    lines = text.splitlines()
    if len(lines) <= head_lines + tail_lines + 1:
        return _truncate_text_middle(text, max_chars)

    omitted = len(lines) - head_lines - tail_lines
    excerpt = "\n".join(
        [
            *lines[:head_lines],
            f"... <{omitted} lines omitted> ...",
            *lines[-tail_lines:],
        ]
    )
    return _truncate_text_middle(excerpt, max_chars)


def _build_line_ranges_from_indexes(
    indexes: list[int],
    total_lines: int,
    context_lines: int,
) -> list[tuple[int, int]]:
    """
    注目行インデックスから前後文脈を含む行レンジへ変換する。
    """
    if not indexes or total_lines <= 0:
        return []

    ranges: list[tuple[int, int]] = []
    for idx in sorted(set(indexes)):
        start = max(0, idx - context_lines)
        end = min(total_lines - 1, idx + context_lines)
        if ranges and start <= ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))
    return ranges


def _render_ranges_with_omission_markers(
    lines: list[str],
    ranges: list[tuple[int, int]],
) -> str:
    """
    行レンジを省略マーカー付きで連結する。
    """
    if not lines:
        return ""
    if not ranges:
        return "\n".join(lines)

    rendered: list[str] = []
    cursor = 0
    for start, end in ranges:
        if start > cursor:
            rendered.append(f"... <{start - cursor} lines omitted> ...")
        rendered.extend(lines[start : end + 1])
        cursor = end + 1
    if cursor < len(lines):
        rendered.append(f"... <{len(lines) - cursor} lines omitted> ...")
    return "\n".join(rendered)


def _summarize_stderr_for_llm(
    stderr: str,
    max_chars: int,
    context_lines: int = LLM_STDERR_CONTEXT_LINES,
) -> str:
    """
    stderr は例外行とその前後を優先して抽出する。
    """
    if not stderr:
        return ""

    lines = stderr.splitlines()
    if not lines:
        return stderr
    if len(stderr) <= max_chars:
        return stderr

    focus_indexes: list[int] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Traceback (most recent call last):"):
            focus_indexes.append(i)
            continue
        if _EXCEPTION_LINE_RE.match(stripped):
            focus_indexes.append(i)

    if not focus_indexes:
        for i in range(len(lines) - 1, -1, -1):
            stripped = lines[i].strip()
            if stripped and ":" in stripped:
                focus_indexes.append(i)
                break

    if not focus_indexes:
        return _summarize_head_tail(stderr, max_chars)

    ranges = _build_line_ranges_from_indexes(
        indexes=focus_indexes,
        total_lines=len(lines),
        context_lines=context_lines,
    )
    excerpt = _render_ranges_with_omission_markers(lines, ranges)
    return _truncate_text_middle(excerpt, max_chars)


def _sanitize_execution_outputs_for_llm(
    stdout: Any,
    stderr: Any,
) -> tuple[str, str]:
    """
    LLM 送信用の stdout/stderr をマスク・要約して返す。
    """
    stdout_text = "" if stdout is None else str(stdout)
    stderr_text = "" if stderr is None else str(stderr)

    masked_stdout = _mask_sensitive_log_text(stdout_text)
    masked_stderr = _mask_sensitive_log_text(stderr_text)

    normalized_stdout = _truncate_long_lines(masked_stdout, LLM_LOG_MAX_LINE_CHARS)
    normalized_stderr = _truncate_long_lines(masked_stderr, LLM_LOG_MAX_LINE_CHARS)

    compact_stdout = _summarize_head_tail(normalized_stdout, LLM_STDOUT_MAX_CHARS)
    compact_stderr = _summarize_stderr_for_llm(normalized_stderr, LLM_STDERR_MAX_CHARS)

    return compact_stdout or "(empty)", compact_stderr or "(empty)"


def _strip_code_fence(text: str) -> str:
    """
    LLM が返すコードフェンスを除去する。

    Args:
        text: 処理対象テキスト。
    """
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    lines = cleaned.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    return cleaned


def _extract_first_json_payload(text: str) -> Any:
    """
    文字列中の最初の JSON payload を取り出す。

    Args:
        text: 処理対象テキスト。
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


def _parse_evaluator_analysis_output(raw_text: str) -> EvaluatorAnalysisOutput:
    """
    Evaluator 用 LLM 解析出力をパースする。

    Args:
        raw_text: LLM の生テキスト。
    """
    cleaned = _strip_code_fence(raw_text)
    payload = _extract_first_json_payload(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("Evaluator analysis output must be a JSON object.")
    return EvaluatorAnalysisOutput.model_validate(payload)


def _should_use_evaluator_llm_analysis() -> bool:
    """
    Evaluator の LLM 解析を有効化すべきか判定する。
    """
    flag = (os.getenv("ENABLE_EVALUATOR_LLM_ANALYSIS") or "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    return bool(os.getenv("OPENAI_API_KEY"))
