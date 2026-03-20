from __future__ import annotations

import time
from typing import Any

from litellm import completion


class LLMClientError(Exception):
    """LLM呼び出し失敗時に投げる統一例外。"""


def _extract_text_from_response(response: Any) -> str:
    """
    LiteLLMレスポンスから最初のテキストを安全に取り出す。
    """
    try:
        content = response["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        raise LLMClientError(
            "Invalid response format from LLM provider."
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise LLMClientError("LLM returned empty content.")

    return content

def _extract_tool_response(response: Any) -> dict[str, Any]:
    """
    LiteLLMレスポンスから content / tool_calls を取り出す。
    content は tool call 時に空のことがあるため許容する。
    """
    try:
        message = response["choices"][0]["message"]
    except Exception as exc:  # noqa: BLE001
        raise LLMClientError("Invalid response format from LLM provider.") from exc

    content = message.get("content")
    if content is None:
        content_text = ""
    elif isinstance(content, str):
        content_text = content
    else:
        raise LLMClientError("LLM returned non-string content.")

    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        raise LLMClientError("LLM returned invalid tool_calls format.")

    if not content_text.strip() and not tool_calls:
        raise LLMClientError("LLM returned neither content nor tool_calls.")

    return {
        "content": content_text,
        "tool_calls": tool_calls,
    }


def generate_with_tools(
    system_prompt: str,
    user_prompt: str,
    model: str,
    tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any] = "auto",
    temperature: float = 1.0,
    timeout: float = 30.0,
    max_retries: int = 2,
    base_backoff_seconds: float = 1.0,
) -> dict:
    """
    Tool Calling付きでLLMに問い合わせ、content/tool_callsを返す。
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0.")
    if timeout <= 0:
        raise ValueError("timeout must be > 0.")
    if not isinstance(tools, list) or not tools:
        raise ValueError("tools must be a non-empty list.")

    attempt = 0
    last_error: Exception | None = None

    while attempt <= max_retries:
        try:
            response = completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                timeout=timeout,
            )
            return _extract_tool_response(response)
        except LLMClientError:
            raise
        # 指数バックオフ
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries:
                break
            backoff = base_backoff_seconds * (2**attempt)
            time.sleep(backoff)
            attempt += 1

    raise LLMClientError(
        f"LLM tool request failed after {max_retries + 1} attempt(s): {last_error}"
    ) from last_error


def generate_text(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 1.0,
    timeout: float = 30.0,
    max_retries: int = 2,
    base_backoff_seconds: float = 1.0,
) -> str:
    """
    LLMへテキスト生成を依頼し、失敗時はリトライ付きで結果を返す。
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0.")

    if timeout <= 0:
        raise ValueError("timeout must be > 0.")

    attempt = 0
    last_error: Exception | None = None

    while attempt <= max_retries:
        try:
            response = completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                timeout=timeout,
            )
            return _extract_text_from_response(response)
        except LLMClientError:
            # フォーマット不正は再試行しても改善しづらいため即時失敗
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries:
                break
            backoff = base_backoff_seconds * (2**attempt)
            time.sleep(backoff)
            attempt += 1

    raise LLMClientError(
        f"LLM request failed after {max_retries + 1} attempt(s): {last_error}"
    ) from last_error
