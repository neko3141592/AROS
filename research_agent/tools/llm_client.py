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
