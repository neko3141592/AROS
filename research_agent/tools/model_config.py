from __future__ import annotations

import os


def get_model_name(preferred_env_var: str, default: str = "gpt-4o-mini") -> str:
    """
    ノード固有の環境変数を優先し、未設定なら共通 MODEL_NAME を使う。

    Args:
        preferred_env_var: 優先して参照する環境変数名。
        default: どちらも未設定時の既定モデル名。
    """
    return (
        os.environ.get(preferred_env_var)
        or os.environ.get("MODEL_NAME")
        or default
    )
