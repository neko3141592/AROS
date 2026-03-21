from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any, Mapping

import yaml
from langchain_core.messages import SystemMessage


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


class PromptManagerError(Exception):
    """プロンプト管理で発生する例外の基底クラス。"""


class PromptLoadError(PromptManagerError):
    """プロンプト読み込み時のエラー。"""


class PromptValidationError(PromptManagerError):
    """プロンプト定義の不正を示すエラー。"""


class PromptRenderError(PromptManagerError):
    """プロンプト適用（レンダリング）時のエラー。"""


@dataclass(frozen=True)
class PromptDefinition:
    """YAML から読み込んだプロンプト定義。"""

    name: str
    description: str
    input_variables: list[str]
    template: str


def _resolve_prompt_path(prompt_name: str, prompts_dir: Path) -> Path:
    """
    _resolve_prompt_path を実行する。
    
    Args:
        prompt_name: 利用するプロンプト名。
        prompts_dir: プロンプトファイルが格納されたディレクトリ。
    """
    file_name = prompt_name if prompt_name.endswith(".yaml") else f"{prompt_name}.yaml"
    prompt_path = prompts_dir / file_name
    if not prompt_path.exists():
        raise PromptLoadError(f"Prompt file not found: {prompt_path}")
    return prompt_path


def _extract_placeholders(template: str) -> set[str]:
    """
    _extract_placeholders を実行する。
    
    Args:
        template: プレースホルダを含むテンプレート文字列。
    """
    return {
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name
    }


def _format_variable(value: Any) -> str:
    """
    _format_variable を実行する。
    
    Args:
        value: 変換対象の値。
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        if not value:
            return "- (none)"
        return "\n".join(f"- {item}" for item in value)
    return str(value)


def load_prompt(prompt_name: str, prompts_dir: Path | None = None) -> PromptDefinition:
    """
    YAML 形式のプロンプト定義を読み込んで返す。
    
    Args:
        prompt_name: 利用するプロンプト名。
        prompts_dir: プロンプトファイルが格納されたディレクトリ。
    """
    resolved_dir = prompts_dir or PROMPTS_DIR
    prompt_path = _resolve_prompt_path(prompt_name, resolved_dir)

    with prompt_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise PromptValidationError(
            f"Prompt YAML must be a mapping object: {prompt_path}"
        )

    required_keys = {"name", "description", "input_variables", "template"}
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        missing_str = ", ".join(sorted(missing_keys))
        raise PromptValidationError(
            f"Prompt YAML is missing required keys ({missing_str}): {prompt_path}"
        )

    input_variables = data["input_variables"]
    if (
        not isinstance(input_variables, list)
        or not input_variables
        or any(not isinstance(v, str) or not v.strip() for v in input_variables)
    ):
        raise PromptValidationError(
            f"'input_variables' must be a non-empty list[str]: {prompt_path}"
        )

    template = data["template"]
    if not isinstance(template, str) or not template.strip():
        raise PromptValidationError(
            f"'template' must be a non-empty string: {prompt_path}"
        )

    placeholders = _extract_placeholders(template)
    undefined = placeholders - set(input_variables)
    if undefined:
        undefined_str = ", ".join(sorted(undefined))
        raise PromptValidationError(
            f"Template placeholders are not declared in input_variables ({undefined_str}): {prompt_path}"
        )

    return PromptDefinition(
        name=str(data["name"]),
        description=str(data["description"]),
        input_variables=input_variables,
        template=template,
    )


def apply_prompt(
    prompt_definition: PromptDefinition, variables: Mapping[str, Any]
) -> str:
    """
    読み込んだプロンプト定義に変数を適用して、最終的な文字列を返す。
    
    Args:
        prompt_definition: 読み込んだプロンプト定義。
        variables: テンプレート埋め込みに使う変数辞書。
    """
    missing = [
        variable
        for variable in prompt_definition.input_variables
        if variable not in variables
    ]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise PromptRenderError(
            f"Missing variables for prompt '{prompt_definition.name}': {missing_str}"
        )

    render_values = {
        variable: _format_variable(variables[variable])
        for variable in prompt_definition.input_variables
    }

    try:
        return prompt_definition.template.format(**render_values)
    except KeyError as exc:
        missing_key = str(exc).strip("'")
        raise PromptRenderError(
            f"Failed to render prompt '{prompt_definition.name}'. Missing key in template: {missing_key}"
        ) from exc


def render_prompt(
    prompt_name: str, variables: Mapping[str, Any], prompts_dir: Path | None = None
) -> str:
    """
    プロンプトを読み込み、変数適用までをまとめて実行する。
    
    Args:
        prompt_name: 利用するプロンプト名。
        variables: テンプレート埋め込みに使う変数辞書。
        prompts_dir: プロンプトファイルが格納されたディレクトリ。
    """
    prompt_definition = load_prompt(prompt_name=prompt_name, prompts_dir=prompts_dir)
    return apply_prompt(prompt_definition=prompt_definition, variables=variables)


def build_system_message(
    prompt_name: str, variables: Mapping[str, Any], prompts_dir: Path | None = None
) -> SystemMessage:
    """
    プロンプトを SystemMessage へ変換する。
    
    Args:
        prompt_name: 利用するプロンプト名。
        variables: テンプレート埋め込みに使う変数辞書。
        prompts_dir: プロンプトファイルが格納されたディレクトリ。
    """
    prompt_text = render_prompt(
        prompt_name=prompt_name,
        variables=variables,
        prompts_dir=prompts_dir,
    )
    return SystemMessage(content=prompt_text)

