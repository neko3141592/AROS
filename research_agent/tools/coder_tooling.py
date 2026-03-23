from __future__ import annotations

from typing import Any, Callable, Mapping

from tools.workspace_tools import (
    create_file as workspace_create_file,
    edit_file as workspace_edit_file,
    list_files as workspace_list_files,
    read_file as workspace_read_file,
    replace_string as workspace_replace_string,
    run_shell_command as workspace_run_shell_command,
)

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files under workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_dir": {"type": "string"},
                    "recursive": {"type": "boolean"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file under workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create or overwrite a file under workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                    "overwrite": {"type": "boolean"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace a line range in a file under workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "new_text": {"type": "string"},
                },
                "required": ["file_path", "start_line", "end_line", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": (
                "Run an allowlisted read-only shell command inside workspace for exploration only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_sec": {"type": "number"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_string",
            "description": "Replace string in a file under workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["file_path", "old", "new"],
            },
        },
    },
]


ToolFunction = Callable[..., Any]
TOOL_FUNCTIONS: Mapping[str, ToolFunction] = {
    "list_files": workspace_list_files,
    "read_file": workspace_read_file,
    "create_file": workspace_create_file,
    "edit_file": workspace_edit_file,
    "replace_string": workspace_replace_string,
    "run_shell_command": workspace_run_shell_command,
}
