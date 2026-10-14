"""``[security] confirm_tools`` forces approval for extra tools."""

from __future__ import annotations

from typing import Any

from openjarvis.core.types import ToolCall, ToolResult
from openjarvis.tools import _stubs
from openjarvis.tools._stubs import BaseTool, ToolExecutor, ToolSpec


class _WriteTool(BaseTool):
    tool_id = "file_write"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name="file_write", description="write")

    def execute(self, **params: Any) -> ToolResult:
        return ToolResult(tool_name="file_write", content="written")


def _call() -> ToolCall:
    return ToolCall(id="1", name="file_write", arguments="{}")


def test_configured_tool_requires_confirmation(monkeypatch) -> None:
    monkeypatch.setattr(
        _stubs, "_configured_confirm_tools", lambda: frozenset({"file_write"})
    )
    prompts: list[str] = []

    denied = ToolExecutor(
        [_WriteTool()],
        interactive=True,
        confirm_callback=lambda p: prompts.append(p) or False,
    ).execute(_call())
    assert not denied.success and "denied" in denied.content and prompts

    unattended = ToolExecutor([_WriteTool()]).execute(_call())
    assert not unattended.success and "requires confirmation" in unattended.content

    approved = ToolExecutor(
        [_WriteTool()], interactive=True, confirm_callback=lambda p: True
    ).execute(_call())
    assert approved.success and approved.content == "written"


def test_unlisted_tool_runs_without_confirmation(monkeypatch) -> None:
    monkeypatch.setattr(_stubs, "_configured_confirm_tools", lambda: frozenset())
    assert ToolExecutor([_WriteTool()]).execute(_call()).success


def test_ask_denies_confirmation_without_terminal(monkeypatch) -> None:
    import importlib
    import io

    # ``openjarvis.cli.ask`` the attribute is the click command; get the module.
    ask_module = importlib.import_module("openjarvis.cli.ask")
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert ask_module._terminal_confirm("Allow tool 'shell_exec'?") is False
