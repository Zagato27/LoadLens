"""Manual generator: nothing runs anywhere, start and stop only mark the window of the run."""

from __future__ import annotations

from typing import Optional

from ..tools import ToolTemplate
from .base import LOG_TAIL_BYTES, LaunchPlan, LaunchResult, Probe


class ManualExecutor:
    enforces_deadline = False
    supports_signals = True
    supports_control = False

    def launch(self, plan: LaunchPlan) -> LaunchResult:
        return LaunchResult(ref=plan.ref)

    def probe(self, ref: str) -> Probe:
        return Probe(state="running")

    def signal(self, ref: str, signal: str, grace_seconds: int = 30) -> None:
        return None

    def kill(self, ref: str) -> None:
        return None

    def logs(self, ref: str, max_bytes: int = LOG_TAIL_BYTES) -> str:
        return ""

    def cleanup(self, ref: str) -> None:
        return None

    def control(self, ref: str, argv: tuple[str, ...], cwd: str = "") -> tuple[int, str]:
        return 1, ""

    def list_scripts(self, scripts_dir: str, tool: ToolTemplate) -> Optional[list[str]]:
        return None


__all__ = ["ManualExecutor"]
