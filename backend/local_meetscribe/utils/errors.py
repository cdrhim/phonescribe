from __future__ import annotations


class LocalMeetScribeError(RuntimeError):
    """Base class for expected user-facing errors."""


class MissingDependencyError(LocalMeetScribeError):
    def __init__(self, package: str, install_hint: str) -> None:
        super().__init__(f"Missing optional dependency '{package}'. {install_hint}")
        self.package = package
        self.install_hint = install_hint


class ExternalToolError(LocalMeetScribeError):
    def __init__(self, tool: str, install_hint: str) -> None:
        super().__init__(f"Missing external tool '{tool}'. {install_hint}")
        self.tool = tool
        self.install_hint = install_hint


class ModelUnavailableError(LocalMeetScribeError):
    def __init__(self, model: str, hint: str) -> None:
        super().__init__(f"Model unavailable: {model}. {hint}")
        self.model = model
        self.hint = hint
