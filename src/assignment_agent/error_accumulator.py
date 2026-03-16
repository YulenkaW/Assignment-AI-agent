"""Structured error accumulation shared across the assignment agent."""

from __future__ import annotations

import logging


class ErrorRecord:
    """Store one structured runtime issue with concise context."""

    def __init__(self, component: str, summary: str, detail: str = "", severity: str = "error") -> None:
        self.component = component
        self.summary = summary
        self.detail = detail
        self.severity = severity

    def render_text(self) -> str:
        """Return a short human-readable form."""
        if self.detail:
            return f"[{self.severity}] {self.component}: {self.summary} ({self.detail})"
        return f"[{self.severity}] {self.component}: {self.summary}"


class ErrorAccumulator:
    """Collect runtime issues while also emitting standard logs."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.records = []

    def add(self, component: str, summary: str, detail: str = "", severity: str = "error") -> ErrorRecord:
        """Record and log one issue."""
        record = ErrorRecord(component, summary, detail, severity)
        self.records.append(record)
        self._log_record(record)
        return record

    def add_exception(self, component: str, summary: str, error: Exception, severity: str = "error") -> ErrorRecord:
        """Record one exception with its concrete message."""
        return self.add(component, summary, str(error), severity)

    def has_errors(self) -> bool:
        """Return True when any error-level issues were recorded."""
        for record in self.records:
            if record.severity == "error":
                return True
        return False

    def get_records(self) -> list[ErrorRecord]:
        """Return a copy of the accumulated issues."""
        return list(self.records)

    def clear(self) -> None:
        """Clear accumulated records before a new query."""
        self.records.clear()

    def _log_record(self, record: ErrorRecord) -> None:
        """Mirror the structured issue into the logging system."""
        message = record.render_text()
        if record.severity == "warning":
            self.logger.warning(message)
            return
        if record.severity == "info":
            self.logger.info(message)
            return
        self.logger.error(message)
