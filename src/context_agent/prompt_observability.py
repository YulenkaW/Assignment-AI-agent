"""Prompt assembly observability helpers.

This file records what entered the final prompt, what was truncated, and what was
removed. It exists to make prompt behavior explainable during debugging and demos.
"""

from __future__ import annotations


class PromptSectionDecision:
    """Stores one prompt assembly decision for a section."""

    def __init__(self, label: str, status: str, token_count: int, priority: int, detail_text: str = "") -> None:
        self.label = label
        self.status = status
        self.token_count = token_count
        self.priority = priority
        self.detail_text = detail_text


class PromptAssemblyReport:
    """Stores the full observability report for one assembled prompt."""

    def __init__(self) -> None:
        self.decisions = []
        self.total_input_tokens = 0

    def add_decision(self, label: str, status: str, token_count: int, priority: int, detail_text: str = "") -> None:
        """Store one section decision."""
        self.decisions.append(PromptSectionDecision(label, status, token_count, priority, detail_text))

    def set_total_input_tokens(self, token_count: int) -> None:
        """Store the total prompt token count for the kept sections."""
        self.total_input_tokens = token_count

    def render_text(self) -> str:
        """Render the report as plain text for debugging and demos."""
        lines = [f"total_input_tokens: {self.total_input_tokens}"]
        for decision in self.decisions:
            detail_suffix = f" ({decision.detail_text})" if decision.detail_text else ""
            lines.append(
                f"label={decision.label} status={decision.status} tokens={decision.token_count} priority={decision.priority}{detail_suffix}"
            )
        return "\n".join(lines)
