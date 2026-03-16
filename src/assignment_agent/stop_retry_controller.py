"""Stop and retry controller for the separate assignment agent."""

from __future__ import annotations

from .contracts import AnalysisReport, StopDecision


class StopRetryController:
    """Decide whether the controller should stop or do one more retrieval cycle."""

    def __init__(self, max_iterations: int = 2) -> None:
        self.max_iterations = max_iterations

    def decide(self, iteration_count: int, analysis_report: AnalysisReport | None, remaining_retrieval_tokens: int) -> StopDecision:
        """Return the stop or retry decision."""
        if iteration_count >= self.max_iterations:
            return StopDecision(True, "max iterations reached")
        if analysis_report is None:
            return StopDecision(True, "no analysis required")
        if analysis_report.recommended_next_action != "retrieve_more_context":
            return StopDecision(True, "enough evidence found")
        if remaining_retrieval_tokens < 300:
            return StopDecision(True, "token budget getting tight")
        return StopDecision(False, "retrieve more context from execution evidence")
