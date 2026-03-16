"""Final response formatting for the separate assignment agent."""

from __future__ import annotations

from .contracts import AnalysisReport, ReasoningOutcome, RouteDecision


class ResponseGenerator:
    """Format the grounded reasoning outcome for the user."""

    def generate(self, route_decision: RouteDecision, reasoning_outcome: ReasoningOutcome, analysis_report: AnalysisReport | None) -> str:
        """Return the final answer text."""
        if route_decision.task_type == "code_understanding":
            return self._build_understanding_response(reasoning_outcome)
        return self._build_debug_response(reasoning_outcome, analysis_report)

    def _build_understanding_response(self, reasoning_outcome: ReasoningOutcome) -> str:
        """Return the understanding response."""
        lines = [reasoning_outcome.summary_text]
        if reasoning_outcome.evidence_lines:
            lines.append("Evidence:")
            for evidence_line in reasoning_outcome.evidence_lines:
                lines.append(f"- {evidence_line}")
        if reasoning_outcome.next_steps:
            lines.append("Next steps:")
            for next_step in reasoning_outcome.next_steps:
                lines.append(f"- {next_step}")
        return "\n".join(lines)

    def _build_debug_response(self, reasoning_outcome: ReasoningOutcome, analysis_report: AnalysisReport | None) -> str:
        """Return the build or test debugging response."""
        lines = [reasoning_outcome.summary_text]
        if analysis_report is not None and analysis_report.first_reported_error:
            lines.append(f"First reported error: {analysis_report.first_reported_error}")
        if reasoning_outcome.evidence_lines:
            lines.append("Evidence:")
            for evidence_line in reasoning_outcome.evidence_lines:
                lines.append(f"- {evidence_line}")
        if reasoning_outcome.next_steps:
            lines.append("Next steps:")
            for next_step in reasoning_outcome.next_steps:
                lines.append(f"- {next_step}")
        return "\n".join(lines)
