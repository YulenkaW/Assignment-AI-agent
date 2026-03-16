"""Build and test output analysis for the separate assignment agent."""

from __future__ import annotations

import logging

from .contracts import AnalysisReport, ParsedCommandOutput, RootCauseCandidate
from .error_accumulator import ErrorAccumulator


class BuildTestAnalyzer:
    """Parse execution evidence into structured root-cause analysis.
    This produces:
    first reported error
    root cause candidates
    relevant files
    line numbers
    recommended next action"""

    def __init__(self, error_accumulator: ErrorAccumulator | None = None) -> None:
        self.error_accumulator = error_accumulator
        self.logger = logging.getLogger(__name__)
        self.environment_error_markers = (
            "no space left on device",
            "there is not enough space on the disk",
            "access is denied",
            "cannot write file",
            "unable to find mspdbcore.dll",
            "permission denied",
            "no cmake_cxx_compiler could be found",
            "tell cmake where to find the compiler",
        )

    def analyze(self, parsed_outputs: list[ParsedCommandOutput]) -> AnalysisReport:
        """Create the structured analysis report for parsed output."""
        first_reported_error = ""
        root_cause_candidates = []
        relevant_files = []
        line_numbers = []
        found_environment_error = False

        for parsed_output in parsed_outputs:
            if parsed_output.missing_command:
                message = f"Required command is unavailable while running: {parsed_output.command_text}"
                if not first_reported_error:
                    first_reported_error = message
                root_cause_candidates.append(RootCauseCandidate(message, confidence=0.95))
                if self.error_accumulator is not None:
                    self.error_accumulator.add("build_test_analyzer", "Environment command is missing", parsed_output.command_text)
                continue

            for error_line in parsed_output.error_lines:
                if not first_reported_error:
                    first_reported_error = error_line
                if self._is_environment_error(error_line):
                    found_environment_error = True
                    root_cause_candidates.append(RootCauseCandidate(error_line, confidence=0.98))
                else:
                    root_cause_candidates.append(RootCauseCandidate(error_line, confidence=0.6))

            for file_path, line_number in parsed_output.file_references:
                relevant_files.append(file_path)
                line_numbers.append(line_number)
                summary_text = f"{file_path}:{line_number} referenced by {parsed_output.command_text}"
                root_cause_candidates.append(RootCauseCandidate(summary_text, file_path, line_number, 0.9))

        failing_outputs = [parsed_output for parsed_output in parsed_outputs if parsed_output.exit_code != 0]
        if not root_cause_candidates and not failing_outputs:
            return AnalysisReport("", [], relevant_files, line_numbers, "answer")

        if not root_cause_candidates:
            fallback_summary = self._summarize_unparsed_failure(parsed_outputs)
            if not first_reported_error:
                first_reported_error = fallback_summary
            root_cause_candidates.append(RootCauseCandidate(fallback_summary, confidence=1.0))

        root_cause_candidates = self._deduplicate_candidates(root_cause_candidates)
        recommended_next_action = self._recommend_next_action(parsed_outputs, root_cause_candidates, found_environment_error)
        self.logger.debug(
            "analysis parsed_outputs=%d candidates=%d next_action=%s",
            len(parsed_outputs),
            len(root_cause_candidates),
            recommended_next_action,
        )
        return AnalysisReport(first_reported_error, root_cause_candidates, relevant_files, line_numbers, recommended_next_action)

    def _deduplicate_candidates(self, candidates: list[RootCauseCandidate]) -> list[RootCauseCandidate]:
        """Collapse duplicate root-cause candidates."""
        deduplicated = []
        seen_keys = set()
        for candidate in candidates:
            key = (candidate.summary_text, candidate.file_path, candidate.line_number)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduplicated.append(candidate)
        deduplicated.sort(key=self._candidate_sort_key, reverse=True)
        return deduplicated[:6]

    def _candidate_sort_key(self, candidate: RootCauseCandidate) -> float:
        """Return the root-cause ranking key."""
        return candidate.confidence

    def _recommend_next_action(
        self,
        parsed_outputs: list[ParsedCommandOutput],
        candidates: list[RootCauseCandidate],
        found_environment_error: bool,
    ) -> str:
        """Recommend the next controller action."""
        if found_environment_error:
            return "answer_with_limited_evidence"
        for parsed_output in parsed_outputs:
            if parsed_output.missing_command:
                return "answer_with_limited_evidence"
            if parsed_output.file_references:
                return "retrieve_more_context"
            if parsed_output.exit_code != 0:
                return "answer_with_limited_evidence"
        return "answer"

    def _summarize_unparsed_failure(self, parsed_outputs: list[ParsedCommandOutput]) -> str:
        """Return a deterministic summary when a command failed without structured error lines."""
        failing_outputs = [parsed_output for parsed_output in parsed_outputs if parsed_output.exit_code != 0]
        if not failing_outputs:
            return "No build or test failure was detected."
        first_failure = failing_outputs[0]
        if first_failure.exit_code == 124:
            return f"Command timed out while running: {first_failure.command_text}"
        return f"Command failed without parsable error lines while running: {first_failure.command_text}"

    def _is_environment_error(self, error_line: str) -> bool:
        """Return True when an error line clearly points to the local environment."""
        lowered_line = error_line.lower()
        return any(marker in lowered_line for marker in self.environment_error_markers)
