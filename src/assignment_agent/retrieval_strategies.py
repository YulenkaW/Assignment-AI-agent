"""Goal-driven retrieval strategies for the assignment agent."""

from __future__ import annotations

from .contracts import AnalysisReport, EvidenceRequirements, RetrievalStep, RouteDecision


class RetrievalStrategy:
    """Build evidence requirements and retrieval steps for one task kind."""

    task_kind = "mixed"

    def build_evidence_requirements(
        self,
        query_text: str,
        route_decision: RouteDecision,
        analysis_report: AnalysisReport | None,
    ) -> EvidenceRequirements:
        raise NotImplementedError

    def build_retrieval_steps(
        self,
        query_text: str,
        route_decision: RouteDecision,
        analysis_report: AnalysisReport | None,
        literal_text: str,
    ) -> list[RetrievalStep]:
        raise NotImplementedError

    def _dedupe_steps(self, steps: list[RetrievalStep]) -> list[RetrievalStep]:
        """Keep the first occurrence of each operator in the fallback chain."""
        unique_steps = []
        seen_operators = set()
        for step in steps:
            if step.operator_name in seen_operators:
                continue
            seen_operators.add(step.operator_name)
            unique_steps.append(step)
        return unique_steps


class DefinitionRetrievalStrategy(RetrievalStrategy):
    """Collect the minimal evidence needed for definition-style answers."""

    task_kind = "definition_lookup"

    def build_evidence_requirements(
        self,
        query_text: str,
        route_decision: RouteDecision,
        analysis_report: AnalysisReport | None,
    ) -> EvidenceRequirements:
        return EvidenceRequirements(
            "Need the defining file, the declaration or implementation chunk, and optionally one neighbor chunk.",
            ["defining file", "declaration or implementation chunk", "optional neighboring chunk"],
            minimum_files=1,
            supporting_chunks_per_file=1,
            neighbor_chunks=1,
        )

    def build_retrieval_steps(
        self,
        query_text: str,
        route_decision: RouteDecision,
        analysis_report: AnalysisReport | None,
        literal_text: str,
    ) -> list[RetrievalStep]:
        lowered_query = query_text.lower()
        steps = []
        if "/" in lowered_query or lowered_query.endswith((".hpp", ".h", ".cpp", ".cc", ".cxx")):
            steps.append(RetrievalStep("path", "use the explicit path-like cue first"))
        steps.append(RetrievalStep("symbol", "find the defining symbol directly"))
        steps.append(RetrievalStep("path", "fallback to identifier-oriented path lookup"))
        steps.append(RetrievalStep("keyword", "fallback to lexical retrieval if direct lookup is weak"))
        return self._dedupe_steps(steps)


class LocationRetrievalStrategy(DefinitionRetrievalStrategy):
    """Location questions use the same evidence as definition questions."""

    task_kind = "location_lookup"


class ResponsibilityRetrievalStrategy(RetrievalStrategy):
    """Collect multiple implementation files for architecture or ownership questions."""

    task_kind = "responsibility_analysis"

    def build_evidence_requirements(
        self,
        query_text: str,
        route_decision: RouteDecision,
        analysis_report: AnalysisReport | None,
    ) -> EvidenceRequirements:
        return EvidenceRequirements(
            "Need several candidate files plus one or two supporting chunks from each before synthesis.",
            ["3-5 candidate files", "supporting implementation chunks", "cross-file synthesis"],
            minimum_files=3,
            supporting_chunks_per_file=1,
            neighbor_chunks=1,
            synthesis_required=True,
        )

    def build_retrieval_steps(
        self,
        query_text: str,
        route_decision: RouteDecision,
        analysis_report: AnalysisReport | None,
        literal_text: str,
    ) -> list[RetrievalStep]:
        return [
            RetrievalStep("keyword", "start with concept-oriented retrieval"),
            RetrievalStep("symbol", "fallback to symbol-aware retrieval for named concepts"),
            RetrievalStep("path", "use file names as a late fallback"),
        ]


class LiteralSearchRetrievalStrategy(RetrievalStrategy):
    """Keep directed search narrow unless the exact pass fails."""

    task_kind = "literal_search"

    def build_evidence_requirements(
        self,
        query_text: str,
        route_decision: RouteDecision,
        analysis_report: AnalysisReport | None,
    ) -> EvidenceRequirements:
        return EvidenceRequirements(
            "Need exact literal or scoped file matches before broader expansion.",
            ["exact literal or scoped match"],
            minimum_files=1,
            supporting_chunks_per_file=1,
        )

    def build_retrieval_steps(
        self,
        query_text: str,
        route_decision: RouteDecision,
        analysis_report: AnalysisReport | None,
        literal_text: str,
    ) -> list[RetrievalStep]:
        steps = []
        if literal_text:
            steps.append(RetrievalStep("literal_text", "start with exact literal search"))
        steps.append(RetrievalStep("keyword", "use lexical fallback only if the exact pass is weak"))
        return self._dedupe_steps(steps)


class BuildFailureRetrievalStrategy(RetrievalStrategy):
    """Collect the failing file, output evidence, and nearby implementation context."""

    task_kind = "build_failure_analysis"

    def build_evidence_requirements(
        self,
        query_text: str,
        route_decision: RouteDecision,
        analysis_report: AnalysisReport | None,
    ) -> EvidenceRequirements:
        return EvidenceRequirements(
            "Need parsed output, the failing file chunk, and nearby implementation context.",
            ["parsed build output", "failing file chunk", "surrounding lines or related header"],
            minimum_files=1,
            supporting_chunks_per_file=1,
            neighbor_chunks=1,
        )

    def build_retrieval_steps(
        self,
        query_text: str,
        route_decision: RouteDecision,
        analysis_report: AnalysisReport | None,
        literal_text: str,
    ) -> list[RetrievalStep]:
        steps = []
        if analysis_report is not None and analysis_report.relevant_files:
            steps.append(RetrievalStep("execution_guided", "start from failing files reported by execution analysis"))
        steps.append(RetrievalStep("path", "use the failing path as the next strongest signal"))
        steps.append(RetrievalStep("keyword", "fallback to lexical retrieval for related implementation"))
        return self._dedupe_steps(steps)


class TestFailureRetrievalStrategy(BuildFailureRetrievalStrategy):
    """Test-failure questions extend build-failure retrieval with test-oriented evidence."""

    task_kind = "test_failure_analysis"

    def build_evidence_requirements(
        self,
        query_text: str,
        route_decision: RouteDecision,
        analysis_report: AnalysisReport | None,
    ) -> EvidenceRequirements:
        return EvidenceRequirements(
            "Need the failing test name or file, the tested implementation file, and a concise error summary.",
            ["failing test name or file", "implementation chunk", "error output summary"],
            minimum_files=2,
            supporting_chunks_per_file=1,
            neighbor_chunks=1,
        )

    def build_retrieval_steps(
        self,
        query_text: str,
        route_decision: RouteDecision,
        analysis_report: AnalysisReport | None,
        literal_text: str,
    ) -> list[RetrievalStep]:
        steps = []
        if analysis_report is not None and analysis_report.relevant_files:
            steps.append(RetrievalStep("execution_guided", "start from failing test output"))
        if literal_text:
            steps.append(RetrievalStep("literal_text", "use the failing test name or literal as a direct anchor"))
        steps.append(RetrievalStep("keyword", "expand to related tests and implementation files"))
        steps.append(RetrievalStep("path", "use path lookup as a final fallback"))
        return self._dedupe_steps(steps)


class MixedRetrievalStrategy(RetrievalStrategy):
    """Blend execution-guided and conceptual retrieval when the query mixes goals."""

    task_kind = "mixed"

    def build_evidence_requirements(
        self,
        query_text: str,
        route_decision: RouteDecision,
        analysis_report: AnalysisReport | None,
    ) -> EvidenceRequirements:
        return EvidenceRequirements(
            "Need both execution evidence and supporting repository context before answering.",
            ["execution evidence", "supporting repository chunks"],
            minimum_files=2,
            supporting_chunks_per_file=1,
            neighbor_chunks=1,
            synthesis_required=True,
        )

    def build_retrieval_steps(
        self,
        query_text: str,
        route_decision: RouteDecision,
        analysis_report: AnalysisReport | None,
        literal_text: str,
    ) -> list[RetrievalStep]:
        steps = []
        if analysis_report is not None and analysis_report.relevant_files:
            steps.append(RetrievalStep("execution_guided", "anchor mixed retrieval in execution evidence first"))
        steps.append(RetrievalStep("symbol", "collect direct code evidence for the explained symbol"))
        steps.append(RetrievalStep("keyword", "fallback to lexical expansion for broader context"))
        return self._dedupe_steps(steps)
