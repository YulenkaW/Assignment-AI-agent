"""Goal-driven retrieval planner for budget-aware and evidence-aware retrieval."""

from __future__ import annotations

import re

from .contracts import AnalysisReport, RetrievalBudget, RetrievalPlan, RouteDecision
from .retrieval_strategies import (
    BuildFailureRetrievalStrategy,
    DefinitionRetrievalStrategy,
    LiteralSearchRetrievalStrategy,
    LocationRetrievalStrategy,
    MixedRetrievalStrategy,
    ResponsibilityRetrievalStrategy,
    TestFailureRetrievalStrategy,
)


class RetrievalPlanner:
    """Plan retrieval around the task objective and required evidence."""

    LITERAL_PATTERN = re.compile(r"['\"]([^'\"]+)['\"]")
    UNQUOTED_LITERAL_PATTERN = re.compile(r"\bcontain(?:s)?\s+([A-Za-z0-9_./:-]{3,})\b|\bfind\s+text\s+([A-Za-z0-9_./:-]{3,})\b")
    SCOPE_PATTERNS = {
        "tests/": re.compile(r"\b(?:under|in)\s+tests\b|\btests\s+directory\b"),
        "src/": re.compile(r"\b(?:under|in)\s+src\b|\bsrc\s+directory\b"),
        "include/": re.compile(r"\b(?:under|in)\s+include\b|\binclude\s+directory\b"),
        "docs/": re.compile(r"\b(?:under|in)\s+docs\b|\bdocs\s+directory\b"),
    }
    RESPONSIBILITY_PATTERNS = (
        re.compile(r"\bwhich\s+files?\b"),
        re.compile(r"\bwhat\s+files?\b"),
        re.compile(r"\bresponsible\b"),
        re.compile(r"\barchitecture\b"),
        re.compile(r"\bstructure\b"),
        re.compile(r"\bserialization\b"),
        re.compile(r"\bserializer\b"),
        re.compile(r"\bparser\b"),
        re.compile(r"\bparsing\b"),
    )

    def __init__(self) -> None:
        self.strategies = {
            "definition_lookup": DefinitionRetrievalStrategy(),
            "location_lookup": LocationRetrievalStrategy(),
            "responsibility_analysis": ResponsibilityRetrievalStrategy(),
            "literal_search": LiteralSearchRetrievalStrategy(),
            "build_failure_analysis": BuildFailureRetrievalStrategy(),
            "test_failure_analysis": TestFailureRetrievalStrategy(),
            "mixed": MixedRetrievalStrategy(),
        }

    def create_plan(
        self,
        query_text: str,
        route_decision: RouteDecision,
        retrieval_token_capacity: int,
        analysis_report: AnalysisReport | None = None,
    ) -> RetrievalPlan:
        """Create the goal-driven retrieval plan for the current state."""
        literal_text = self._extract_literal_text(query_text, route_decision)
        scope_prefixes = self._extract_scope_prefixes(query_text)
        task_kind = self._classify_task(query_text, route_decision, analysis_report, literal_text)
        budget = self._build_budget(retrieval_token_capacity, task_kind)
        strategy = self.strategies[task_kind]
        evidence_requirements = strategy.build_evidence_requirements(query_text, route_decision, analysis_report)
        retrieval_steps = strategy.build_retrieval_steps(query_text, route_decision, analysis_report, literal_text)
        preferred_files = analysis_report.relevant_files if analysis_report is not None else None
        preferred_lines = analysis_report.line_numbers if analysis_report is not None else None
        return RetrievalPlan(
            task_kind,
            query_text,
            budget,
            evidence_requirements,
            retrieval_steps,
            preferred_files=preferred_files,
            preferred_lines=preferred_lines,
            literal_text=literal_text,
            scope_prefixes=scope_prefixes,
        )

    def _classify_task(
        self,
        query_text: str,
        route_decision: RouteDecision,
        analysis_report: AnalysisReport | None,
        literal_text: str,
    ) -> str:
        """Classify retrieval by the answer objective, not by backend operator."""
        lowered_query = query_text.lower()
        if analysis_report is not None and route_decision.needs_execution:
            if "test" in lowered_query or any("test" in file_path.lower() for file_path in analysis_report.relevant_files):
                return "test_failure_analysis"
            return "build_failure_analysis"
        if route_decision.query_mode == "search":
            return "literal_search"
        if route_decision.task_type == "mixed":
            return "mixed"
        if self._is_responsibility_query(lowered_query):
            return "responsibility_analysis"
        if "what does" in lowered_query or "what is" in lowered_query:
            return "definition_lookup"
        if self._is_location_query(lowered_query):
            return "location_lookup"
        if route_decision.needs_execution:
            return "build_failure_analysis"
        if literal_text and route_decision.query_mode == "search":
            return "literal_search"
        return "definition_lookup"

    def _build_budget(self, retrieval_token_capacity: int, task_kind: str) -> RetrievalBudget:
        """Build a retrieval-budget policy object for one task kind."""
        max_retrieval_tokens = max(300, retrieval_token_capacity)
        if max_retrieval_tokens <= 900:
            result_limit = 2
        elif max_retrieval_tokens <= 1500:
            result_limit = 3
        elif task_kind in {"responsibility_analysis", "mixed"}:
            result_limit = 4
        else:
            result_limit = 3

        if task_kind == "responsibility_analysis":
            return RetrievalBudget(max_retrieval_tokens, max(result_limit, 4), 1, neighbor_limit=1)
        if task_kind in {"build_failure_analysis", "test_failure_analysis", "mixed"}:
            return RetrievalBudget(max_retrieval_tokens, max(result_limit, 3), 1, neighbor_limit=1)
        return RetrievalBudget(max_retrieval_tokens, result_limit, 1, neighbor_limit=1)

    def _extract_literal_text(self, query_text: str, route_decision: RouteDecision) -> str:
        """Return the first quoted literal from the query when present."""
        match = self.LITERAL_PATTERN.search(query_text)
        if match is not None:
            return match.group(1).strip()
        if route_decision.query_mode != "search":
            return ""
        unquoted_match = self.UNQUOTED_LITERAL_PATTERN.search(query_text)
        if unquoted_match is None:
            return ""
        literal_text = unquoted_match.group(1) or unquoted_match.group(2) or ""
        return literal_text.strip()

    def _extract_scope_prefixes(self, query_text: str) -> list[str]:
        """Extract directory scopes such as `tests/` from the query."""
        lowered_query = query_text.lower()
        prefixes = []
        for scope_prefix, pattern in self.SCOPE_PATTERNS.items():
            if pattern.search(lowered_query):
                prefixes.append(scope_prefix)
        return prefixes

    def _is_responsibility_query(self, lowered_query: str) -> bool:
        """Return True for architecture or ownership questions."""
        for pattern in self.RESPONSIBILITY_PATTERNS:
            if pattern.search(lowered_query):
                return True
        return False

    def _is_location_query(self, lowered_query: str) -> bool:
        """Return True when the user is primarily asking where something is defined."""
        return "where is" in lowered_query or "where are" in lowered_query or "defined" in lowered_query
