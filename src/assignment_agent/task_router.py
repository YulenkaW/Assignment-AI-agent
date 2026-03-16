"""Deterministic task router for the separate assignment agent."""

from __future__ import annotations

import logging
import os
import re

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover
    ChatPromptTemplate = None
    ChatOpenAI = None

from .contracts import QueryRequest, RouteDecision
from .execution_intent import ExecutionIntent


def _compile_word_patterns(*phrases: str) -> tuple[re.Pattern[str], ...]:
    """Compile one simple word or phrase regex per lexical routing hint."""
    patterns = []
    for phrase in phrases:
        parts = [re.escape(part) for part in phrase.split()]
        if len(parts) == 1:
            patterns.append(re.compile(rf"\b{parts[0]}\b"))
            continue
        patterns.append(re.compile(r"\b" + r"\s+".join(parts) + r"\b"))
    return tuple(patterns)


class TaskRouter:
    """Classify the query before planning.

    Routing is deterministic so the planner cannot silently override the task family.
    This avoids the common inconsistency where the router says "understanding" and the
    planner unexpectedly runs build commands anyway.
    understanding questions prefer retrieval first
    build/test/debug questions may execute commands first
    mixed questions may do both
    """

    ACTION_OPENERS = ("run ", "execute ", "open ", "use ", "try ")

    # These lexical hints cover the non-command path. Command-oriented routing is
    # handled separately by ExecutionIntent so this list can stay small.
    SEARCH_PATTERNS = _compile_word_patterns(
        "search",
        "find",
        "grep",
        "text",
        "directory",
        "which files",
        "what files",
    ) + (
        re.compile(r"\bcontain(?:s)?\b"),
        re.compile(r"\bmatch(?:es)?\b"),
        # Quoted literals are strong search signals for exact text lookups.
        re.compile(r"['\"][^'\"]+['\"]"),
    )
    UNDERSTANDING_PATTERNS = _compile_word_patterns(
        "class",
        "function",
        "file",
        "files",
        "where",
        "what does",
        "architecture",
        "structure",
        "symbol",
        "explain",
        "how",
        "why",
        "responsible",
    )
    EXPLANATION_PATTERN = re.compile(r"\b(explain|why|where|what|how)\b")

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        self.logger = logging.getLogger(__name__)
        self.chat_model = self._build_chat_model()

    def route(self, request: QueryRequest) -> RouteDecision:
        """Return the route contract for the user request."""
        query_text = request.query_text.lower()
        execution_intent = ExecutionIntent.from_query(request.query_text)
        execution_hits = 1 if execution_intent.requests_execution() else 0
        search_hits = self._count_hits(query_text, self.SEARCH_PATTERNS)
        understanding_hits = self._count_hits(query_text, self.UNDERSTANDING_PATTERNS)
        if execution_intent.extract_makefile_name():
            understanding_hits = max(0, understanding_hits - 1)

        if execution_hits == 0 and search_hits == 0 and self._should_try_model_fallback(query_text):
            fallback_decision = self._route_with_model_fallback(request.query_text)
            if fallback_decision is not None:
                return fallback_decision

        if search_hits > 0 and execution_hits == 0:
            return RouteDecision(
                "code_understanding",
                0.97,
                True,
                False,
                "retrieval_first",
                self._build_subgoals(request.query_text, query_mode="search", include_execution=False, include_understanding=True),
                query_mode="search",
                execution_requested=False,
            )

        if execution_hits > 0 and understanding_hits > 0:
            return RouteDecision(
                "mixed",
                0.95,
                True,
                True,
                "execution_first",
                self._build_subgoals(request.query_text, query_mode="mixed", include_execution=True, include_understanding=True),
                query_mode="mixed",
                execution_requested=True,
            )
        if execution_hits > 0:
            return RouteDecision(
                "build_test_debug",
                0.95,
                True,
                True,
                "execution_first",
                self._build_subgoals(request.query_text, query_mode="execution", include_execution=True, include_understanding=False),
                query_mode="execution",
                execution_requested=True,
            )
        return RouteDecision(
            "code_understanding",
            0.90,
            True,
            False,
            "retrieval_first",
            self._build_subgoals(request.query_text, query_mode="understanding", include_execution=False, include_understanding=True),
            query_mode="understanding",
            execution_requested=False,
        )

    def _count_hits(self, query_text: str, patterns: tuple[re.Pattern[str], ...]) -> int:
        """Count lexical term hits for one route family."""
        hit_count = 0
        for pattern in patterns:
            if pattern.search(query_text):
                hit_count += 1
        return hit_count

    def _build_subgoals(self, query_text: str, query_mode: str, include_execution: bool, include_understanding: bool) -> list[str]:
        """Create explicit subgoals for mixed-task handling."""
        subgoals = []
        if include_understanding:
            if query_mode == "search":
                subgoals.append("find matching files or text in the repository")
            else:
                subgoals.append("retrieve relevant repository evidence")
        if include_execution:
            subgoals.append("run build or test commands and capture deterministic output")
        # Add explanation only when the query explicitly asks for one.
        if self.EXPLANATION_PATTERN.search(query_text.lower()):
            subgoals.append("prepare a grounded explanation")
        return subgoals

    def _should_try_model_fallback(self, lowered_query: str) -> bool:
        """Return True when deterministic signals are weak but the query still looks action-oriented."""
        return self._can_call_model() and lowered_query.startswith(self.ACTION_OPENERS)

    def _route_with_model_fallback(self, query_text: str) -> RouteDecision | None:
        """Use the model only as a last-resort intent evaluator."""
        try:
            route_name = self._classify_with_model(query_text)
        except Exception as error:
            self.logger.warning("Falling back to deterministic routing after model intent failure: %s", error)
            return None
        if route_name == "build_test_debug":
            return RouteDecision(
                "build_test_debug",
                0.72,
                True,
                True,
                "execution_first",
                self._build_subgoals(query_text, query_mode="execution", include_execution=True, include_understanding=False),
                query_mode="execution",
                execution_requested=True,
            )
        if route_name == "mixed":
            return RouteDecision(
                "mixed",
                0.72,
                True,
                True,
                "execution_first",
                self._build_subgoals(query_text, query_mode="mixed", include_execution=True, include_understanding=True),
                query_mode="mixed",
                execution_requested=True,
            )
        if route_name == "search":
            return RouteDecision(
                "code_understanding",
                0.70,
                True,
                False,
                "retrieval_first",
                self._build_subgoals(query_text, query_mode="search", include_execution=False, include_understanding=True),
                query_mode="search",
                execution_requested=False,
            )
        return None

    def _classify_with_model(self, query_text: str) -> str:
        """Ask the model to choose one high-level route for ambiguous intent."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Classify the query into exactly one label: build_test_debug, mixed, search, or code_understanding. Respond with only the label.",
                ),
                ("human", "Query: {query}"),
            ]
        )
        chain = prompt | self.chat_model
        response = chain.invoke({"query": query_text})
        return response.content.strip()

    def _build_chat_model(self):
        """Create the OpenAI-backed chat model when configuration exists."""
        if ChatOpenAI is None:
            return None
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        return ChatOpenAI(model=self.model_name, temperature=0)

    def _can_call_model(self) -> bool:
        """Return True when model-backed fallback routing is available."""
        return self.chat_model is not None and ChatPromptTemplate is not None
