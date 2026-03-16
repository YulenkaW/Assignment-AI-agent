"""Conversation and retrieval memory stores.

The assignment asks for different memory behavior depending on task type. This file
keeps task-scoped conversation and summary memory explicit. Shared grounded entity
facts live in `entity_fact_store.py` so they remain easy to audit and explain.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

try:
    from langchain.memory import ConversationBufferMemory
except ImportError:  # pragma: no cover
    ConversationBufferMemory = None

from .context_budget_manager import TokenBudgetManager


class ConversationTurn:
    """Represents one stored conversational turn."""

    def __init__(self, role: str, content: str, token_count: int) -> None:
        self.role = role
        self.content = content
        self.token_count = token_count


class RollingConversationMemory:
    """Stores recent conversation history under a token budget."""

    def __init__(self, budget_manager: TokenBudgetManager, max_tokens: int = 1200) -> None:
        self.budget_manager = budget_manager
        self.max_tokens = max_tokens
        self.turns = deque()
        self.total_tokens = 0
        self.langchain_memory = self._create_langchain_memory()

    def _create_langchain_memory(self):
        """Create a LangChain memory helper when the dependency is installed."""
        if ConversationBufferMemory is None:
            return None
        return ConversationBufferMemory(return_messages=False)

    def add_turn(self, role: str, content: str) -> None:
        """Store a new turn and evict old turns when the buffer is full."""
        token_count = self.budget_manager.count_text_tokens(content)
        turn = ConversationTurn(role, content, token_count)
        self.turns.append(turn)
        self.total_tokens += token_count
        if self.langchain_memory is not None:
            if role == "user":
                self.langchain_memory.chat_memory.add_user_message(content)
            else:
                self.langchain_memory.chat_memory.add_ai_message(content)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Drop the oldest turns until the rolling memory fits its budget."""
        while self.total_tokens > self.max_tokens and self.turns:
            removed_turn = self.turns.popleft()
            self.total_tokens -= removed_turn.token_count

    def render(self) -> str:
        """Render the recent turns as plain text for prompt assembly."""
        lines = []
        for turn in self.turns:
            lines.append(f"{turn.role}: {turn.content}")
        return "\n".join(lines)


class TaskConversationMemoryRouter:
    """Keeps separate rolling conversation buffers for each task family."""

    UNDERSTANDING_TASK = "understanding"
    BUILD_EXECUTION_TASK = "build_execution"

    def __init__(self, budget_manager: TokenBudgetManager, understanding_max_tokens: int = 700, build_max_tokens: int = 700) -> None:
        self.understanding_memory = RollingConversationMemory(budget_manager, max_tokens=understanding_max_tokens)
        self.build_execution_memory = RollingConversationMemory(budget_manager, max_tokens=build_max_tokens)

    def add_turn(self, task_type: str, role: str, content: str) -> None:
        """Store a turn in the memory buffer dedicated to the active task type."""
        memory = self.get_memory_for_task(task_type)
        memory.add_turn(role, content)

    def render_for_task(self, task_type: str) -> str:
        """Render the conversation buffer that belongs to the active task type."""
        memory = self.get_memory_for_task(task_type)
        return memory.render()

    def get_memory_for_task(self, task_type: str) -> RollingConversationMemory:
        """Return the correct rolling memory object for the requested task."""
        if task_type == self.BUILD_EXECUTION_TASK:
            return self.build_execution_memory
        return self.understanding_memory


class SummaryMemoryStore:
    """Stores stable summaries for files and directories."""

    def __init__(self) -> None:
        self.summaries = {}

    def remember_summary(self, key: str, summary_text: str) -> None:
        """Save or replace a summary record."""
        self.summaries[key] = summary_text

    def find_relevant_summaries(self, query_terms: Iterable[str], limit: int = 8) -> list[tuple[str, str]]:
        """Return the summaries that lexically match the current query."""
        lowered_terms = []
        for term in query_terms:
            if term:
                lowered_terms.append(term.lower())
        matches = []
        for key, summary_text in self.summaries.items():
            haystack = f"{key} {summary_text}".lower()
            score = 0
            for term in lowered_terms:
                if term in haystack:
                    score += 1
            if score > 0:
                matches.append((score, key, summary_text))
        matches.sort(reverse=True)
        trimmed_matches = matches[:limit]
        return [(key, summary_text) for _, key, summary_text in trimmed_matches]


class BuildExecutionMemory:
    """Stores a short chain-of-events memory for build and test runs."""

    def __init__(self, max_steps: int = 8) -> None:
        self.max_steps = max_steps
        self.steps = deque()

    def add_step(self, command_text: str, outcome_text: str) -> None:
        """Record one build step and keep only the most recent ones."""
        entry_text = f"command: {command_text}\noutcome: {outcome_text}"
        self.steps.append(entry_text)
        while len(self.steps) > self.max_steps:
            self.steps.popleft()

    def render(self) -> str:
        """Render the recent build chain as plain text."""
        return "\n\n".join(self.steps)
