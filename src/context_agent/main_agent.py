"""Main agent orchestration.

This file keeps the main control flow explicit and readable. The agent routes the
request into one of two happy-path workflows:

- codebase understanding
- build and execution
"""

from __future__ import annotations

from pathlib import Path
import os
import re

from .agent_models import AgentAnswer, PromptSection, RetrievedChunkMatch
from .build_command_executor import BuildCommandExecutor
from .build_failure_analyzer import BuildFailureAnalyzer
from .codebase_indexer import CodebaseIndexer
from .context_budget_manager import TokenBudgetManager
from .conversation_memory_store import BuildExecutionMemory, SummaryMemoryStore, TaskConversationMemoryRouter
from .entity_fact_store import EntityFactStore
from .llm_gateway import DeterministicDemoResponder, LlmGateway
from .repository_retrieval_engine import RepositoryRetrievalEngine


class ContextAwareCodebaseAgent:
    """Coordinates indexing, retrieval, command execution, memory, and LLM calls."""

    TASK_UNDERSTANDING = "understanding"
    TASK_BUILD_EXECUTION = "build_execution"

    SECTION_TOKEN_CAPS = {
        TASK_UNDERSTANDING: {
            "task": 120,
            "query": 300,
            "conversation": 500,
            "summaries": 700,
            "retrieval": 1800,
            "command_output": 0,
            "build_chain": 0,
        },
        TASK_BUILD_EXECUTION: {
            "task": 120,
            "query": 300,
            "conversation": 450,
            "summaries": 400,
            "retrieval": 1300,
            "command_output": 1100,
            "build_chain": 500,
            "build_analysis": 320,
        },
    }

    def __init__(self, repository_path: str | Path, model_name: str | None = None, max_total_tokens: int = 5000) -> None:
        self.repository_path = Path(repository_path).resolve()
        if not self.repository_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {self.repository_path}")

        self.model_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        self.budget_manager = TokenBudgetManager(self.model_name, max_total_tokens=max_total_tokens)

        self.summary_memory = SummaryMemoryStore()
        self.entity_memory = EntityFactStore()
        self.task_conversation_memories = TaskConversationMemoryRouter(self.budget_manager)
        self.build_memory = BuildExecutionMemory()

        self.indexer = CodebaseIndexer(self.repository_path)
        self.indexer.build_index()
        self._seed_memories_from_index()

        self.retrieval_engine = RepositoryRetrievalEngine(self.indexer, self.summary_memory, self.entity_memory)
        self.command_executor = BuildCommandExecutor(self.repository_path)
        self.failure_analyzer = BuildFailureAnalyzer()
        self.llm_gateway = LlmGateway(self.model_name)
        self.fallback_responder = DeterministicDemoResponder()

    def answer_query(self, query_text: str) -> AgentAnswer:
        """Route the user request into the right agent path and answer it."""
        task_type = self._classify_task(query_text)
        self.task_conversation_memories.add_turn(task_type, "user", query_text)

        if task_type == self.TASK_BUILD_EXECUTION:
            answer = self._answer_build_execution_query(query_text)
        else:
            answer = self._answer_understanding_query(query_text)

        self.task_conversation_memories.add_turn(task_type, "assistant", answer.answer_text)
        return answer

    def _seed_memories_from_index(self) -> None:
        """Populate summary and entity memory using the freshly built index."""
        for indexed_file in self.indexer.indexed_files.values():
            self.summary_memory.remember_summary(indexed_file.path, indexed_file.summary)
            for symbol_name in indexed_file.symbols[:8]:
                self.entity_memory.remember_grounded_entity(
                    symbol_name,
                    f"{symbol_name} appears in {indexed_file.path}",
                    "code",
                    indexed_file.path,
                )

    def _classify_task(self, query_text: str) -> str:
        """Choose the happy-path workflow based on the user request."""
        lowered_query = query_text.lower()
        build_terms = ["build", "cmake", "make", "ctest", "test suite", "compile", "compiler", "failing", "run the tests"]
        for term in build_terms:
            if term in lowered_query:
                return self.TASK_BUILD_EXECUTION
        return self.TASK_UNDERSTANDING

    def _answer_understanding_query(self, query_text: str) -> AgentAnswer:
        """Handle architecture and source-understanding questions."""
        retrieved_matches = self.retrieval_engine.search(query_text, limit=5)
        prompt_text, prompt_report = self._build_prompt_text(query_text, self.TASK_UNDERSTANDING, retrieved_matches, [])

        if self.llm_gateway.can_call_model():
            try:
                answer_text = self.llm_gateway.generate_answer(
                    "You explain a local C++ codebase using only grounded retrieved context. Say clearly when evidence is limited.",
                    prompt_text,
                )
            except Exception:
                answer_text = self.fallback_responder.build_understanding_answer(query_text, retrieved_matches)
        else:
            answer_text = self.fallback_responder.build_understanding_answer(query_text, retrieved_matches)

        return AgentAnswer(self.TASK_UNDERSTANDING, answer_text, prompt_report=prompt_report)

    def _answer_build_execution_query(self, query_text: str) -> AgentAnswer:
        """Handle configure, build, test, and failure-explanation requests."""
        command_list = self._extract_commands_from_query(query_text)
        if not command_list:
            command_list = self.command_executor.build_happy_path_commands()

        command_results = []
        error_chunk = None
        preferred_path = ""
        failure_query_text = query_text
        failure_analysis = None
        execution_notes = []
        retry_used = False

        while True:
            retry_requested = False
            for command_parts in command_list:
                command_result = self.command_executor.run_command(command_parts)
                command_results.append(command_result)
                output_summary = self._summarize_output(command_result.get_combined_output(), 800)
                self.build_memory.add_step(command_result.get_command_text(), output_summary)

                if command_result.return_code == 0:
                    continue

                error_location = self.command_executor.find_error_location(command_result.get_combined_output())
                if error_location is not None:
                    preferred_path = error_location[0]
                    error_chunk = self.retrieval_engine.find_chunk_for_file_reference(error_location[0], error_location[1])
                    self.entity_memory.remember_grounded_entity(
                        error_location[0],
                        f"Build failure referenced {error_location[0]}:{error_location[1]}",
                        "build",
                        error_location[0],
                    )
                    failure_analysis = self.failure_analyzer.analyze_output(
                        command_result.get_combined_output(),
                        error_location[0],
                        error_location[1],
                        command_result.get_command_text(),
                    )
                else:
                    failure_analysis = self.failure_analyzer.analyze_output(
                        command_result.get_combined_output(),
                        command_text=command_result.get_command_text(),
                    )
                failure_query_text = failure_analysis.build_retrieval_query(query_text)

                if failure_analysis.should_retry_with_happy_path and not retry_used and self._can_retry_with_happy_path(command_list):
                    retry_used = True
                    retry_requested = True
                    command_list = self.command_executor.build_happy_path_commands()
                    execution_notes.append("Retry note: the initial command path lacked setup, so the agent retried once with the default configure/build/test flow.")
                    self.build_memory.add_step("retry decision", execution_notes[-1])
                    break
                execution_notes.append(f"Stop note: {failure_analysis.stop_reason}")
                break
            if retry_requested:
                continue
            break

        retrieved_matches = self.retrieval_engine.search(failure_query_text, limit=3, preferred_path=preferred_path)
        if error_chunk is not None:
            retrieved_matches.insert(0, RetrievedChunkMatch(999, "exact compiler error location", error_chunk))
        if failure_analysis is not None:
            retrieved_matches = self._expand_failure_context(query_text, failure_analysis, retrieved_matches, preferred_path, error_chunk)

        prompt_text, prompt_report = self._build_prompt_text(
            query_text,
            self.TASK_BUILD_EXECUTION,
            retrieved_matches,
            command_results,
            failure_analysis,
            execution_notes,
        )

        if self.llm_gateway.can_call_model():
            try:
                answer_text = self.llm_gateway.generate_answer(
                    "You explain CMake, compiler, and test results for a local C++ codebase. Ground every claim in command output or retrieved code.",
                    prompt_text,
                )
            except Exception:
                answer_text = self.fallback_responder.build_execution_answer(command_results, error_chunk, failure_analysis, execution_notes)
        else:
            answer_text = self.fallback_responder.build_execution_answer(command_results, error_chunk, failure_analysis, execution_notes)

        return AgentAnswer(self.TASK_BUILD_EXECUTION, answer_text, command_results, prompt_report)

    def _extract_commands_from_query(self, query_text: str) -> list[list[str]]:
        """Extract explicitly quoted commands from the user request."""
        quoted_commands = re.findall(r"`([^`]+)`", query_text)
        command_list = []
        for quoted_command in quoted_commands:
            command_list.append(quoted_command.split())
        return command_list

    def _build_prompt_text(
        self,
        query_text: str,
        task_type: str,
        retrieved_matches,
        command_results,
        failure_analysis=None,
        execution_notes=None,
    ) -> tuple[str, object]:
        """Assemble a prompt with per-section caps and return an observability report."""
        sections = []
        sections.append(PromptSection("task", f"task_type: {task_type}", 100))
        sections.append(PromptSection("query", f"user_query:\n{query_text}", 100))
        sections.append(
            PromptSection(
                "conversation",
                f"recent_turns:\n{self.task_conversation_memories.render_for_task(task_type)}",
                70,
            )
        )

        summary_matches = self.summary_memory.find_relevant_summaries(query_text.split(), limit=5)
        if summary_matches:
            summary_lines = []
            for key, summary_text in summary_matches:
                summary_lines.append(f"{key}: {summary_text}")
            sections.append(PromptSection("summaries", "summary_memory:\n" + "\n".join(summary_lines), 80))

        if task_type == self.TASK_BUILD_EXECUTION:
            sections.append(PromptSection("build_chain", f"build_chain:\n{self.build_memory.render()}", 85))
            if failure_analysis is not None:
                analysis_lines = [
                    f"failure_kind: {failure_analysis.failure_kind}",
                    f"stop_reason: {failure_analysis.stop_reason}",
                ]
                if failure_analysis.error_line:
                    analysis_lines.append(f"error_line: {failure_analysis.error_line}")
                if execution_notes:
                    analysis_lines.append("execution_notes:\n" + "\n".join(execution_notes))
                sections.append(PromptSection("build_analysis", "\n".join(analysis_lines), 90))

        section_priority = 95
        for command_result in command_results:
            command_text = command_result.get_command_text()
            command_output = self._summarize_output(command_result.get_combined_output(), 1200)
            command_section_text = (
                f"command: {command_text}\n"
                f"exit_code: {command_result.return_code}\n"
                f"output:\n{command_output}"
            )
            sections.append(PromptSection("command_output", command_section_text, section_priority))
            section_priority -= 1

        retrieval_priority = 92
        for retrieved_match in retrieved_matches:
            retrieval_text = (
                f"retrieval_reason: {retrieved_match.reason}\n"
                f"file: {retrieved_match.chunk.path}:{retrieved_match.chunk.start_line}-{retrieved_match.chunk.end_line}\n"
                f"symbols: {', '.join(retrieved_match.chunk.symbols)}\n"
                f"content:\n{retrieved_match.chunk.content}"
            )
            sections.append(PromptSection("retrieval", retrieval_text, retrieval_priority))
            retrieval_priority -= 2

        section_caps = self.SECTION_TOKEN_CAPS[task_type]
        kept_sections, prompt_report = self.budget_manager.trim_sections_with_caps(sections, section_caps)
        prompt_parts = []
        for section in kept_sections:
            prompt_parts.append(section.text)
        return "\n\n".join(prompt_parts), prompt_report

    def _can_retry_with_happy_path(self, command_list: list[list[str]]) -> bool:
        """Return True when one retry with the default command sequence is safe."""
        happy_path_commands = self.command_executor.build_happy_path_commands()
        if len(command_list) != len(happy_path_commands):
            return True
        for current_command, happy_path_command in zip(command_list, happy_path_commands):
            if current_command[:2] != happy_path_command[:2]:
                return True
        return False

    def _expand_failure_context(self, query_text: str, failure_analysis, retrieved_matches, preferred_path: str, error_chunk):
        """Retrieve additional code context when the first failure chunk is too weak."""
        if failure_analysis.failure_kind != "code":
            return retrieved_matches
        if self._has_actionable_failure_context(retrieved_matches, failure_analysis, error_chunk):
            return retrieved_matches

        expanded_query = failure_analysis.build_retrieval_query(query_text)
        extra_matches = self.retrieval_engine.search(expanded_query, limit=5, preferred_path=preferred_path)
        merged_matches = []
        seen_locations = set()
        for match in list(retrieved_matches) + extra_matches:
            location_key = f"{match.chunk.path}:{match.chunk.start_line}:{match.chunk.end_line}"
            if location_key in seen_locations:
                continue
            merged_matches.append(match)
            seen_locations.add(location_key)
            if len(merged_matches) >= 5:
                break
        return merged_matches

    def _has_actionable_failure_context(self, retrieved_matches, failure_analysis, error_chunk) -> bool:
        """Return True when the current retrieval already explains the failing code area."""
        if error_chunk is not None:
            return True
        if not retrieved_matches:
            return False
        for match in retrieved_matches:
            if failure_analysis.file_path and match.chunk.path == failure_analysis.file_path:
                return True
            if self._chunk_matches_failure_identifiers(match.chunk.content, failure_analysis.identifiers):
                return True
        return False

    def _chunk_matches_failure_identifiers(self, chunk_content: str, identifiers: list[str]) -> bool:
        """Return True when any failure identifier appears in the retrieved chunk."""
        lowered_content = chunk_content.lower()
        for identifier in identifiers:
            if identifier.lower() in lowered_content:
                return True
        return False

    def _summarize_output(self, output_text: str, limit: int) -> str:
        """Shorten large command output before it enters memory or the prompt."""
        cleaned_output = output_text.strip()
        if len(cleaned_output) <= limit:
            return cleaned_output
        first_half = cleaned_output[: limit // 2]
        second_half = cleaned_output[-limit // 2 :]
        return f"{first_half}\n...\n{second_half}"
