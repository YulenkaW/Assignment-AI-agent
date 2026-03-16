"""Context manager and token-budget enforcement."""

from __future__ import annotations

import logging

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None

from .contracts import ExecutionBatch, PlanStep, PromptAssemblyReport, PromptSection, RetrievalBatch, RouteDecision, WorkingMemory


class ContextManager:
    """Own the prompt budget and working-memory selection.
    This decides what evidence actually fits into the prompt:
user query
route
plan
selected retrieval chunks
selected execution results
analysis summary
external memory summaries
It trims and compresses based on token budget."""

    def __init__(self, model_name: str, max_total_tokens: int = 5000, reserved_output_tokens: int = 900, reserved_instruction_tokens: int = 500) -> None:
        self.model_name = model_name
        self.max_total_tokens = max_total_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.reserved_instruction_tokens = reserved_instruction_tokens
        self.logger = logging.getLogger(__name__)
        self.encoding = self._load_encoding(model_name)

    def get_retrieval_token_capacity(self) -> int:
        """Return the token budget that retrieval is allowed to consume."""
        safe_threshold = self.max_total_tokens - self.reserved_output_tokens - self.reserved_instruction_tokens
        if safe_threshold < 800:
            return 800
        return safe_threshold

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken when available."""
        if self.encoding is None:
            return max(1, len(text.split()))
        return len(self.encoding.encode(text))

    def assemble_working_memory(
        self,
        query_text: str,
        route_decision: RouteDecision,
        plan_steps: list[PlanStep],
        retrieval_batch: RetrievalBatch | None,
        execution_batches: list[ExecutionBatch],
        analysis_text: str,
        external_memory_records,
        conversation_turns: list[tuple[str, str]] | None = None,
    ) -> WorkingMemory:
        """Assemble the final prompt text with trimming and summarization."""
        sections = []
        sections.append(PromptSection("query", f"user_query:\n{query_text}", 100))
        sections.append(PromptSection("route", self._render_route(route_decision), 95))
        if conversation_turns:
            sections.append(PromptSection("conversation", self._render_conversation(conversation_turns), 93))
        sections.append(PromptSection("plan", self._render_plan(plan_steps), 90))
        sections.append(PromptSection("external_memory", self._render_external_memory(external_memory_records), 70))

        selected_candidates = []
        if retrieval_batch is not None:
            for offset, candidate in enumerate(retrieval_batch.candidates):
                selected_candidates.append(candidate)
                sections.append(PromptSection("retrieval", self._render_retrieval_candidate(candidate), 85 - offset))

        priority = 82
        for execution_batch in execution_batches:
            for result in execution_batch.results:
                sections.append(PromptSection("execution", self._render_execution_result(result), priority))
                priority -= 1

        if analysis_text:
            sections.append(PromptSection("analysis", f"analysis:\n{analysis_text}", 88))

        kept_sections, report = self._trim_sections(sections)
        prompt_parts = []
        for section in kept_sections:
            prompt_parts.append(section.text)
        return WorkingMemory("\n\n".join(prompt_parts), report, selected_candidates)

    def _load_encoding(self, model_name: str):
        """Load the model tokenizer when available.

        In restricted environments `tiktoken.encoding_for_model()` may try to fetch
        remote metadata or encoder files. The assignment agent must still run, so we
        fall back to a local base encoding or plain word counting instead of failing.
        """
        if tiktoken is None:
            return None
        try:
            return tiktoken.encoding_for_model(model_name)
        except Exception as error:  # pragma: no cover - depends on local tokenizer cache state
            self.logger.debug("Falling back to local token encoding for %s: %s", model_name, error)
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception as fallback_error:  # pragma: no cover
                self.logger.debug("Falling back to approximate token counting: %s", fallback_error)
                return None

    def _render_route(self, route_decision: RouteDecision) -> str:
        """Render router output for the reasoning layer."""
        subgoal_text = ", ".join(route_decision.subgoals)
        return (
            f"route_task_type: {route_decision.task_type}\n"
            f"confidence: {route_decision.confidence}\n"
            f"needs_retrieval: {route_decision.needs_retrieval}\n"
            f"needs_execution: {route_decision.needs_execution}\n"
            f"preferred_flow: {route_decision.preferred_flow}\n"
            f"subgoals: {subgoal_text}"
        )

    def _render_plan(self, plan_steps: list[PlanStep]) -> str:
        """Render planner steps for traceability."""
        lines = []
        for step in plan_steps:
            lines.append(f"{step.name}: {step.action_type} [{step.status}] {step.detail_text}")
        return "plan_steps:\n" + "\n".join(lines)

    def _render_external_memory(self, external_memory_records) -> str:
        """Render persisted external memory."""
        lines = []
        for record in external_memory_records:
            lines.append(f"{record.key}: {record.summary_text}")
        return "external_memory:\n" + "\n".join(lines)

    def _render_conversation(self, conversation_turns: list[tuple[str, str]]) -> str:
        """Render recent conversation turns for follow-up support."""
        lines = []
        recent_turns = conversation_turns[-6:]
        for role, text in recent_turns:
            lines.append(f"{role}: {text}")
        return "conversation:\n" + "\n".join(lines)

    def _render_retrieval_candidate(self, candidate) -> str:
        """Render one retrieval candidate."""
        return (
            f"retrieval_reason: {candidate.reason}\n"
            f"file_path: {candidate.chunk.file_path}\n"
            f"chunk_id: {candidate.chunk_id}\n"
            f"chunk_type: {candidate.chunk.chunk_type}\n"
            f"symbols: {', '.join(candidate.chunk.symbols)}\n"
            f"content:\n{candidate.chunk.content}"
        )

    def _render_execution_result(self, result) -> str:
        """Render one command execution result."""
        output_text = result.get_combined_output().strip()
        if self.count_tokens(output_text) > 400:
            output_text = self._compress_text(output_text, 400)
        return (
            f"phase: {result.phase_name}\n"
            f"command: {result.get_command_text()}\n"
            f"exit_code: {result.exit_code}\n"
            f"output:\n{output_text}"
        )

    def _compress_text(self, text: str, token_limit: int) -> str:
        """Compress long text by keeping the start and end."""
        if token_limit <= 0:
            return ""
        lines = text.splitlines()
        if len(lines) <= 12:
            return text
        kept_lines = lines[:6] + ["[omitted middle output]"] + lines[-6:]
        candidate_text = "\n".join(kept_lines)
        if self.count_tokens(candidate_text) <= token_limit:
            return candidate_text
        words = text.split()
        kept_words = []
        for word in words:
            kept_words.append(word)
            candidate_text = " ".join(kept_words)
            if self.count_tokens(candidate_text) > token_limit:
                kept_words.pop()
                break
        return " ".join(kept_words)

    def _trim_sections(self, sections: list[PromptSection]) -> tuple[list[PromptSection], PromptAssemblyReport]:
        """Trim prompt sections to the input budget."""
        ordered_sections = sorted(sections, key=self._section_sort_key, reverse=True)
        report = PromptAssemblyReport()
        total_tokens = 0
        kept_sections = []
        max_input_tokens = self.max_total_tokens - self.reserved_output_tokens

        for section in ordered_sections:
            section_tokens = self.count_tokens(section.text)
            if total_tokens + section_tokens <= max_input_tokens:
                kept_sections.append(section)
                total_tokens += section_tokens
                report.add_decision(section.label, "kept", section_tokens, "")
                continue

            truncated_text = self._compress_text(section.text, max_input_tokens - total_tokens)
            if truncated_text and truncated_text != section.text and self.count_tokens(truncated_text) + total_tokens <= max_input_tokens:
                truncated_tokens = self.count_tokens(truncated_text)
                kept_sections.append(PromptSection(section.label, truncated_text, section.priority))
                total_tokens += truncated_tokens
                report.add_decision(section.label, "truncated", truncated_tokens, "trimmed by global budget")
            else:
                report.add_decision(section.label, "dropped", section_tokens, "global budget exceeded")

        report.set_total_tokens(total_tokens)
        return kept_sections, report

    def _section_sort_key(self, section: PromptSection) -> int:
        """Return the deterministic section sort key."""
        return section.priority
