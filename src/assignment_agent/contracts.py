"""Core wrapper classes shared by the separate assignment agent"""

from __future__ import annotations


class QueryRequest:
    """Stores the incoming user request and the repository root."""

    def __init__(self, repository_path: str, query_text: str) -> None:
        self.repository_path = repository_path
        self.query_text = query_text


class RouteDecision:
    """Stores deterministic task-routing output."""

    def __init__(
        self,
        task_type: str,
        confidence: float,
        needs_retrieval: bool,
        needs_execution: bool,
        preferred_flow: str,
        subgoals: list[str],
        query_mode: str = "understanding",
        execution_requested: bool | None = None,
    ) -> None:
        self.task_type = task_type
        self.confidence = confidence
        self.needs_retrieval = needs_retrieval
        self.needs_execution = needs_execution
        self.preferred_flow = preferred_flow
        self.subgoals = subgoals
        self.query_mode = query_mode
        self.execution_requested = needs_execution if execution_requested is None else execution_requested


class PlanStep:
    """Stores one planner step."""

    STATUS_PENDING = "pending"
    STATUS_COMPLETED = "completed"

    def __init__(self, name: str, action_type: str, detail_text: str) -> None:
        self.name = name
        self.action_type = action_type
        self.detail_text = detail_text
        self.status = self.STATUS_PENDING

    def mark_completed(self) -> None:
        """Mark the step as completed."""
        self.status = self.STATUS_COMPLETED


class AgentPlan:
    """Stores the planner output and the execution policy for one query."""

    def __init__(
        self,
        route_decision: RouteDecision,
        retrieval_query: str,
        stop_conditions: list[str],
        steps: list[PlanStep],
    ) -> None:
        self.route_decision = route_decision
        self.retrieval_query = retrieval_query
        self.stop_conditions = stop_conditions
        self.steps = steps


class RetrievalBudget:
    """Stores the retrieval-budget policy for one query."""

    def __init__(self, max_retrieval_tokens: int, result_limit: int, supporting_chunks_per_file: int, neighbor_limit: int = 0) -> None:
        self.max_retrieval_tokens = max_retrieval_tokens
        self.result_limit = result_limit
        self.supporting_chunks_per_file = supporting_chunks_per_file
        self.neighbor_limit = neighbor_limit


class EvidenceRequirements:
    """Stores the minimum evidence needed before answering."""

    def __init__(
        self,
        summary_text: str,
        required_items: list[str],
        minimum_files: int = 1,
        supporting_chunks_per_file: int = 1,
        neighbor_chunks: int = 0,
        synthesis_required: bool = False,
    ) -> None:
        self.summary_text = summary_text
        self.required_items = required_items
        self.minimum_files = minimum_files
        self.supporting_chunks_per_file = supporting_chunks_per_file
        self.neighbor_chunks = neighbor_chunks
        self.synthesis_required = synthesis_required


class RetrievalStep:
    """Stores one retrieval operator in the ordered fallback chain."""

    def __init__(self, operator_name: str, rationale: str) -> None:
        self.operator_name = operator_name
        self.rationale = rationale


class RetrievalPlan:
    """Stores the goal-driven retrieval planner decision."""

    def __init__(
        self,
        task_kind: str,
        query_text: str,
        budget: RetrievalBudget,
        evidence_requirements: EvidenceRequirements,
        retrieval_steps: list[RetrievalStep],
        preferred_files: list[str] | None = None,
        preferred_lines: list[int] | None = None,
        literal_text: str = "",
        scope_prefixes: list[str] | None = None,
    ) -> None:
        self.task_kind = task_kind
        self.query_text = query_text
        self.budget = budget
        self.evidence_requirements = evidence_requirements
        self.retrieval_steps = retrieval_steps
        self.preferred_files = preferred_files or []
        self.preferred_lines = preferred_lines or []
        self.literal_text = literal_text
        self.scope_prefixes = scope_prefixes or []
        self.limit = budget.result_limit
        self.search_type = retrieval_steps[0].operator_name if retrieval_steps else task_kind


class ActionProposal:
    """Stores the orchestrator's proposed next action."""

    def __init__(self, action_name: str, rationale: str, source: str) -> None:
        self.action_name = action_name
        self.rationale = rationale
        self.source = source


class RepositoryChunk:
    """Stores one chunk of source code or build metadata."""

    def __init__(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        chunk_type: str,
        summary: str,
        content: str,
        symbols: list[str],
        token_count: int = 0,
    ) -> None:
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.chunk_type = chunk_type
        self.summary = summary
        self.content = content
        self.symbols = symbols
        self.token_count = token_count

    def get_location_text(self) -> str:
        """Return a stable file:line-span reference."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


class IndexedFile:
    """Stores the repository index representation for one file."""

    def __init__(self, file_path: str, file_type: str, symbols: list[str], summary: str, chunks: list[RepositoryChunk]) -> None:
        self.file_path = file_path
        self.file_type = file_type
        self.symbols = symbols
        self.summary = summary
        self.chunks = chunks


class RetrievalCandidate:
    """Stores one retrieved result and its evidence metadata."""

    def __init__(self, file_path: str, chunk_id: str, relevance_score: int, reason: str, chunk: RepositoryChunk) -> None:
        self.file_path = file_path
        self.chunk_id = chunk_id
        self.relevance_score = relevance_score
        self.reason = reason
        self.chunk = chunk


class RetrievalBatch:
    """Stores the retrieval output handed to the context manager."""

    def __init__(
        self,
        candidates: list[RetrievalCandidate],
        search_type: str,
        dropped_candidates: list[RetrievalCandidate] | None = None,
        literal_text: str = "",
        scope_prefixes: list[str] | None = None,
    ) -> None:
        self.candidates = candidates
        self.search_type = search_type
        self.dropped_candidates = dropped_candidates or []
        self.literal_text = literal_text
        self.scope_prefixes = scope_prefixes or []


class ExternalMemoryRecord:
    """Stores one persisted external-memory record."""

    def __init__(self, key: str, summary_text: str, source_path: str) -> None:
        self.key = key
        self.summary_text = summary_text
        self.source_path = source_path


class PromptSection:
    """Stores one prompt section before token trimming."""

    def __init__(self, label: str, text: str, priority: int) -> None:
        self.label = label
        self.text = text
        self.priority = priority


class PromptAssemblyDecision:
    """Stores one keep/drop/truncate decision made by the context manager."""

    def __init__(self, label: str, status: str, token_count: int, detail_text: str) -> None:
        self.label = label
        self.status = status
        self.token_count = token_count
        self.detail_text = detail_text


class PromptAssemblyReport:
    """Stores the prompt-assembly audit trail."""

    def __init__(self) -> None:
        self.total_tokens = 0
        self.decisions = []

    def add_decision(self, label: str, status: str, token_count: int, detail_text: str) -> None:
        """Record one prompt assembly decision."""
        self.decisions.append(PromptAssemblyDecision(label, status, token_count, detail_text))

    def set_total_tokens(self, total_tokens: int) -> None:
        """Store the final prompt token count."""
        self.total_tokens = total_tokens

    def render_text(self) -> str:
        """Render the report in a simple textual format."""
        lines = [f"total_tokens={self.total_tokens}"]
        for decision in self.decisions:
            detail_suffix = f" ({decision.detail_text})" if decision.detail_text else ""
            lines.append(f"{decision.label}: {decision.status} tokens={decision.token_count}{detail_suffix}")
        return "\n".join(lines)


class WorkingMemory:
    """Stores the selected prompt text and the audit report."""

    def __init__(self, prompt_text: str, report: PromptAssemblyReport, selected_chunks: list[RetrievalCandidate]) -> None:
        self.prompt_text = prompt_text
        self.report = report
        self.selected_chunks = selected_chunks


class CommandExecutionResult:
    """Stores the output of one executed command."""

    def __init__(self, command_parts: list[str], exit_code: int, stdout_text: str, stderr_text: str, phase_name: str = "execution") -> None:
        self.command_parts = command_parts
        self.exit_code = exit_code
        self.stdout_text = stdout_text
        self.stderr_text = stderr_text
        self.phase_name = phase_name

    def get_command_text(self) -> str:
        """Return the printable command string."""
        return " ".join(self.command_parts)

    def get_combined_output(self) -> str:
        """Return merged stdout and stderr output."""
        parts = []
        if self.stdout_text.strip():
            parts.append(self.stdout_text)
        if self.stderr_text.strip():
            parts.append(self.stderr_text)
        return "\n".join(parts)


class ExecutionBatch:
    """Stores the full execution path output for one phase."""

    def __init__(self, phase_name: str, results: list[CommandExecutionResult]) -> None:
        self.phase_name = phase_name
        self.results = results

    def has_failure(self) -> bool:
        """Return True when any command failed."""
        for result in self.results:
            if result.exit_code != 0:
                return True
        return False


class CapturedCommandOutput:
    """Stores normalized command output before parsing."""

    def __init__(self, command_text: str, exit_code: int, stdout_text: str, stderr_text: str, combined_output: str) -> None:
        self.command_text = command_text
        self.exit_code = exit_code
        self.stdout_text = stdout_text
        self.stderr_text = stderr_text
        self.combined_output = combined_output


class ParsedCommandOutput:
    """Stores parsed command evidence."""

    def __init__(
        self,
        command_text: str,
        exit_code: int,
        error_lines: list[str],
        file_references: list[tuple[str, int]],
        missing_command: bool,
    ) -> None:
        self.command_text = command_text
        self.exit_code = exit_code
        self.error_lines = error_lines
        self.file_references = file_references
        self.missing_command = missing_command


class RootCauseCandidate:
    """Stores one parsed failure candidate."""

    def __init__(self, summary_text: str, file_path: str = "", line_number: int | None = None, confidence: float = 0.0) -> None:
        self.summary_text = summary_text
        self.file_path = file_path
        self.line_number = line_number
        self.confidence = confidence


class AnalysisReport:
    """Stores parsed execution evidence and suggested next actions."""

    def __init__(
        self,
        first_reported_error: str,
        root_cause_candidates: list[RootCauseCandidate],
        relevant_files: list[str],
        line_numbers: list[int],
        recommended_next_action: str,
    ) -> None:
        self.first_reported_error = first_reported_error
        self.root_cause_candidates = root_cause_candidates
        self.relevant_files = relevant_files
        self.line_numbers = line_numbers
        self.recommended_next_action = recommended_next_action


class StopDecision:
    """Stores stop or retry control output."""

    def __init__(self, should_stop: bool, reason: str) -> None:
        self.should_stop = should_stop
        self.reason = reason


class ReasoningOutcome:
    """Stores the evidence-grounded reasoning result before formatting."""

    def __init__(self, summary_text: str, evidence_lines: list[str], next_steps: list[str]) -> None:
        self.summary_text = summary_text
        self.evidence_lines = evidence_lines
        self.next_steps = next_steps


class QueryDiagnostics:
    """Stores per-query diagnostics for UI display and verification."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.processing_time_ms = 0
        self.openai_configured = False
        self.route_task_type = ""
        self.query_mode = ""
        self.preferred_flow = ""
        self.retrieval_task_kind = ""
        self.retrieval_search_type = ""
        self.retrieval_candidates = 0
        self.retrieval_dropped = 0
        self.selected_chunks = 0
        self.external_memory_hits = 0
        self.external_memory_total = 0
        self.execution_batches = 0
        self.executed_commands = 0
        self.session_turns_used = 0
        self.cmake_available = False
        self.ctest_available = False
        self.orchestrator_attempts = 0
        self.orchestrator_successes = 0
        self.orchestrator_fallbacks = 0
        self.reasoning_attempts = 0
        self.reasoning_successes = 0
        self.reasoning_fallbacks = 0
        self.fallback_events = []

    def total_openai_attempts(self) -> int:
        """Return the total number of attempted OpenAI-backed calls."""
        return self.orchestrator_attempts + self.reasoning_attempts

    def total_openai_successes(self) -> int:
        """Return the total number of successful OpenAI-backed calls."""
        return self.orchestrator_successes + self.reasoning_successes

    def total_openai_fallbacks(self) -> int:
        """Return the total number of model-path fallbacks."""
        return self.orchestrator_fallbacks + self.reasoning_fallbacks

    def add_fallback(self, component: str, detail_text: str) -> None:
        """Record one fallback event for UI inspection."""
        self.fallback_events.append(f"{component}: {detail_text}")

    def render_text(self) -> str:
        """Render diagnostics in a compact text format."""
        lines = [
            f"model={self.model_name}",
            f"request_processing_time_ms={self.processing_time_ms}",
            f"openai_configured={self.openai_configured}",
            f"route_task_type={self.route_task_type}",
            f"query_mode={self.query_mode}",
            f"preferred_flow={self.preferred_flow}",
            f"retrieval_task_kind={self.retrieval_task_kind}",
            f"retrieval_search_type={self.retrieval_search_type}",
            f"retrieval_candidates={self.retrieval_candidates}",
            f"retrieval_dropped={self.retrieval_dropped}",
            f"selected_chunks={self.selected_chunks}",
            f"external_memory_hits={self.external_memory_hits}",
            f"external_memory_total={self.external_memory_total}",
            f"execution_batches={self.execution_batches}",
            f"executed_commands={self.executed_commands}",
            f"session_turns_used={self.session_turns_used}",
            f"cmake_available={self.cmake_available}",
            f"ctest_available={self.ctest_available}",
            f"orchestrator_attempts={self.orchestrator_attempts}",
            f"orchestrator_successes={self.orchestrator_successes}",
            f"orchestrator_fallbacks={self.orchestrator_fallbacks}",
            f"reasoning_attempts={self.reasoning_attempts}",
            f"reasoning_successes={self.reasoning_successes}",
            f"reasoning_fallbacks={self.reasoning_fallbacks}",
            f"total_openai_attempts={self.total_openai_attempts()}",
            f"total_openai_successes={self.total_openai_successes()}",
            f"total_openai_fallbacks={self.total_openai_fallbacks()}",
        ]
        if self.fallback_events:
            lines.append("fallback_events:")
            for event_text in self.fallback_events:
                lines.append(f"- {event_text}")
        return "\n".join(lines)


class AgentResponse:
    """Stores the final user-facing response and internal artifacts."""

    def __init__(
        self,
        task_type: str,
        answer_text: str,
        prompt_report: PromptAssemblyReport | None = None,
        execution_batches: list[ExecutionBatch] | None = None,
        analysis_report: AnalysisReport | None = None,
        error_records = None,
        route_decision: RouteDecision | None = None,
        diagnostics: QueryDiagnostics | None = None,
        external_memory_records: list[ExternalMemoryRecord] | None = None,
    ) -> None:
        self.task_type = task_type
        self.answer_text = answer_text
        self.prompt_report = prompt_report
        self.execution_batches = execution_batches or []
        self.analysis_report = analysis_report
        self.error_records = error_records or []
        self.route_decision = route_decision
        self.diagnostics = diagnostics
        self.external_memory_records = external_memory_records or []
