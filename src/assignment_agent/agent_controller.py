"""Top-level controller for the separate assignment agent."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
import logging
import os
import re
import time

from .build_runner import BuildRunner
from .build_test_analyzer import BuildTestAnalyzer
from .command_executor import CommandExecutor
from .command_output_capture import CommandOutputCapture
from .context_manager import ContextManager
from .contracts import AgentResponse, AnalysisReport, ExecutionBatch, QueryDiagnostics, QueryRequest, RetrievalBatch
from .env_loader import load_project_env
from .error_accumulator import ErrorAccumulator
from .external_memory_store import ExternalMemoryStore
from .output_parser import OutputParser
from .planning_layer import PlanningLayer
from .reasoning_engine import ReasoningEngine
from .repository_index import RepositoryIndex
from .repository_service import RepositoryService
from .response_generator import ResponseGenerator
from .retrieval_planner import RetrievalPlanner
from .stop_retry_controller import StopRetryController
from .task_router import TaskRouter
from .test_runner import TestRunner
from .workspace_paths import WorkspacePaths


class AssignmentAgentController:
    """Coordinate routing, planning, retrieval, execution, analysis, and response."""

    # Match short follow-up prompts that depend on the previous turn.
    FOLLOW_UP_PATTERN = re.compile(
        r"^(?:and\s+)?(?:what|where|why|how|does|do|did|it|that|those|these|them|what about|which)\b",
        re.IGNORECASE,
    )
    # Match self-contained code anchors such as symbols, namespaces, or source paths.
    SELF_CONTAINED_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]*|::|[A-Za-z0-9_./-]+\.(?:hpp|h|cpp|cc|cxx)")

    def __init__(self, repository_path: str | Path, model_name: str | None = None, max_total_tokens: int = 5000) -> None:
        self.repository_path = Path(repository_path).resolve()
        load_project_env(self.repository_path)
        self.model_name = model_name or "gpt-4.1-mini"
        self.logger = logging.getLogger(__name__)
        self.error_accumulator = ErrorAccumulator()
        self.project_root = Path(__file__).resolve().parents[2]
        self.workspace_paths = WorkspacePaths(self.repository_path, self.project_root)

        self.repository_index = RepositoryIndex(self.repository_path)
        self.repository_index.build()

        self.task_router = TaskRouter()
        self.context_manager = ContextManager(self.model_name, max_total_tokens=max_total_tokens)
        self.planning_layer = PlanningLayer()
        self.retrieval_planner = RetrievalPlanner()
        self.repository_service = RepositoryService(self.repository_index)
        self.external_memory_store = ExternalMemoryStore(self.project_root / ".assignment_agent_memory", self.repository_path)
        self.external_memory_store.seed_file_summaries(self.repository_index)
        self.build_runner = BuildRunner(self.repository_path, self.workspace_paths, self.error_accumulator)
        self.test_runner = TestRunner(self.repository_path, self.workspace_paths, self.error_accumulator)
        self.command_output_capture = CommandOutputCapture()
        self.output_parser = OutputParser(self.error_accumulator)
        self.analyzer = BuildTestAnalyzer(self.error_accumulator)
        self.stop_retry_controller = StopRetryController()
        self.reasoning_engine = ReasoningEngine(self.model_name)
        self.response_generator = ResponseGenerator()
        self.progress_callback: Callable[[str], None] | None = None

    def answer_query(
        self,
        query_text: str,
        conversation_turns: list[tuple[str, str]] | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        """Run the end-to-end assignment agent flow."""
        start_time = time.perf_counter()
        self.error_accumulator.clear()
        self.progress_callback = progress_callback
        try:
            diagnostics = self._build_diagnostics(conversation_turns)
            request = QueryRequest(str(self.repository_path), query_text)
            route_decision = self.task_router.route(request)
            self._report_progress(f"Routed query as {route_decision.task_type} ({route_decision.query_mode}).")
            plan = self.planning_layer.create_plan(request, route_decision)
            self.logger.debug("route=%s preferred_flow=%s", route_decision.task_type, route_decision.preferred_flow)
            diagnostics.route_task_type = route_decision.task_type
            diagnostics.query_mode = route_decision.query_mode
            diagnostics.preferred_flow = route_decision.preferred_flow

            retrieval_query = self._build_retrieval_query(query_text, conversation_turns)
            retrieval_batch, execution_batches, analysis_report = self._run_agent_loop(
                query_text,
                retrieval_query,
                route_decision,
                plan,
                diagnostics,
            )
            diagnostics.execution_batches = len(execution_batches)
            diagnostics.executed_commands = sum(len(execution_batch.results) for execution_batch in execution_batches)

            external_memory_records = self.external_memory_store.find_relevant_records(retrieval_query, limit=5)
            diagnostics.external_memory_total = self.external_memory_store.get_record_count()
            diagnostics.external_memory_hits = len(external_memory_records)
            analysis_text = self._render_analysis(analysis_report)

            working_memory = self.context_manager.assemble_working_memory(
                query_text,
                route_decision,
                plan.steps,
                retrieval_batch,
                execution_batches,
                analysis_text,
                external_memory_records,
                conversation_turns,
            )
            diagnostics.selected_chunks = len(working_memory.selected_chunks)
            self._mark_step_completed(plan, "reasoning")

            reasoning_outcome = self.reasoning_engine.reason(
                query_text,
                route_decision,
                retrieval_batch,
                execution_batches,
                analysis_report,
                working_memory,
                diagnostics,
            )
            answer_text = self.response_generator.generate(route_decision, reasoning_outcome, analysis_report)
            self._mark_step_completed(plan, "response")
            diagnostics.processing_time_ms = int((time.perf_counter() - start_time) * 1000)
            return AgentResponse(
                route_decision.task_type,
                answer_text,
                working_memory.report,
                execution_batches,
                analysis_report,
                self.error_accumulator.get_records(),
                route_decision,
                diagnostics,
                external_memory_records,
            )
        finally:
            self.progress_callback = None

    def _run_agent_loop(
        self,
        query_text: str,
        retrieval_query: str,
        route_decision,
        plan,
        diagnostics: QueryDiagnostics,
    ) -> tuple[RetrievalBatch | None, list[ExecutionBatch], AnalysisReport | None]:
        """Run the small deterministic controller loop for retrieval and execution."""
        retrieval_batch: RetrievalBatch | None = None
        execution_batches: list[ExecutionBatch] = []
        parsed_outputs = []
        analysis_report: AnalysisReport | None = None
        iteration_count = 0

        while iteration_count < self.stop_retry_controller.max_iterations:
            allowed_actions = self._build_allowed_actions(query_text, route_decision, retrieval_batch, execution_batches, parsed_outputs, analysis_report)
            next_action = self._select_next_action(query_text, route_decision, allowed_actions, retrieval_batch, execution_batches, parsed_outputs, analysis_report)
            self.logger.debug("iteration=%d allowed=%s chosen=%s", iteration_count, allowed_actions, next_action)

            if next_action in ("answer_user", "stop"):
                break
            if next_action == "retrieve_context":
                self._report_progress("Retrieving repository context.")
                retrieval_batch = self._run_retrieval(retrieval_query, route_decision, analysis_report, plan, diagnostics)
            elif next_action == "run_build":
                self._report_progress("Starting build phase.")
                execution_batches.append(self.build_runner.run_with_progress(query_text, self.progress_callback))
                self._mark_step_completed(plan, "build")
            elif next_action == "run_tests":
                self._report_progress("Starting test phase.")
                execution_batches.append(self.test_runner.run_with_progress(query_text, self.progress_callback))
                self._mark_step_completed(plan, "test")
            elif next_action == "parse_output":
                self._report_progress("Parsing command output.")
                parsed_outputs = self._parse_execution_batches(execution_batches)
                self._mark_step_completed(plan, "output_parse")
            elif next_action == "analyze_output":
                self._report_progress("Analyzing execution results.")
                analysis_report = self._analyze_outputs(parsed_outputs, plan)
                stop_decision = self.stop_retry_controller.decide(
                    iteration_count,
                    analysis_report,
                    self.context_manager.get_retrieval_token_capacity(),
                )
                self.logger.debug("analysis_next=%s stop=%s reason=%s", analysis_report.recommended_next_action, stop_decision.should_stop, stop_decision.reason)
                if stop_decision.should_stop and analysis_report.recommended_next_action != "retrieve_more_context":
                    break

            iteration_count += 1

        return retrieval_batch, execution_batches, analysis_report

    def _select_next_action(
        self,
        query_text: str,
        route_decision,
        allowed_actions: list[str],
        retrieval_batch,
        execution_batches,
        parsed_outputs,
        analysis_report,
    ) -> str:
        """Choose the next action from a deterministic controller state machine."""
        if not allowed_actions:
            return "stop"
        if route_decision.needs_execution:
            if not self._has_build_batch(execution_batches) and "run_build" in allowed_actions:
                return self._validate_action("run_build", allowed_actions)
            if self._should_run_tests(query_text, execution_batches) and "run_tests" in allowed_actions:
                return self._validate_action("run_tests", allowed_actions)
            if execution_batches and not parsed_outputs and "parse_output" in allowed_actions:
                return self._validate_action("parse_output", allowed_actions)
            if parsed_outputs and analysis_report is None and "analyze_output" in allowed_actions:
                return self._validate_action("analyze_output", allowed_actions)
            if "retrieve_context" in allowed_actions:
                return self._validate_action("retrieve_context", allowed_actions)
            return self._validate_action(allowed_actions[0], allowed_actions)
        if route_decision.needs_retrieval and retrieval_batch is None and "retrieve_context" in allowed_actions:
            return self._validate_action("retrieve_context", allowed_actions)
        return self._validate_action(allowed_actions[0], allowed_actions)

    def _validate_action(self, proposed_action: str, allowed_actions: list[str]) -> str:
        """Return one allowed action and record a warning when a proposal is invalid."""
        if proposed_action in allowed_actions:
            return proposed_action
        fallback_action = allowed_actions[0] if allowed_actions else "stop"
        self.error_accumulator.add(
            "agent_controller",
            "Rejected disallowed action",
            f"proposed={proposed_action} fallback={fallback_action}",
            "warning",
        )
        return fallback_action

    def _run_retrieval(self, query_text: str, route_decision, analysis_report, plan, diagnostics: QueryDiagnostics):
        """Plan retrieval first, then execute it."""
        retrieval_plan = self.retrieval_planner.create_plan(
            query_text,
            route_decision,
            self.context_manager.get_retrieval_token_capacity(),
            analysis_report,
        )
        retrieval_batch = self.repository_service.retrieve(retrieval_plan)
        diagnostics.retrieval_task_kind = retrieval_plan.task_kind
        diagnostics.retrieval_search_type = retrieval_plan.search_type
        diagnostics.retrieval_candidates = len(retrieval_batch.candidates)
        diagnostics.retrieval_dropped = len(retrieval_batch.dropped_candidates)
        self._mark_step_completed(plan, "retrieval_planning")
        self._mark_step_completed(plan, "retrieval")
        self.logger.debug("retrieval search_type=%s matches=%d", retrieval_plan.search_type, len(retrieval_batch.candidates))
        return retrieval_batch

    def _analyze_outputs(self, parsed_outputs, plan) -> AnalysisReport:
        """Analyze parsed output and persist the most useful note."""
        analysis_report = self.analyzer.analyze(parsed_outputs)
        if analysis_report.first_reported_error:
            self.external_memory_store.remember_note(
                f"analysis:{analysis_report.first_reported_error[:40]}",
                analysis_report.first_reported_error,
                analysis_report.relevant_files[0] if analysis_report.relevant_files else "execution",
            )
        self._mark_step_completed(plan, "analysis")
        return analysis_report

    def _build_allowed_actions(
        self,
        query_text: str,
        route_decision,
        retrieval_batch,
        execution_batches,
        parsed_outputs,
        analysis_report,
    ) -> list[str]:
        """Build the deterministic allowed action list."""
        actions = []
        if route_decision.needs_retrieval and retrieval_batch is None and route_decision.preferred_flow == "retrieval_first":
            actions.append("retrieve_context")
        if route_decision.needs_execution and not self._has_build_batch(execution_batches):
            actions.append("run_build")
        if route_decision.needs_execution and self._should_run_tests(query_text, execution_batches):
            actions.append("run_tests")
        if execution_batches and not parsed_outputs:
            actions.append("parse_output")
        if parsed_outputs and analysis_report is None:
            actions.append("analyze_output")
        if route_decision.needs_retrieval and retrieval_batch is None and analysis_report is not None and analysis_report.recommended_next_action == "retrieve_more_context":
            actions.append("retrieve_context")
        if route_decision.task_type == "mixed" and retrieval_batch is None and self._has_build_batch(execution_batches):
            actions.append("retrieve_context")
        if not actions:
            actions.append("answer_user")
        return actions

    def _parse_execution_batches(self, execution_batches) -> list:
        """Capture and parse every execution batch."""
        parsed_outputs = []
        for execution_batch in execution_batches:
            captured_outputs = self.command_output_capture.capture_batch(execution_batch)
            parsed_outputs.extend(self.output_parser.parse_outputs(captured_outputs))
        return parsed_outputs

    def _render_analysis(self, analysis_report) -> str:
        """Render the analysis report for the working memory."""
        if analysis_report is None:
            return ""
        lines = [f"first_reported_error: {analysis_report.first_reported_error}"]
        for candidate in analysis_report.root_cause_candidates:
            lines.append(
                f"root_cause_candidate: {candidate.summary_text} "
                f"path={candidate.file_path} line={candidate.line_number} confidence={candidate.confidence}"
            )
        lines.append(f"recommended_next_action: {analysis_report.recommended_next_action}")
        return "\n".join(lines)

    def _has_build_batch(self, execution_batches) -> bool:
        """Return True when a build batch already exists."""
        return any(execution_batch.phase_name == "build" for execution_batch in execution_batches)

    def _has_test_batch(self, execution_batches) -> bool:
        """Return True when a test batch already exists."""
        return any(execution_batch.phase_name == "test" for execution_batch in execution_batches)

    def _should_run_tests(self, query_text: str, execution_batches) -> bool:
        """Return True when tests should run next."""
        if self._has_test_batch(execution_batches):
            return False
        if not self._has_build_batch(execution_batches):
            return False
        latest_build_batch = None
        for execution_batch in reversed(execution_batches):
            if execution_batch.phase_name == "build":
                latest_build_batch = execution_batch
                break
        if latest_build_batch is None or latest_build_batch.has_failure():
            return False
        return self.test_runner.should_run_tests(query_text)

    def _mark_step_completed(self, plan, step_name: str) -> None:
        """Mark the named plan step as completed."""
        for step in plan.steps:
            if step.name == step_name:
                step.mark_completed()

    def _build_diagnostics(self, conversation_turns: list[tuple[str, str]] | None) -> QueryDiagnostics:
        """Create per-query diagnostics for UI display."""
        diagnostics = QueryDiagnostics(self.model_name)
        diagnostics.openai_configured = bool(os.environ.get("OPENAI_API_KEY"))
        diagnostics.session_turns_used = len(conversation_turns or [])
        diagnostics.cmake_available = CommandExecutor.resolve_command_path("cmake") is not None
        diagnostics.ctest_available = CommandExecutor.resolve_command_path("ctest") is not None
        return diagnostics

    def _build_retrieval_query(self, query_text: str, conversation_turns: list[tuple[str, str]] | None) -> str:
        """Expand short follow-up questions with recent user context for retrieval."""
        if not conversation_turns:
            return query_text
        if self.SELF_CONTAINED_PATTERN.search(query_text):
            return query_text
        if self.FOLLOW_UP_PATTERN.search(query_text) is None:
            return query_text
        recent_user_turns = [text for role, text in conversation_turns if role == "user"]
        if not recent_user_turns:
            return query_text
        return f"{recent_user_turns[-1]} {query_text}"

    def _report_progress(self, message_text: str) -> None:
        """Send one progress message to the caller when available."""
        if self.progress_callback is not None:
            self.progress_callback(message_text)
