"""Reasoning layer with optional OpenAI-backed synthesis."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover
    ChatPromptTemplate = None
    ChatOpenAI = None

from .contracts import AnalysisReport, ExecutionBatch, QueryDiagnostics, ReasoningOutcome, RetrievalBatch, RouteDecision, WorkingMemory
from .build_artifact_inspector import BuildArtifactInspector
from .execution_intent import ExecutionIntent


class ReasoningEngine:
    """Produce the final grounded reasoning outcome.
    This produces the actual explanation.
    It uses:
    deterministic fallback reasoning if no model/API is active
    optional OpenAI-backed reasoning if configured
    Important:
    it does not decide relevance from scratch
    it reasons over already selected evidence
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.chat_model = self._build_chat_model()
        self.build_artifact_inspector = BuildArtifactInspector()
        self.command_unavailable_markers = (
            "required command is unavailable",
            "the system cannot find the file specified",
            "[winerror 2]",
            "command not found",
            "is not recognized as the name of a cmdlet",
            "no such file or directory",
        )
        self.file_name_pattern = re.compile(r"[A-Za-z0-9_./-]+\.(?:hpp|h|cpp|cc|cxx|ipp|inl)")

    def reason(
        self,
        query_text: str,
        route_decision: RouteDecision,
        retrieval_batch: RetrievalBatch | None,
        execution_batches: list[ExecutionBatch],
        analysis_report: AnalysisReport | None,
        working_memory: WorkingMemory,
        diagnostics: QueryDiagnostics | None = None,
    ) -> ReasoningOutcome:
        """Create the grounded reasoning outcome."""
        missing_command = self._find_missing_command_name(execution_batches, analysis_report)
        if missing_command:
            return self._reason_deterministically(query_text, route_decision, retrieval_batch, execution_batches, analysis_report)
        if route_decision.task_type == "code_understanding":
            deterministic_outcome = self._reason_deterministically(
                query_text,
                route_decision,
                retrieval_batch,
                execution_batches,
                analysis_report,
            )
            return self._polish_grounded_understanding(query_text, deterministic_outcome, diagnostics)
        if route_decision.needs_execution and execution_batches:
            return self._reason_deterministically(query_text, route_decision, retrieval_batch, execution_batches, analysis_report)
        if working_memory is not None and self._can_call_model():
            if diagnostics is not None:
                diagnostics.reasoning_attempts += 1
            try:
                outcome = self._reason_with_model(query_text, route_decision, execution_batches, analysis_report, working_memory)
                if diagnostics is not None:
                    diagnostics.reasoning_successes += 1
                return outcome
            except Exception as error:
                self.logger.warning("Falling back after model reasoning failed: %s", error)
                self.chat_model = None
                if diagnostics is not None:
                    diagnostics.reasoning_fallbacks += 1
                    diagnostics.add_fallback("reasoning_engine", str(error))
        return self._reason_deterministically(query_text, route_decision, retrieval_batch, execution_batches, analysis_report)

    def _build_chat_model(self):
        """Create the OpenAI-backed chat model when configuration exists."""
        if ChatOpenAI is None:
            return None
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        return ChatOpenAI(model=self.model_name, temperature=0)

    def _can_call_model(self) -> bool:
        """Return True when model calls are available."""
        return self.chat_model is not None and ChatPromptTemplate is not None

    def _reason_with_model(
        self,
        query_text: str,
        route_decision: RouteDecision,
        execution_batches: list[ExecutionBatch],
        analysis_report: AnalysisReport | None,
        working_memory: WorkingMemory,
    ) -> ReasoningOutcome:
        """Use the configured model for the final explanation only."""
        system_text = (
            "You are a grounded repository-analysis assistant. "
            "Use deterministic evidence as ground truth. "
            "If evidence is incomplete, say so explicitly. "
            "When execution evidence is present, commands have already been attempted locally by the controller. "
            "Do not claim that you cannot run commands here. "
            "If a tool or executable was unavailable, report that concrete local failure instead."
        )
        analysis_text = ""
        if analysis_report is not None:
            analysis_text = analysis_report.recommended_next_action
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_text),
                (
                    "human",
                    "Query:\n{query}\n\nTask Type:\n{task_type}\n\nAnalysis Hint:\n{analysis_hint}\n\nWorking Memory:\n{memory}",
                ),
            ]
        )
        chain = prompt | self.chat_model
        response = chain.invoke(
            {
                "query": query_text,
                "task_type": route_decision.task_type,
                "analysis_hint": analysis_text,
                "memory": working_memory.prompt_text,
            }
        )
        evidence_lines = ["Used selected prompt evidence with deterministic routing, parsing, and token budgeting."]
        next_steps = self._build_user_next_steps(execution_batches, analysis_report)
        return ReasoningOutcome(response.content, evidence_lines, next_steps)

    def _reason_deterministically(
        self,
        query_text: str,
        route_decision: RouteDecision,
        retrieval_batch: RetrievalBatch | None,
        execution_batches: list[ExecutionBatch],
        analysis_report: AnalysisReport | None,
    ) -> ReasoningOutcome:
        """Build a deterministic explanation when API access is unavailable."""
        evidence_lines = []

        if retrieval_batch is not None:
            for candidate in retrieval_batch.candidates[:3]:
                evidence_lines.append(f"{candidate.chunk.get_location_text()} because {candidate.reason}")

        for execution_batch in execution_batches:
            for result in execution_batch.results:
                evidence_lines.append(f"{result.get_command_text()} exited with code {result.exit_code}")

        if route_decision.task_type == "code_understanding":
            summary_text = self._build_understanding_summary(query_text, route_decision, retrieval_batch)
        else:
            summary_text = self._build_execution_summary(query_text, execution_batches, analysis_report, retrieval_batch)

        next_steps = self._build_user_next_steps(execution_batches, analysis_report)
        return ReasoningOutcome(summary_text, evidence_lines, next_steps)

    def _polish_grounded_understanding(
        self,
        query_text: str,
        deterministic_outcome: ReasoningOutcome,
        diagnostics: QueryDiagnostics | None,
    ) -> ReasoningOutcome:
        """Rewrite grounded repository answers into clearer prose without adding new facts."""
        if not self._can_call_model():
            return deterministic_outcome

        if diagnostics is not None:
            diagnostics.reasoning_attempts += 1
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Rewrite the grounded repository answer into clear, user-friendly prose. "
                        "Use only the supplied grounded summary and evidence. "
                        "Do not add facts, files, symbols, or behavior that are not already supported. "
                        "Keep the meaning intact and prefer a short paragraph.",
                    ),
                    (
                        "human",
                        "Query:\n{query}\n\nGrounded Summary:\n{summary}\n\nEvidence:\n{evidence}",
                    ),
                ]
            )
            chain = prompt | self.chat_model
            response = chain.invoke(
                {
                    "query": query_text,
                    "summary": deterministic_outcome.summary_text,
                    "evidence": "\n".join(deterministic_outcome.evidence_lines) or "No evidence lines.",
                }
            )
            polished_summary = response.content.strip()
            if not polished_summary:
                return deterministic_outcome
            if diagnostics is not None:
                diagnostics.reasoning_successes += 1
            return ReasoningOutcome(
                polished_summary,
                deterministic_outcome.evidence_lines,
                deterministic_outcome.next_steps,
            )
        except Exception as error:
            self.logger.warning("Falling back after grounded-understanding polish failed: %s", error)
            self.chat_model = None
            if diagnostics is not None:
                diagnostics.reasoning_fallbacks += 1
                diagnostics.add_fallback("reasoning_engine", str(error))
            return deterministic_outcome

    def _build_user_next_steps(
        self,
        execution_batches: list[ExecutionBatch],
        analysis_report: AnalysisReport | None,
    ) -> list[str]:
        """Translate internal control hints into user-facing next steps."""
        next_steps = []
        missing_command = self._find_missing_command_name(execution_batches, analysis_report)
        if missing_command:
            next_steps.append(f"Install {missing_command} and confirm `{missing_command} --version` works in the same shell.")
            if missing_command == "cmake":
                next_steps.append("After installing CMake, confirm `ctest --version` also works because the test path depends on it.")

        if analysis_report is None:
            return next_steps
        if analysis_report.recommended_next_action == "retrieve_more_context":
            next_steps.append("Inspect the referenced file region before changing code.")
        elif analysis_report.recommended_next_action == "answer_with_limited_evidence":
            next_steps.append("Execution evidence is incomplete, so treat the explanation as partial.")
        return next_steps

    def _build_understanding_summary(
        self,
        query_text: str,
        route_decision: RouteDecision,
        retrieval_batch: RetrievalBatch | None,
    ) -> str:
        """Build the deterministic understanding summary."""
        if retrieval_batch is None:
            return f"I could not ground an answer for: {query_text}"
        if retrieval_batch.search_type == "literal_text" or retrieval_batch.literal_text:
            return self._build_search_summary(retrieval_batch)
        if self._is_file_responsibility_query(query_text):
            return self._build_file_responsibility_summary(retrieval_batch)
        if route_decision.query_mode == "search":
            return self._build_search_summary(retrieval_batch)
        if not retrieval_batch.candidates:
            return f"I could not ground an answer for: {query_text}"
        if self._is_file_summary_query(query_text):
            return self._build_file_summary(query_text, retrieval_batch)
        return self._build_symbol_summary(query_text, retrieval_batch)

    def _build_execution_summary(
        self,
        query_text: str,
        execution_batches: list[ExecutionBatch],
        analysis_report: AnalysisReport | None,
        retrieval_batch: RetrievalBatch | None,
    ) -> str:
        """Build the deterministic build/test/debug summary."""
        if not execution_batches:
            return "No commands were executed."
        execution_intent = ExecutionIntent.from_query(query_text)
        missing_command = self._find_missing_command_name(execution_batches, analysis_report)
        if missing_command:
            return (
                f"The build or test flow did not reach repository code. "
                f"`{missing_command}` is unavailable on this machine, so the current failure is environmental rather than a code defect."
            )
        if execution_intent.requests_show_cmake_options():
            options_summary = self._build_cmake_options_summary(execution_batches)
            if options_summary:
                return options_summary
        if execution_intent.requests_build_targets():
            targets_summary = self._build_build_target_summary(execution_batches)
            if targets_summary:
                return targets_summary
        if execution_intent.requests_list_tests():
            tests_summary = self._build_test_list_summary(execution_batches)
            if tests_summary:
                return tests_summary
        if execution_intent.requests_configure_only():
            configure_summary = self._build_configure_only_summary(execution_batches)
            if configure_summary:
                return configure_summary
        explicit_target = execution_intent.extract_build_target_name()
        any_failure = False
        for execution_batch in execution_batches:
            if execution_batch.has_failure():
                any_failure = True
                break
        if not any_failure:
            if execution_intent.requests_test_execution():
                return "The requested test flow completed without a detected failure."
            if explicit_target:
                return f"The requested build target `{explicit_target}` completed without a detected failure."
            return "The requested build flow completed without a detected failure."

        summary_parts = []
        if analysis_report is not None and analysis_report.root_cause_candidates:
            top_candidate = analysis_report.root_cause_candidates[0]
            summary_parts.append(f"Likely root cause: {top_candidate.summary_text}")
        if retrieval_batch is not None and retrieval_batch.candidates:
            summary_parts.append(f"Relevant code: {retrieval_batch.candidates[0].chunk.get_location_text()}")
        return " ".join(summary_parts)

    def _build_cmake_options_summary(self, execution_batches: list[ExecutionBatch]) -> str:
        """Summarize selected CMake cache options from a successful options query."""
        options = []
        seen_names = set()
        for result in self._iter_execution_results(execution_batches):
            if result.exit_code != 0:
                continue
            if "-LAH" not in result.command_parts:
                continue
            for raw_line in result.stdout_text.splitlines():
                line = raw_line.strip()
                match = re.match(r"([A-Za-z0-9_]+):([A-Za-z_]+)=(.*)", line)
                if match is None:
                    continue
                option_name = match.group(1)
                if option_name not in {"BUILD_TESTING", "JSON_BuildTests", "JSON_Valgrind", "JSON_FastTests", "JSON_Diagnostics", "JSON_Diagnostic_Positions"} and not option_name.startswith("JSON_"):
                    continue
                if option_name in seen_names:
                    continue
                seen_names.add(option_name)
                options.append(f"{option_name}={match.group(3)}")
                if len(options) >= 8:
                    break
        if not options:
            return ""
        return "CMake cache options: " + ", ".join(options) + "."

    def _build_build_target_summary(self, execution_batches: list[ExecutionBatch]) -> str:
        """Summarize available build targets from a successful target-help query."""
        targets = []
        seen_targets = set()
        for result in self._iter_execution_results(execution_batches):
            if result.exit_code != 0:
                continue
            if "--target" not in result.command_parts or "help" not in result.command_parts:
                continue
            for raw_line in result.get_combined_output().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("The following") or line.startswith("..."):
                    continue
                if re.fullmatch(r"[A-Za-z0-9_.:+-]+", line) is None:
                    continue
                if line in seen_targets:
                    continue
                seen_targets.add(line)
                targets.append(line)
                if len(targets) >= 12:
                    break
        if not targets:
            build_directory = self._extract_build_directory(execution_batches)
            if build_directory is not None:
                targets.extend(self.build_artifact_inspector.list_build_targets(build_directory)[:20])
        if not targets:
            return ""
        return "Available build targets: " + ", ".join(targets) + "."

    def _build_test_list_summary(self, execution_batches: list[ExecutionBatch]) -> str:
        """Summarize tests discovered by `ctest -N`."""
        test_names = []
        seen_names = set()
        for result in self._iter_execution_results(execution_batches):
            if result.exit_code != 0:
                continue
            if result.command_parts[0] != "ctest" or "-N" not in result.command_parts:
                continue
            for raw_line in result.get_combined_output().splitlines():
                match = re.search(r"Test\s+#\d+:\s+(.+)", raw_line)
                if match is None:
                    continue
                test_name = match.group(1).strip()
                if test_name in seen_names:
                    continue
                seen_names.add(test_name)
                test_names.append(test_name)
                if len(test_names) >= 12:
                    break
        if not test_names:
            build_directory = self._extract_build_directory(execution_batches)
            if build_directory is not None:
                test_names.extend(self.build_artifact_inspector.list_ctest_tests(build_directory)[:20])
        if not test_names:
            return ""
        return "Discovered tests: " + ", ".join(test_names) + "."

    def _build_configure_only_summary(self, execution_batches: list[ExecutionBatch]) -> str:
        """Summarize a successful configure-only execution path."""
        for result in self._iter_execution_results(execution_batches):
            if result.exit_code != 0 or result.command_parts[0] != "cmake" or "-S" not in result.command_parts:
                continue
            enabled_flags = []
            for option_name in ("JSON_Valgrind", "JSON_FastTests", "JSON_Diagnostics", "JSON_Diagnostic_Positions"):
                flag = f"-D{option_name}=ON"
                if flag in result.command_parts:
                    enabled_flags.append(f"{option_name}=ON")
            if enabled_flags:
                return "CMake configure completed successfully with " + ", ".join(enabled_flags) + "."
            return "CMake configure completed successfully."
        return ""

    def _extract_build_directory(self, execution_batches: list[ExecutionBatch]) -> Path | None:
        """Extract the configured build directory from prior commands."""
        for result in self._iter_execution_results(execution_batches):
            if result.command_parts[0] == "cmake" and "-B" in result.command_parts:
                option_index = result.command_parts.index("-B")
                if option_index + 1 < len(result.command_parts):
                    return Path(result.command_parts[option_index + 1])
            if result.command_parts[0] == "cmake" and "--build" in result.command_parts:
                option_index = result.command_parts.index("--build")
                if option_index + 1 < len(result.command_parts):
                    return Path(result.command_parts[option_index + 1])
        return None

    def _build_search_summary(self, retrieval_batch: RetrievalBatch) -> str:
        """Summarize literal text or scoped search results."""
        unique_matches = []
        seen_paths = set()
        for candidate in retrieval_batch.candidates:
            if candidate.file_path in seen_paths:
                continue
            seen_paths.add(candidate.file_path)
            unique_matches.append(candidate)
        scope_text = ""
        if retrieval_batch.scope_prefixes:
            scope_text = f" under {', '.join(retrieval_batch.scope_prefixes)}"
        if retrieval_batch.literal_text:
            match_intro = f"Found {len(unique_matches)} matching file(s){scope_text} containing `{retrieval_batch.literal_text}`"
        else:
            match_intro = f"Found {len(unique_matches)} matching file(s){scope_text}"
        if not unique_matches:
            if retrieval_batch.literal_text:
                return f"No matches were found{scope_text} containing `{retrieval_batch.literal_text}`."
            return f"No matches were found{scope_text}."
        locations = ", ".join(candidate.chunk.get_location_text() for candidate in unique_matches[:5])
        return f"{match_intro}: {locations}."

    def _build_file_responsibility_summary(self, retrieval_batch: RetrievalBatch) -> str:
        """Summarize the top files for conceptual file-selection questions."""
        lines = []
        seen_paths = set()
        prioritized_candidates = self._prioritize_core_files(retrieval_batch.candidates)
        for candidate in prioritized_candidates:
            if candidate.file_path in seen_paths:
                continue
            seen_paths.add(candidate.file_path)
            lines.append(f"{candidate.file_path} ({self._describe_file_role(candidate.file_path)})")
            if len(lines) >= 5:
                break
        if not lines:
            return "I could not identify grounded file candidates for that repository concept."
        return "Primary files: " + "; ".join(lines) + "."

    def _build_symbol_summary(self, query_text: str, retrieval_batch: RetrievalBatch) -> str:
        """Summarize a symbol using its definition chunk and nearby comments."""
        top_candidate = retrieval_batch.candidates[0]
        symbol_name = self._extract_primary_symbol(query_text, top_candidate)
        location_text = top_candidate.chunk.get_location_text()
        doc_summary = self._extract_doc_summary(top_candidate.chunk.content)
        capability_summary = self._extract_capability_summary(top_candidate.chunk.content)

        sentences = [f"`{symbol_name}` is defined in {location_text}."]
        if doc_summary:
            sentences.append(doc_summary)
        if capability_summary:
            sentences.append(capability_summary)
        elif top_candidate.chunk.symbols:
            sentences.append(f"The retrieved region contains symbols: {', '.join(top_candidate.chunk.symbols[:6])}.")
        return " ".join(sentences)

    def _build_file_summary(self, query_text: str, retrieval_batch: RetrievalBatch) -> str:
        """Summarize one concrete file instead of treating the query like a symbol lookup."""
        target_file_name = self._extract_requested_file_name(query_text)
        candidate = self._select_file_candidate(target_file_name, retrieval_batch)
        location_text = candidate.chunk.get_location_text()
        doc_summary = self._extract_doc_summary(candidate.chunk.content)

        sentences = [f"`{candidate.file_path}` is the matched file, grounded by {location_text}."]
        if doc_summary:
            sentences.append(doc_summary)
        capability_summary = self._extract_file_capability_summary(candidate.file_path, candidate.chunk.content)
        if capability_summary:
            sentences.append(capability_summary)
        elif candidate.chunk.symbols:
            sentences.append(f"Key symbols in the retrieved file region include {', '.join(candidate.chunk.symbols[:6])}.")
        return " ".join(sentences)

    def _extract_primary_symbol(self, query_text: str, candidate) -> str:
        """Choose the most likely symbol name referenced by the query."""
        lowered_query = query_text.lower()
        for symbol_name in candidate.chunk.symbols:
            if symbol_name.lower() in lowered_query:
                return symbol_name

        # Extract code-like tokens so generic prose words do not become the symbol guess.
        code_like_terms = re.findall(r"[A-Za-z_][A-Za-z0-9_:]*", query_text)
        ignored_terms = {"what", "where", "why", "how", "does", "do", "did", "the", "class", "function", "defined"}
        for term in code_like_terms:
            lowered_term = term.lower()
            if lowered_term in ignored_terms:
                continue
            if "_" in term or "::" in term or (any(character.isupper() for character in term) and len(term) >= 5):
                return term
        if candidate.chunk.symbols:
            return candidate.chunk.symbols[0]
        return candidate.file_path.rsplit("/", 1)[-1]

    def _extract_doc_summary(self, content: str) -> str:
        """Extract a short doc-comment summary from the retrieved chunk."""
        comment_lines = []
        for line in content.splitlines():
            stripped_line = line.strip()
            if stripped_line.startswith(("///", "//", "/*", "*")):
                cleaned_line = stripped_line.lstrip("/").lstrip("*").strip()
                cleaned_line = cleaned_line.replace("@brief", "").strip()
                if cleaned_line and not cleaned_line.startswith("@sa"):
                    comment_lines.append(cleaned_line)
                    if len(comment_lines) >= 2:
                        break
                continue
            if comment_lines:
                break
        if not comment_lines:
            return ""
        return " ".join(comment_lines)

    def _extract_requested_file_name(self, query_text: str) -> str:
        """Return the file name explicitly mentioned in the query, if any."""
        match = self.file_name_pattern.search(query_text)
        if match is None:
            return ""
        return Path(match.group(0)).name.lower()

    def _select_file_candidate(self, target_file_name: str, retrieval_batch: RetrievalBatch):
        """Prefer the candidate whose file name matches the file named by the user."""
        if not target_file_name:
            return retrieval_batch.candidates[0]
        for candidate in retrieval_batch.candidates:
            if Path(candidate.file_path).name.lower() == target_file_name:
                return candidate
        return retrieval_batch.candidates[0]

    def _extract_file_capability_summary(self, file_path: str, content: str) -> str:
        """Infer one concise file-level summary from the retrieved content."""
        lowered_path = file_path.lower()
        lowered_content = content.lower()
        if lowered_path.endswith("json.hpp"):
            return (
                "It is the main public header for the library: it pulls together many internal headers and "
                "defines the `basic_json` class template that represents JSON values and exposes the public API."
            )
        if "#include" in lowered_content and "class " in lowered_content:
            return "It combines included dependencies with core type or class definitions used by the library."
        if "#include" in lowered_content:
            return "It primarily acts as a header that wires together related library components."
        return ""

    def _extract_capability_summary(self, content: str) -> str:
        """Infer a concise capability summary from known member names."""
        capabilities = []
        lowered_content = content.lower()
        if "reference_tokens" in lowered_content:
            capabilities.append("It stores the pointer as reference tokens.")
        if "to_string(" in lowered_content:
            capabilities.append("It can convert the pointer back into JSON Pointer string form.")
        if "push_back(" in lowered_content or "operator/=" in lowered_content:
            capabilities.append("It supports appending tokens or array indices to build longer paths.")
        if "parent_pointer(" in lowered_content or "pop_back(" in lowered_content:
            capabilities.append("It can move to or remove the parent segment.")
        if "back(" in lowered_content or "empty(" in lowered_content:
            capabilities.append("It exposes simple inspection helpers for the last token and root state.")
        if "array_index(" in lowered_content or "split(" in lowered_content:
            capabilities.append("It also parses and validates pointer tokens when navigating arrays.")
        return " ".join(capabilities[:4])

    def _is_file_responsibility_query(self, query_text: str) -> bool:
        """Return True for questions asking which files implement a concept."""
        lowered_query = query_text.lower()
        return "which files" in lowered_query or "what files" in lowered_query or "responsible" in lowered_query

    def _is_file_summary_query(self, query_text: str) -> bool:
        """Return True when the user is asking what one named file is about."""
        lowered_query = query_text.lower()
        if "file" not in lowered_query:
            return False
        return self._extract_requested_file_name(query_text) != ""

    def _describe_file_role(self, file_path: str) -> str:
        """Describe the likely responsibility of a retrieved file."""
        lowered_path = file_path.lower()
        if "detail/output/serializer" in lowered_path:
            return "core serializer implementation"
        if "detail/conversions/to_json" in lowered_path:
            return "C++ to JSON conversion logic"
        if "detail/conversions/from_json" in lowered_path:
            return "JSON to C++ conversion logic"
        if lowered_path.endswith("adl_serializer.hpp"):
            return "ADL customization hook for user-defined types"
        if lowered_path.endswith("json.hpp"):
            return "public header wiring serialization into basic_json"
        return "retrieved implementation candidate"

    def _prioritize_core_files(self, candidates) -> list:
        """Prefer library implementation files over docs and tests for conceptual summaries."""
        core_candidates = []
        fallback_candidates = []
        for candidate in candidates:
            lowered_path = candidate.file_path.lower()
            if lowered_path.startswith("include/") or lowered_path.startswith("src/") or lowered_path.startswith("single_include/"):
                core_candidates.append(candidate)
                continue
            fallback_candidates.append(candidate)
        return core_candidates + fallback_candidates

    def _find_missing_command_name(
        self,
        execution_batches: list[ExecutionBatch],
        analysis_report: AnalysisReport | None,
    ) -> str:
        """Return the missing executable name when execution failed before build/test logic ran."""
        for execution_batch in execution_batches:
            for result in execution_batch.results:
                if result.exit_code != 127:
                    continue
                if result.command_parts:
                    return result.command_parts[0]
        if analysis_report is None:
            return ""
        lowered_error = analysis_report.first_reported_error.lower()
        if not any(marker in lowered_error for marker in self.command_unavailable_markers):
            return ""
        for command_name in ("cmake", "ctest", "make", "ninja"):
            if command_name in lowered_error:
                return command_name
        return ""

    def _iter_execution_results(self, execution_batches: list[ExecutionBatch]):
        """Yield every execution result in chronological order."""
        for execution_batch in execution_batches:
            for result in execution_batch.results:
                yield result
