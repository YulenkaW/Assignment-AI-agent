from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from assignment_agent.agent_controller import AssignmentAgentController
from assignment_agent.build_test_analyzer import BuildTestAnalyzer
from assignment_agent.command_executor import CommandExecutor
from assignment_agent.context_manager import ContextManager
from assignment_agent.contracts import CapturedCommandOutput, EvidenceRequirements, QueryRequest, RetrievalBudget, RetrievalPlan, RetrievalStep
from assignment_agent.error_accumulator import ErrorAccumulator
from assignment_agent.output_parser import OutputParser
from assignment_agent.planning_layer import PlanningLayer
from assignment_agent.repository_index import RepositoryIndex
from assignment_agent.repository_service import RepositoryService
from assignment_agent.retrieval_planner import RetrievalPlanner
from assignment_agent.task_router import TaskRouter
from assignment_agent.workspace_paths import WorkspacePaths


FIXTURE = Path(__file__).parent / "fixtures" / "mini_cpp"


def _always_missing_executable(command_name: str) -> None:
    return None


def _always_true() -> bool:
    return True


def _classify_as_build_test_debug(query_text: str) -> str:
    return "build_test_debug"


def _make_retrieval_plan(
    operator_name: str,
    query_text: str,
    limit: int = 3,
    literal_text: str = "",
    scope_prefixes: list[str] | None = None,
) -> RetrievalPlan:
    return RetrievalPlan(
        "definition_lookup",
        query_text,
        RetrievalBudget(2000, limit, 1, 0),
        EvidenceRequirements("test evidence", ["chunk"]),
        [RetrievalStep(operator_name, "test step")],
        literal_text=literal_text,
        scope_prefixes=scope_prefixes,
    )


def test_router_marks_mixed_queries_explicitly() -> None:
    router = TaskRouter()
    decision = router.route(QueryRequest(str(FIXTURE), "Explain json_pointer and run the tests"))
    assert decision.task_type == "mixed"
    assert decision.needs_retrieval is True
    assert decision.needs_execution is True
    assert decision.preferred_flow == "execution_first"


def test_router_treats_listing_tests_as_execution() -> None:
    router = TaskRouter()
    decision = router.route(QueryRequest(str(FIXTURE), "List tests for this project"))
    assert decision.task_type == "build_test_debug"
    assert decision.needs_execution is True


def test_router_does_not_misclassify_backticked_symbol_questions_as_execution() -> None:
    router = TaskRouter()
    decision = router.route(QueryRequest(str(FIXTURE), "Explain `json_pointer` and where it is defined"))
    assert decision.task_type == "code_understanding"
    assert decision.needs_execution is False


def test_router_treats_tool_driven_configure_and_make_requests_as_execution() -> None:
    router = TaskRouter()
    decision = router.route(QueryRequest(str(FIXTURE), "configure with cmake and then run make"))
    assert decision.task_type == "build_test_debug"
    assert decision.needs_execution is True


def test_router_treats_build_tests_as_execution() -> None:
    router = TaskRouter()
    decision = router.route(QueryRequest(str(FIXTURE), "build tests"))
    assert decision.task_type == "build_test_debug"
    assert decision.needs_execution is True


def test_router_treats_named_build_targets_as_execution() -> None:
    router = TaskRouter()
    decision = router.route(QueryRequest(str(FIXTURE), "run download_test_data"))
    assert decision.task_type == "build_test_debug"
    assert decision.needs_execution is True


def test_router_treats_build_with_ninja_as_execution() -> None:
    router = TaskRouter()
    decision = router.route(QueryRequest(str(FIXTURE), "Build with ninja"))
    assert decision.task_type == "build_test_debug"
    assert decision.needs_execution is True


def test_router_treats_named_makefiles_as_execution() -> None:
    router = TaskRouter()
    decision = router.route(QueryRequest(str(FIXTURE), "run ci.make file"))
    assert decision.task_type == "build_test_debug"
    assert decision.needs_execution is True


def test_router_uses_model_fallback_for_ambiguous_action_queries(monkeypatch) -> None:
    router = TaskRouter()
    monkeypatch.setattr(router, "_can_call_model", _always_true)
    monkeypatch.setattr(router, "_classify_with_model", _classify_as_build_test_debug)
    decision = router.route(QueryRequest(str(FIXTURE), "run the ci pipeline"))
    assert decision.task_type == "build_test_debug"
    assert decision.needs_execution is True


def test_planning_layer_includes_reasoning_and_response_steps() -> None:
    router = TaskRouter()
    planner = PlanningLayer()
    request = QueryRequest(str(FIXTURE), "What does the json_pointer class do?")
    decision = router.route(request)
    plan = planner.create_plan(request, decision)
    step_names = [step.name for step in plan.steps]
    assert "reasoning" in step_names
    assert "response" in step_names


def test_understanding_query_returns_grounded_file_reference() -> None:
    controller = AssignmentAgentController(FIXTURE)
    response = controller.answer_query("What does the json_pointer class do and where is it defined?")
    assert response.task_type == "code_understanding"
    assert "json_pointer" in response.answer_text
    assert "json_pointer.hpp" in response.answer_text
    assert response.prompt_report is not None


def test_build_flow_creates_build_execution_batch() -> None:
    controller = AssignmentAgentController(FIXTURE)
    response = controller.answer_query("Build the project and run the tests.")
    assert response.task_type in ("build_test_debug", "mixed")
    assert response.execution_batches
    assert response.execution_batches[0].phase_name == "build"
    assert response.execution_batches[0].results
    assert response.execution_batches[0].results[0].get_command_text().startswith("cmake") or response.execution_batches[0].results[0].get_command_text().startswith("make")


def test_identifier_first_retrieval_prefers_exact_path_match() -> None:
    repository_index = RepositoryIndex(FIXTURE)
    repository_index.build()
    service = RepositoryService(repository_index)
    batch = service.retrieve(_make_retrieval_plan("path", "What does the json_pointer class do and where is it defined?"))
    assert batch.candidates
    assert batch.candidates[0].file_path.endswith("json_pointer.hpp")
    assert "identifier" in batch.candidates[0].reason


def test_retrieval_planner_builds_definition_plan_from_task_kind() -> None:
    planner = RetrievalPlanner()
    decision = TaskRouter().route(QueryRequest(str(FIXTURE), "What does the json_pointer class do and where is it defined?"))
    retrieval_plan = planner.create_plan("What does the json_pointer class do and where is it defined?", decision, 2000)
    assert retrieval_plan.task_kind == "definition_lookup"
    assert [step.operator_name for step in retrieval_plan.retrieval_steps][:2] == ["symbol", "path"]


def test_retrieval_planner_builds_literal_search_plan_for_directed_search() -> None:
    planner = RetrievalPlanner()
    decision = TaskRouter().route(QueryRequest(str(FIXTURE), "Which files under tests contain 'sbor'?"))
    retrieval_plan = planner.create_plan("Which files under tests contain 'sbor'?", decision, 2000)
    assert retrieval_plan.task_kind == "literal_search"
    assert retrieval_plan.literal_text == "sbor"
    assert retrieval_plan.scope_prefixes == ["tests/"]
    assert [step.operator_name for step in retrieval_plan.retrieval_steps] == ["literal_text", "keyword"]


def test_retrieval_planner_builds_execution_guided_failure_plan() -> None:
    planner = RetrievalPlanner()
    decision = TaskRouter().route(QueryRequest(str(FIXTURE), "Why is the build failing?"))
    analysis_report = BuildTestAnalyzer().analyze(
        OutputParser().parse_outputs(
            [
                CapturedCommandOutput(
                    "cmake --build build",
                    1,
                    "",
                    "include/json_pointer.hpp:6:12: error: expected ';' after class definition",
                    "include/json_pointer.hpp:6:12: error: expected ';' after class definition",
                )
            ]
        )
    )
    retrieval_plan = planner.create_plan("Why is the build failing?", decision, 2000, analysis_report)
    assert retrieval_plan.task_kind == "build_failure_analysis"
    assert retrieval_plan.preferred_files[0].endswith("json_pointer.hpp")
    assert retrieval_plan.retrieval_steps[0].operator_name == "execution_guided"


def test_workspace_paths_keep_build_outputs_outside_repo() -> None:
    workspace_paths = WorkspacePaths(FIXTURE, PROJECT_ROOT)
    build_directory = workspace_paths.get_build_directory()
    assert build_directory.exists()
    assert str(build_directory).startswith(str(PROJECT_ROOT))
    assert not str(build_directory).startswith(str(FIXTURE / "build"))


def test_controller_anchors_workspace_artifacts_to_project_root(monkeypatch) -> None:
    import shutil
    import uuid

    temporary_cwd = PROJECT_ROOT / ".test_memory" / str(uuid.uuid4()) / "cwd"
    temporary_cwd.mkdir(parents=True)
    try:
        monkeypatch.chdir(temporary_cwd)
        controller = AssignmentAgentController(FIXTURE)
        build_directory = controller.workspace_paths.get_build_directory()
        assert str(build_directory).startswith(str(PROJECT_ROOT))
        assert ".assignment_agent_work" in str(build_directory)
    finally:
        shutil.rmtree(temporary_cwd.parent.parent, ignore_errors=True)


def test_command_executor_records_missing_command_error() -> None:
    errors = ErrorAccumulator()
    executor = CommandExecutor(FIXTURE, errors)
    result = executor.run(["git", "reset", "--hard"], "build")
    assert result.exit_code == 126
    assert errors.get_records()
    assert "blocked" in errors.get_records()[0].summary.lower()


def test_command_executor_resolves_windows_fallback_locations(monkeypatch) -> None:
    fallback_root = PROJECT_ROOT / ".test_memory" / "fake_cmake_bin"
    fallback_root.mkdir(parents=True, exist_ok=True)
    fake_cmake = fallback_root / "cmake.exe"
    fake_cmake.write_text("", encoding="utf-8")
    monkeypatch.setattr(CommandExecutor, "WINDOWS_FALLBACK_DIRECTORIES", (fallback_root,))
    monkeypatch.setattr("assignment_agent.command_executor.shutil.which", _always_missing_executable)
    monkeypatch.setattr("assignment_agent.command_executor.os.name", "nt")
    try:
        resolved_path = CommandExecutor.resolve_command_path("cmake")
        assert resolved_path == str(fake_cmake)
    finally:
        fake_cmake.unlink(missing_ok=True)
        fallback_root.rmdir()


def test_output_parser_and_analyzer_prefer_file_backed_root_cause() -> None:
    errors = ErrorAccumulator()
    parser = OutputParser(errors)
    analyzer = BuildTestAnalyzer(errors)
    parsed_outputs = parser.parse_outputs(
        [
            CapturedCommandOutput(
                "cmake --build .assignment-build",
                1,
                "",
                "include/json_pointer.hpp:6:12: error: expected ';' after class definition",
                "include/json_pointer.hpp:6:12: error: expected ';' after class definition",
            )
        ]
    )
    analysis = analyzer.analyze(parsed_outputs)
    assert analysis.recommended_next_action == "retrieve_more_context"
    assert analysis.relevant_files[0].endswith("json_pointer.hpp")
    assert analysis.line_numbers[0] == 6


def test_output_parser_records_missing_command_error() -> None:
    errors = ErrorAccumulator()
    parser = OutputParser(errors)
    parsed_outputs = parser.parse_outputs(
        [
            CapturedCommandOutput(
                "ctest --test-dir build",
                127,
                "",
                "[WinError 2] The system cannot find the file specified",
                "[WinError 2] The system cannot find the file specified",
            )
        ]
    )
    assert parsed_outputs[0].missing_command is True
    assert errors.get_records()
    assert "unavailable" in errors.get_records()[0].summary.lower()


def test_analyzer_reports_timeout_without_claiming_success() -> None:
    analysis = BuildTestAnalyzer().analyze(
        OutputParser().parse_outputs(
        [
            CapturedCommandOutput(
                "cmake --build build --target download_test_data",
                124,
                "",
                "Command '['cmake', '--build', 'build', '--target', 'download_test_data']' timed out after 120 seconds",
                "Command '['cmake', '--build', 'build', '--target', 'download_test_data']' timed out after 120 seconds",
            )
        ]
        )
    )
    assert analysis.root_cause_candidates
    assert "timed out" in analysis.root_cause_candidates[0].summary_text.lower()
    assert analysis.recommended_next_action == "answer_with_limited_evidence"


def test_context_manager_enforces_budget_and_reports_decisions() -> None:
    controller = AssignmentAgentController(FIXTURE, max_total_tokens=1200)
    response = controller.answer_query("What does the json_pointer class do and where is it defined?")
    assert response.prompt_report is not None
    assert response.prompt_report.total_tokens <= 1200
    assert response.prompt_report.decisions


def test_controller_records_disallowed_model_action_as_warning() -> None:
    controller = AssignmentAgentController(FIXTURE)
    chosen_action = controller._validate_action("run_shell", ["retrieve_context"])
    assert chosen_action == "retrieve_context"
    assert controller.error_accumulator.get_records()
    assert "disallowed action" in controller.error_accumulator.get_records()[0].summary.lower()


def test_search_query_under_tests_returns_matches_without_execution() -> None:
    import shutil
    import uuid

    repository_root = PROJECT_ROOT / ".test_memory" / str(uuid.uuid4()) / "repo"
    tests_directory = repository_root / "tests"
    tests_directory.mkdir(parents=True)
    (tests_directory / "alpha.txt").write_text("contains sbor token\n", encoding="utf-8")
    (tests_directory / "beta.txt").write_text("no match here\n", encoding="utf-8")

    try:
        controller = AssignmentAgentController(repository_root)
        response = controller.answer_query("Which files under tests contain 'sbor'?")

        assert response.task_type == "code_understanding"
        assert response.route_decision is not None
        assert response.route_decision.query_mode == "search"
        assert not response.execution_batches
        assert "tests/alpha.txt" in response.answer_text
    finally:
        shutil.rmtree(repository_root.parent, ignore_errors=True)


def test_controller_returns_diagnostics_for_ui_tabs() -> None:
    controller = AssignmentAgentController(FIXTURE)
    response = controller.answer_query("What does the json_pointer class do and where is it defined?")

    assert response.diagnostics is not None
    assert response.diagnostics.processing_time_ms >= 0
    assert response.diagnostics.route_task_type == "code_understanding"
    assert response.diagnostics.retrieval_task_kind == "definition_lookup"
    assert response.diagnostics.selected_chunks >= 1
    assert response.external_memory_records is not None
