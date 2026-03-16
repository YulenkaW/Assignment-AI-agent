from pathlib import Path
import os
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from assignment_agent.build_artifact_inspector import BuildArtifactInspector
from assignment_agent.build_runner import BuildRunner
from assignment_agent.command_line import AssignmentAgentCli
from assignment_agent.command_output_capture import CommandOutputCapture
from assignment_agent.command_safety import CommandSafetyLevel, CommandSafetyPolicy
from assignment_agent.context_manager import ContextManager
from assignment_agent.contracts import (
    AnalysisReport,
    CapturedCommandOutput,
    CommandExecutionResult,
    ExecutionBatch,
    RepositoryChunk,
    RetrievalBatch,
    RetrievalCandidate,
    RootCauseCandidate,
    RouteDecision,
)
from assignment_agent.env_loader import load_project_env
from assignment_agent.error_accumulator import ErrorAccumulator
from assignment_agent.external_memory_store import ExternalMemoryStore
from assignment_agent.output_parser import OutputParser
from assignment_agent.reasoning_engine import ReasoningEngine
from assignment_agent.response_generator import ResponseGenerator
from assignment_agent.stop_retry_controller import StopRetryController
from assignment_agent.test_runner import TestRunner as AgentTestRunner
from assignment_agent.workspace_paths import WorkspacePaths


FIXTURE = Path(__file__).parent / "fixtures" / "mini_cpp"


def _make_candidate(file_path: str, content: str = "class json_pointer {};") -> RetrievalCandidate:
    chunk = RepositoryChunk(file_path, 10, 20, "structure", file_path, content, ["json_pointer"], 20)
    return RetrievalCandidate(file_path, chunk.get_location_text(), 120, "exact identifier filename match", chunk)


def _always_true() -> bool:
    return True


def _raise_runtime_error(*args, **kwargs):
    raise RuntimeError("blocked")


def test_build_runner_prefers_cmake_commands() -> None:
    runner = BuildRunner(FIXTURE, WorkspacePaths(FIXTURE, PROJECT_ROOT), ErrorAccumulator())
    commands = runner._build_commands()
    assert commands[0][0] == "cmake"
    assert "--build" in commands[1]
    assert "--parallel" in commands[1]


def test_command_safety_policy_allows_cmake_build_artifact_writes() -> None:
    build_directory = WorkspacePaths(FIXTURE, PROJECT_ROOT).get_build_directory()
    policy = CommandSafetyPolicy(FIXTURE, [build_directory])
    decision = policy.validate(
        [
            "cmake",
            "-S",
            str(FIXTURE),
            "-B",
            str(build_directory),
            "-DJSON_BuildTests=ON",
        ]
    )
    assert decision.allowed is True
    assert decision.safety_level == CommandSafetyLevel.BUILD_ARTIFACT_WRITE


def test_command_safety_policy_treats_test_listing_as_read_only() -> None:
    build_directory = WorkspacePaths(FIXTURE, PROJECT_ROOT).get_build_directory()
    policy = CommandSafetyPolicy(FIXTURE, [build_directory])
    decision = policy.validate(["ctest", "--test-dir", str(build_directory), "-N"])
    assert decision.allowed is True
    assert decision.safety_level == CommandSafetyLevel.READ_ONLY


def test_command_safety_policy_blocks_writes_into_source_tree() -> None:
    build_directory = WorkspacePaths(FIXTURE, PROJECT_ROOT).get_build_directory()
    policy = CommandSafetyPolicy(FIXTURE, [build_directory])
    decision = policy.validate(["cmake", "-S", str(FIXTURE), "-B", str(FIXTURE / "include")])
    assert decision.allowed is False
    assert decision.safety_level == CommandSafetyLevel.SOURCE_MODIFY


def test_command_safety_policy_allows_repo_makefiles() -> None:
    build_directory = WorkspacePaths(FIXTURE, PROJECT_ROOT).get_build_directory()
    policy = CommandSafetyPolicy(FIXTURE, [build_directory])
    decision = policy.validate(["make", "-f", str(FIXTURE / "ci.make")])
    assert decision.allowed is True
    assert decision.safety_level == CommandSafetyLevel.BUILD_ARTIFACT_WRITE


def test_build_runner_supports_explicit_make_backend_requests(monkeypatch) -> None:
    runner = BuildRunner(FIXTURE, WorkspacePaths(FIXTURE, PROJECT_ROOT), ErrorAccumulator())
    commands = runner._build_commands("configure with cmake and then run make")
    assert commands[0][:2] == ["cmake", "-S"]
    assert "Unix Makefiles" in commands[0]
    assert commands[1][0] == "make"


def test_build_runner_can_run_named_makefiles_directly() -> None:
    runner = BuildRunner(FIXTURE, WorkspacePaths(FIXTURE, PROJECT_ROOT), ErrorAccumulator())
    commands = runner._build_commands("run ci.make file")
    assert commands[0][:2] == ["make", "-f"]
    assert commands[0][2].endswith("ci.make")


def test_build_runner_can_run_named_cmake_targets() -> None:
    runner = BuildRunner(FIXTURE, WorkspacePaths(FIXTURE, PROJECT_ROOT), ErrorAccumulator())
    commands = runner._build_commands("run download_test_data")
    assert commands[0][0] == "cmake"
    assert commands[1][:4] == ["cmake", "--build", str(WorkspacePaths(FIXTURE, PROJECT_ROOT).get_build_directory()), "--target"]
    assert "download_test_data" in commands[1]
    assert "--parallel" not in commands[1]


def test_build_runner_can_list_cmake_targets() -> None:
    runner = BuildRunner(FIXTURE, WorkspacePaths(FIXTURE, PROJECT_ROOT), ErrorAccumulator())
    commands = runner._build_commands("Show build targets")
    assert len(commands) == 1
    assert commands[0][0] == "cmake"
    assert "-S" in commands[0]


def test_build_runner_can_show_cmake_options() -> None:
    runner = BuildRunner(FIXTURE, WorkspacePaths(FIXTURE, PROJECT_ROOT), ErrorAccumulator())
    commands = runner._build_commands("Show CMake options")
    assert commands[0][0] == "cmake"
    assert commands[1][:3] == ["cmake", "-LAH", "-N"]


def test_build_runner_supports_configure_only_valgrind_requests() -> None:
    runner = BuildRunner(FIXTURE, WorkspacePaths(FIXTURE, PROJECT_ROOT), ErrorAccumulator())
    commands = runner._build_commands("Run the cmake for json valgrind")
    assert len(commands) == 1
    assert "-DJSON_Valgrind=ON" in commands[0]


def test_build_runner_treats_plain_configure_as_configure_only() -> None:
    runner = BuildRunner(FIXTURE, WorkspacePaths(FIXTURE, PROJECT_ROOT), ErrorAccumulator())
    commands = runner._build_commands("Configure with cmake")
    assert len(commands) == 1
    assert commands[0][0] == "cmake"
    assert "-S" in commands[0]


def test_build_runner_uses_configure_only_for_list_tests_requests() -> None:
    runner = BuildRunner(FIXTURE, WorkspacePaths(FIXTURE, PROJECT_ROOT), ErrorAccumulator())
    commands = runner._build_commands("List tests")
    assert len(commands) == 1
    assert commands[0][0] == "cmake"
    assert "-S" in commands[0]


def test_test_runner_prefers_ctest_commands() -> None:
    runner = AgentTestRunner(FIXTURE, WorkspacePaths(FIXTURE, PROJECT_ROOT), ErrorAccumulator())
    commands = runner._build_commands()
    assert commands[0][0] == "ctest"
    assert "--test-dir" in commands[0]
    assert "--output-on-failure" in commands[0]


def test_test_runner_supports_explicit_make_test_requests(monkeypatch) -> None:
    runner = AgentTestRunner(FIXTURE, WorkspacePaths(FIXTURE, PROJECT_ROOT), ErrorAccumulator())
    monkeypatch.setattr(runner, "_has_make", _always_true)
    commands = runner._build_commands("run make test")
    assert commands[0][0] == "make"
    assert commands[0][-1] == "test"


def test_build_runner_supports_explicit_ninja_backend_requests(monkeypatch) -> None:
    runner = BuildRunner(FIXTURE, WorkspacePaths(FIXTURE, PROJECT_ROOT), ErrorAccumulator())
    commands = runner._build_commands("Build with ninja")
    assert "Ninja" in commands[0]
    assert commands[1][0] == "ninja"


def test_test_runner_can_list_tests() -> None:
    runner = AgentTestRunner(FIXTURE, WorkspacePaths(FIXTURE, PROJECT_ROOT), ErrorAccumulator())
    commands = runner._build_commands("List tests")
    assert commands[0][-1] == "-N"


def test_test_runner_can_filter_tests_by_query() -> None:
    runner = AgentTestRunner(FIXTURE, WorkspacePaths(FIXTURE, PROJECT_ROOT), ErrorAccumulator())
    commands = runner._build_commands("Run only test parser_case")
    assert commands[0][:3] == ["ctest", "--test-dir", str(WorkspacePaths(FIXTURE, PROJECT_ROOT).get_build_directory())]
    assert "-R" in commands[0]
    assert "parser_case" in commands[0]


def test_output_parser_ignores_successful_cmake_help_text_with_error_words() -> None:
    parser = OutputParser()
    parsed_outputs = parser.parse_outputs(
        [
            CapturedCommandOutput(
                "cmake -LAH -N build",
                0,
                "// Path to the memory checking command, used for memory error detection.\nBUILD_TESTING:BOOL=ON",
                "",
                "// Path to the memory checking command, used for memory error detection.\nBUILD_TESTING:BOOL=ON",
            )
        ]
    )
    assert parsed_outputs[0].error_lines == []


def test_command_output_capture_normalizes_batch() -> None:
    batch = ExecutionBatch("build", [CommandExecutionResult(["cmake", "--build", "build"], 1, "out", "err", "build")])
    outputs = CommandOutputCapture().capture_batch(batch)
    assert len(outputs) == 1
    assert outputs[0].combined_output == "out\nerr"


def test_external_memory_store_can_seed_remember_and_search() -> None:
    import shutil
    import uuid

    storage = PROJECT_ROOT / ".test_memory" / str(uuid.uuid4())

    class FakeIndexedFile:
        def __init__(self, file_path: str, summary: str) -> None:
            self.file_path = file_path
            self.summary = summary

    class FakeIndex:
        def __init__(self) -> None:
            self.files = {
                "include/json_pointer.hpp": FakeIndexedFile("include/json_pointer.hpp", "json pointer header"),
                "src/main.cpp": FakeIndexedFile("src/main.cpp", "main entry"),
            }

    store = ExternalMemoryStore(storage, FIXTURE)
    store.seed_file_summaries(FakeIndex())
    store.remember_note("note:json_pointer", "json_pointer is defined in a header", "include/json_pointer.hpp")
    matches = store.find_relevant_records("where is json_pointer defined", limit=3)
    assert matches
    assert any("json_pointer" in record.summary_text for record in matches)
    shutil.rmtree(storage.parent, ignore_errors=True)


def test_external_memory_store_prefers_core_files_for_concept_queries() -> None:
    import shutil

    storage = PROJECT_ROOT / ".test_memory" / str(uuid.uuid4())
    store = ExternalMemoryStore(storage, FIXTURE)
    try:
        store.records = []
        store.remember_note(
            "core:serializer",
            "JSON serialization output serializer dump to_json from_json",
            "include/nlohmann/detail/output/serializer.hpp",
        )
        store.remember_note(
            "thirdparty:fuzzer",
            "Fuzzer util for Windows",
            "tests/thirdparty/Fuzzer/FuzzerUtilWindows.cpp",
        )
        store.remember_note(
            "license:bsd",
            "BSD license text",
            "LICENSES/BSD-3-Clause.txt",
        )

        matches = store.find_relevant_records("Which files are responsible for JSON serialization?", limit=3)

        assert matches
        assert matches[0].source_path == "include/nlohmann/detail/output/serializer.hpp"
    finally:
        shutil.rmtree(storage, ignore_errors=True)


def test_external_memory_store_requires_literal_match_for_quoted_search_queries() -> None:
    import shutil

    storage = PROJECT_ROOT / ".test_memory" / str(uuid.uuid4())
    store = ExternalMemoryStore(storage, FIXTURE)
    try:
        store.records = []
        store.remember_note("tests:alpha", "contains sbor token", "tests/alpha.txt")
        store.remember_note("tests:beta", "contains another token", "tests/beta.txt")
        store.remember_note("tests:thirdparty", "fuzzer util", "tests/thirdparty/Fuzzer/FuzzerUtilWindows.cpp")

        matches = store.find_relevant_records("Which files under tests contain 'sbor'?", limit=5)

        assert len(matches) == 1
        assert matches[0].source_path == "tests/alpha.txt"
    finally:
        shutil.rmtree(storage, ignore_errors=True)


def test_reasoning_engine_deterministic_understanding_summary() -> None:
    engine = ReasoningEngine("gpt-4.1-mini")
    batch = RetrievalBatch([_make_candidate("include/json_pointer.hpp")], "path")
    route = RouteDecision("code_understanding", 0.9, True, False, "retrieval_first", ["retrieve relevant repository evidence"])
    outcome = engine.reason("What does json_pointer do?", route, batch, [], None, None)
    assert "include/json_pointer.hpp" in outcome.summary_text
    assert outcome.evidence_lines


def test_reasoning_engine_deterministic_execution_summary() -> None:
    engine = ReasoningEngine("gpt-4.1-mini")
    analysis = AnalysisReport(
        "include/json_pointer.hpp:6: error",
        [RootCauseCandidate("Likely missing semicolon", "include/json_pointer.hpp", 6, 0.9)],
        ["include/json_pointer.hpp"],
        [6],
        "retrieve_more_context",
    )
    batch = RetrievalBatch([_make_candidate("include/json_pointer.hpp")], "execution_guided")
    execution = [ExecutionBatch("build", [CommandExecutionResult(["cmake", "--build", "build"], 1, "", "error", "build")])]
    route = RouteDecision("build_test_debug", 0.95, True, True, "execution_first", ["run build or test commands and capture deterministic output"])
    outcome = engine.reason("Why is the build failing?", route, batch, execution, analysis, None)
    assert "Likely root cause" in outcome.summary_text
    assert outcome.next_steps


def test_reasoning_engine_summarizes_successful_target_execution() -> None:
    engine = ReasoningEngine("gpt-4.1-mini")
    route = RouteDecision("build_test_debug", 0.95, True, True, "execution_first", ["run build or test commands and capture deterministic output"])
    execution = [ExecutionBatch("build", [CommandExecutionResult(["cmake", "--build", "build", "--target", "download_test_data"], 0, "", "", "build")])]

    outcome = engine.reason("run download_test_data", route, None, execution, None, None)

    assert "download_test_data" in outcome.summary_text


def test_reasoning_engine_can_list_targets_from_solution_files() -> None:
    import shutil

    storage = PROJECT_ROOT / ".test_memory" / str(uuid.uuid4())
    build_directory = storage / "build"
    build_directory.mkdir(parents=True)
    (build_directory / "sample.sln").write_text(
        'Project("{GUID}") = "ALL_BUILD", "ALL_BUILD.vcxproj", "{A}"\n'
        'Project("{GUID}") = "download_test_data", "tests\\\\download_test_data.vcxproj", "{B}"\n'
        'Project("{GUID}") = "test-parser", "tests\\\\test-parser.vcxproj", "{C}"\n',
        encoding="utf-8",
    )
    engine = ReasoningEngine("gpt-4.1-mini")
    route = RouteDecision("build_test_debug", 0.95, True, True, "execution_first", ["run build or test commands and capture deterministic output"])
    execution = [ExecutionBatch("build", [CommandExecutionResult(["cmake", "-S", "repo", "-B", str(build_directory)], 0, "", "", "build")])]
    try:
        outcome = engine.reason("Show build targets", route, None, execution, None, None)
        assert "download_test_data" in outcome.summary_text
        assert "test-parser" in outcome.summary_text
    finally:
        shutil.rmtree(storage, ignore_errors=True)


def test_build_artifact_inspector_lists_solution_targets() -> None:
    import shutil

    storage = PROJECT_ROOT / ".test_memory" / str(uuid.uuid4())
    build_directory = storage / "build"
    build_directory.mkdir(parents=True)
    (build_directory / "sample.sln").write_text(
        'Project("{GUID}") = "ZERO_CHECK", "ZERO_CHECK.vcxproj", "{A}"\n'
        'Project("{GUID}") = "download_test_data", "tests\\\\download_test_data.vcxproj", "{B}"\n'
        'Project("{GUID}") = "test-parser", "tests\\\\test-parser.vcxproj", "{C}"\n',
        encoding="utf-8",
    )
    inspector = BuildArtifactInspector()
    try:
        targets = inspector.list_build_targets(build_directory)
        assert "download_test_data" in targets
        assert "test-parser" in targets
        assert "ZERO_CHECK" not in targets
    finally:
        shutil.rmtree(storage, ignore_errors=True)


def test_reasoning_engine_summarizes_cmake_options_query() -> None:
    engine = ReasoningEngine("gpt-4.1-mini")
    route = RouteDecision("build_test_debug", 0.95, True, True, "execution_first", ["run build or test commands and capture deterministic output"])
    execution = [
        ExecutionBatch(
            "build",
            [
                CommandExecutionResult(["cmake", "-S", "repo", "-B", "build", "-DJSON_BuildTests=ON", "-DBUILD_TESTING=ON"], 0, "-- Configuring done", "", "build"),
                CommandExecutionResult(["cmake", "-LAH", "-N", "build"], 0, "BUILD_TESTING:BOOL=ON\nJSON_Valgrind:BOOL=OFF\nJSON_FastTests:BOOL=OFF", "", "build"),
            ],
        )
    ]

    outcome = engine.reason("Show CMake options", route, None, execution, None, None)

    assert "CMake cache options:" in outcome.summary_text
    assert "JSON_Valgrind=OFF" in outcome.summary_text


def test_reasoning_engine_search_summary_lists_matching_files() -> None:
    engine = ReasoningEngine("gpt-4.1-mini")
    batch = RetrievalBatch(
        [
            _make_candidate("tests/alpha.txt", "sbor match"),
            _make_candidate("tests/beta.txt", "another sbor match"),
        ],
        "literal_text",
        literal_text="sbor",
        scope_prefixes=["tests/"],
    )
    route = RouteDecision("code_understanding", 0.97, True, False, "retrieval_first", ["find matching files or text in the repository"], query_mode="search")

    outcome = engine.reason("Which files under tests contain 'sbor'?", route, batch, [], None, None)

    assert "tests/alpha.txt" in outcome.summary_text
    assert "sbor" in outcome.summary_text


def test_reasoning_engine_file_responsibility_summary_names_roles() -> None:
    engine = ReasoningEngine("gpt-4.1-mini")
    batch = RetrievalBatch(
        [
            _make_candidate("include/nlohmann/detail/output/serializer.hpp", "class serializer { void dump(); };"),
            _make_candidate("include/nlohmann/detail/conversions/to_json.hpp", "void to_json();"),
            _make_candidate("include/nlohmann/adl_serializer.hpp", "struct adl_serializer {};"),
        ],
        "keyword",
    )
    route = RouteDecision("code_understanding", 0.9, True, False, "retrieval_first", ["retrieve relevant repository evidence"])

    outcome = engine.reason("Which files are responsible for JSON serialization?", route, batch, [], None, None)

    assert "serializer.hpp" in outcome.summary_text
    assert "to_json.hpp" in outcome.summary_text
    assert "ADL" in outcome.summary_text


def test_response_generator_formats_both_modes() -> None:
    generator = ResponseGenerator()
    analysis = AnalysisReport("first error", [RootCauseCandidate("root cause", confidence=0.8)], [], [], "answer")

    understanding_text = generator.generate(
        RouteDecision("code_understanding", 0.9, True, False, "retrieval_first", []),
        type("Outcome", (), {"summary_text": "summary", "evidence_lines": ["e1"], "next_steps": ["n1"]})(),
        None,
    )
    assert "Evidence:" in understanding_text

    debug_text = generator.generate(
        RouteDecision("build_test_debug", 0.95, True, True, "execution_first", []),
        type("Outcome", (), {"summary_text": "debug summary", "evidence_lines": ["e1"], "next_steps": ["n1"]})(),
        analysis,
    )
    assert "First reported error" in debug_text


def test_stop_retry_controller_covers_stop_and_retry() -> None:
    controller = StopRetryController(max_iterations=2)
    analysis = AnalysisReport("", [RootCauseCandidate("candidate", confidence=0.7)], ["file.cpp"], [10], "retrieve_more_context")
    decision_retry = controller.decide(0, analysis, 1000)
    decision_stop = controller.decide(2, analysis, 1000)
    assert decision_retry.should_stop is False
    assert decision_stop.should_stop is True


def test_context_manager_falls_back_to_plain_counting_when_encoding_unavailable(monkeypatch) -> None:
    manager = ContextManager("gpt-4.1-mini")
    manager.encoding = None
    assert manager.count_tokens("one two three") == 3


def test_context_manager_compresses_long_text() -> None:
    manager = ContextManager("gpt-4.1-mini")
    text = "\n".join(f"line {index}" for index in range(30))
    compressed = manager._compress_text(text, 20)
    assert compressed
    assert compressed != text


def test_env_loader_reads_project_and_venv_env_files_without_overriding_shell(monkeypatch) -> None:
    import shutil

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    env_root = PROJECT_ROOT / ".test_memory" / str(uuid.uuid4())
    venv_directory = env_root / ".venv"
    venv_directory.mkdir(parents=True)
    try:
        (venv_directory / ".env").write_text('OPENAI_API_KEY="test-key"\nOPENAI_MODEL=gpt-4.1-mini\n', encoding="utf-8")

        loaded_paths = load_project_env(env_root)

        assert (venv_directory / ".env").resolve() in loaded_paths
        assert os.environ["OPENAI_API_KEY"] == "test-key"
        assert os.environ["OPENAI_MODEL"] == "gpt-4.1-mini"
    finally:
        shutil.rmtree(env_root, ignore_errors=True)


def test_reasoning_engine_falls_back_when_model_call_fails() -> None:
    engine = ReasoningEngine("gpt-4.1-mini")
    engine.chat_model = object()
    engine._reason_with_model = _raise_runtime_error  # type: ignore[method-assign]
    batch = RetrievalBatch([_make_candidate("include/json_pointer.hpp")], "path")
    route = RouteDecision("code_understanding", 0.9, True, False, "retrieval_first", ["retrieve relevant repository evidence"])

    outcome = engine.reason("What does json_pointer do?", route, batch, [], None, None)

    assert "include/json_pointer.hpp" in outcome.summary_text


def test_reasoning_engine_bypasses_model_for_missing_command_failures() -> None:
    engine = ReasoningEngine("gpt-4.1-mini")
    engine.chat_model = object()
    engine._can_call_model = _always_true  # type: ignore[method-assign]
    engine._reason_with_model = _raise_runtime_error  # type: ignore[method-assign]
    route = RouteDecision("build_test_debug", 0.95, True, True, "execution_first", ["run build or test commands and capture deterministic output"])
    execution = [ExecutionBatch("build", [CommandExecutionResult(["cmake", "--build", "build"], 127, "", "[WinError 2] The system cannot find the file specified", "build")])]
    analysis = AnalysisReport(
        "Required command is unavailable while running: cmake --build build",
        [RootCauseCandidate("Required command is unavailable while running: cmake --build build", confidence=0.95)],
        [],
        [],
        "answer_with_limited_evidence",
    )

    outcome = engine.reason("Run the tests", route, None, execution, analysis, type("Memory", (), {"prompt_text": "memory"})())

    assert "unavailable on this machine" in outcome.summary_text
    assert "answer_with_limited_evidence" not in "\n".join(outcome.next_steps)


def test_assignment_agent_cli_parser_supports_expected_options() -> None:
    parser = AssignmentAgentCli().build_parser()
    args = parser.parse_args(["--repo", "repo", "--query", "what", "--show-prompt-report"])
    assert args.repo == "repo"
    assert args.query == "what"
    assert args.show_prompt_report is True


