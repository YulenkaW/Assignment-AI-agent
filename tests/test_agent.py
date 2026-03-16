from pathlib import Path

from context_agent.build_command_executor import BuildCommandExecutor
from context_agent.build_failure_analyzer import BuildFailureAnalyzer
from context_agent.agent_models import CommandExecutionResult
from context_agent.entity_fact_store import EntityFactStore
from context_agent.main_agent import ContextAwareCodebaseAgent


FIXTURE = Path(__file__).parent / "fixtures" / "mini_cpp"


def test_understanding_query_retrieves_symbol() -> None:
    agent = ContextAwareCodebaseAgent(FIXTURE)
    response = agent.answer_query("What does the json_pointer class do and where is it defined?")
    assert response.task_type == "understanding"
    assert "json_pointer" in response.answer_text
    assert "json_pointer.hpp" in response.answer_text
    assert response.prompt_report is not None


def test_build_failure_parser_extracts_location() -> None:
    executor = BuildCommandExecutor(FIXTURE)
    output_text = "include/json_pointer.hpp:6:12: error: expected ';' after class definition"
    error_location = executor.find_error_location(output_text)
    assert error_location == ("include/json_pointer.hpp", 6)


def test_entity_memory_keeps_multiple_grounded_facts() -> None:
    store = EntityFactStore()
    store.remember_grounded_entity("parse", "parse appears in include/a.hpp", "code", "include/a.hpp")
    store.remember_grounded_entity("parse", "parse appears in include/b.hpp", "code", "include/b.hpp")
    facts = store.get_all_facts_for_entity("parse")
    assert len(facts) == 2


def test_build_failure_analyzer_extracts_identifiers() -> None:
    analyzer = BuildFailureAnalyzer()
    analysis = analyzer.analyze_output("src/main.cpp:10: error: unknown type name json_pointer", "src/main.cpp", 10)
    retrieval_query = analysis.build_retrieval_query("Build the project")
    assert "json_pointer" in retrieval_query
    assert "src/main.cpp" in retrieval_query


def test_build_failure_analyzer_classifies_environment_failures() -> None:
    analyzer = BuildFailureAnalyzer()
    analysis = analyzer.analyze_output("[WinError 2] The system cannot find the file specified", command_text="cmake --build build")
    assert analysis.failure_kind == "environment"
    assert analysis.should_retry_with_happy_path is False
    assert "environmental" in analysis.stop_reason


def test_build_failure_analyzer_requests_retry_for_missing_build_setup() -> None:
    analyzer = BuildFailureAnalyzer()
    analysis = analyzer.analyze_output("Error: could not load cache", command_text="cmake --build build")
    assert analysis.should_retry_with_happy_path is True
    assert "retry" in analysis.stop_reason.lower()


def test_build_execution_query_retries_once_with_happy_path(monkeypatch) -> None:
    agent = ContextAwareCodebaseAgent(FIXTURE)
    command_outputs = [
        CommandExecutionResult(["cmake", "--build", "missing-build"], 1, "", "Error: could not load cache"),
        CommandExecutionResult(["cmake", "-S", str(FIXTURE), "-B", str(FIXTURE / ".agent-build")], 0, "configured", ""),
        CommandExecutionResult(["cmake", "--build", str(FIXTURE / ".agent-build"), "--parallel"], 0, "built", ""),
        CommandExecutionResult(["ctest", "--test-dir", str(FIXTURE / ".agent-build"), "--output-on-failure"], 0, "tests passed", ""),
    ]
    state = {"index": 0}

    def fake_run_command(command_parts, working_directory=None):
        result = command_outputs[state["index"]]
        state["index"] += 1
        return result

    def fake_extract_commands(query_text: str) -> list[list[str]]:
        return [["cmake", "--build", "missing-build"]]

    monkeypatch.setattr(agent.command_executor, "run_command", fake_run_command)
    monkeypatch.setattr(agent, "_extract_commands_from_query", fake_extract_commands)

    response = agent.answer_query("Run `cmake --build missing-build` and explain the result.")

    assert len(response.command_results) == 4
    assert "Retry note" in response.answer_text
    assert "tests passed" in response.answer_text
