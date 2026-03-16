from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from assignment_agent.agent_controller import AssignmentAgentController


REAL_REPO = Path(r"C:\Users\Yuliya\source\repos\jsonOpenSource")


def test_real_repo_understanding_path_exists_and_can_answer() -> None:
    if not REAL_REPO.exists():
        return
    controller = AssignmentAgentController(REAL_REPO)
    response = controller.answer_query("What does the json_pointer class do and where is it defined?")
    assert response.task_type == "code_understanding"
    assert response.prompt_report is not None
    assert response.answer_text
    assert "json_pointer" in response.answer_text


def test_real_repo_build_path_can_be_invoked() -> None:
    if not REAL_REPO.exists():
        return
    if os.environ.get("RUN_REAL_REPO_BUILD_TESTS") != "1":
        return
    controller = AssignmentAgentController(REAL_REPO)
    response = controller.answer_query("Build the project and run the tests. If any test fails, explain why.")
    assert response.execution_batches
    assert response.execution_batches[0].phase_name == "build"
