# Assignment Agent Design

This file documents the separate `assignment_agent` package used for the goal-driven submission.

## Architecture mapping

1. Input and routing
- `task_router.py`: classifies the user request into `code_understanding`, `build_test_debug`, or `mixed`
- `planning_layer.py`: creates the high-level execution plan and stop conditions
- `agent_controller.py`: runs the deterministic controller loop and chooses the next allowed step

2. Retrieval and repo understanding
- `retrieval_planner.py`: chooses task kind, evidence requirements, budget, and ordered retrieval steps
- `repository_index.py`: scans files, extracts symbols, and builds structure-aware chunks
- `repository_service.py`: runs path search, symbol search, keyword search, and chunk retrieval
- `external_memory_store.py`: persists lightweight summaries and execution notes outside the prompt window

3. Context control
- `context_manager.py`: enforces the 5,000-token policy, assembles working memory, and records prompt decisions
- `contracts.py`: defines the wrapper classes passed between modules

4. Execution and analysis
- `build_runner.py`: runs configure and build commands against the local repository
- `test_runner.py`: runs tests when the query and build state require them
- `command_output_capture.py`: normalizes command output before parsing
- `output_parser.py`: extracts deterministic error lines and file references
- `build_test_analyzer.py`: ranks likely root causes and recommends the next action
- `stop_retry_controller.py`: decides whether the controller should stop or retrieve more evidence

5. Reasoning and output
- `reasoning_engine.py`: uses deterministic synthesis or optional OpenAI-backed explanation
- `response_generator.py`: formats the final answer
- `agent_controller.py`: coordinates the full workflow
- `command_line.py`: exposes the separate CLI entry point

## Agentic control model

This package uses a constrained agentic pattern:

- The controller owns the loop and chooses the next allowed step deterministically.
- Retrieval, token budgeting, command execution, output parsing, and stop conditions remain deterministic.
- The model is used for reasoning and optional answer polishing, not unconstrained repository search.

## Inconsistency handling

The package explicitly addresses these implementation risks:

- Router/planner mismatch: the planner consumes a concrete `RouteDecision` contract instead of reclassifying the task.
- Retrieval/context conflict: retrieval is budget-aware before retrieval runs, not only after prompt assembly.
- Header/source disagreement: retrieval expands from a header to a matching implementation file when available.
- Execution evidence priority: generic fallback retrieval is skipped after environment-level command failures unless mixed-task logic requires more context.
- Stop/retry drift: the stop controller owns the retry decision instead of scattering loop logic across modules.
- Generic-answer risk: repository questions are answered from retrieved repo evidence before any optional model-based wording pass.

## OpenAI usage

OpenAI-backed reasoning is optional.
If `OPENAI_API_KEY` is set, `reasoning_engine.py` can use `ChatOpenAI` for final explanation and wording polish.
If not, the package stays runnable through deterministic retrieval, execution, and reasoning.
