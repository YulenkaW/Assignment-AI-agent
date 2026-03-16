# Assignment Agent

This repository contains a separate assignment-focused implementation in `src/assignment_agent` for the context-constrained C++ agent task.
The target repository is a local copy of `nlohmann/json`, and the agent is designed to answer code-understanding queries and run real local build and test commands while enforcing a hard 5,000-token context limit.

## Assignment Fit

The implementation follows the assignment requirements shown in the PDF:

- It works on a local pre-downloaded repository.
- It handles two task families:
  - codebase understanding
  - build and execution
- It executes real local commands instead of simulating them.
- It enforces the 5,000-token cap programmatically.
- It uses a constrained agent loop instead of an unconstrained free-form model loop.

## Architecture

The submission-facing package is `assignment_agent`.
### Main flow

1. User / UI
2. Agent Controller
3. Task Router
4. Retrieval Planner or Execution Path
5. Repository Service / Build Runner / Test Runner
6. Command Output Capture and Output Parser
7. Build/Test Analyzer
8. Context Manager
9. Reasoning Engine
10. Response Generator

### Runtime flow description

#### Runtime data flow

The runtime path is:

1. The user sends a query through the UI or CLI.
2. The controller routes the query and creates a small plan.
3. The controller loop decides whether it needs retrieval, command execution, output parsing, or failure analysis.
4. Repository retrieval returns a `RetrievalBatch` when the query needs code evidence.
5. Build or test execution returns `ExecutionBatch` results when the query needs local commands.
6. Parsed and analyzed execution output can trigger another retrieval pass when command output points to a relevant source file.
7. The context manager assembles a limited working memory that stays under the token cap.
8. The reasoning engine builds the answer from grounded evidence.
9. The response generator formats the final output for the user.

#### Structure overview

The code is organized into five practical layers:

- Entry points:
  - `demo_ui.py`
  - `command_line.py`
- Core orchestration:
  - `AssignmentAgentController`
  - `TaskRouter`
  - `PlanningLayer`
  - `StopRetryController`
  - `ExecutionIntent`
- Retrieval layer:
  - `RetrievalPlanner`
  - `RetrievalStrategies`
  - `RepositoryService`
  - `RepositoryIndex`
  - `ExternalMemoryStore`
- Execution layer:
  - `BuildRunner`
  - `TestRunner`
  - `CommandExecutor`
  - `OutputParser`
  - `BuildTestAnalyzer`
- Synthesis layer:
  - `ContextManager`
  - `ReasoningEngine`
  - `ResponseGenerator`

The controller is the main coordinator. It calls into retrieval and execution services, then merges both evidence streams before answer generation.

#### Retrieval logic data flow

The retrieval path works in this order:

1. The planner reads the query, route decision, and optional execution analysis.
2. It classifies the task into a meaningful objective such as:
   - definition lookup
   - location lookup
   - responsibility analysis
   - literal search
   - build failure analysis
   - test failure analysis
   - mixed
3. It builds an evidence policy for that task, including:
   - how many files are needed
   - how many supporting chunks per file are needed
   - whether neighboring chunks should be expanded
   - how much token budget retrieval may consume
4. It chooses ordered retrieval operators. These are backend methods, not the top-level planner abstraction:
   - execution-guided
   - literal text
   - path
   - symbol
   - keyword
5. The repository service runs those operators against the repository index.
6. Results are merged, deduplicated, neighbor-expanded when requested, and trimmed to the retrieval limit.
7. The final `RetrievalBatch` is handed to the context manager for prompt assembly.

### Module map

- `src/assignment_agent/agent_controller.py`: top-level controller and loop coordination
- `src/assignment_agent/task_router.py`: query classification
- `src/assignment_agent/planning_layer.py`: high-level plan and stop conditions
- `src/assignment_agent/retrieval_planner.py`: goal-driven retrieval planning around task kind, evidence requirements, and budget
- `src/assignment_agent/retrieval_strategies.py`: strategy objects for definition, responsibility, failure analysis, and directed search
- `src/assignment_agent/repository_index.py`: file indexing, symbol extraction, structure-aware chunking
- `src/assignment_agent/repository_service.py`: backend retrieval operators and ordered fallback execution
- `src/assignment_agent/external_memory_store.py`: persisted summaries and notes outside the prompt window
- `src/assignment_agent/context_manager.py`: token budgeting, prompt assembly, and truncation reporting
- `src/assignment_agent/build_runner.py`: configure and build commands
- `src/assignment_agent/test_runner.py`: test execution commands
- `src/assignment_agent/command_output_capture.py`: normalized command-output capture
- `src/assignment_agent/output_parser.py`: deterministic extraction of error lines and file references
- `src/assignment_agent/build_test_analyzer.py`: failure analysis and next-step recommendation
- `src/assignment_agent/stop_retry_controller.py`: stop/retry decision logic
- `src/assignment_agent/reasoning_engine.py`: grounded explanation with optional OpenAI-backed synthesis
- `src/assignment_agent/response_generator.py`: final response formatting
- `src/assignment_agent/contracts.py`: wrapper classes for inter-module contracts

Note: `tool_orchestrator.py` remains in the repository as a legacy helper for older tests and experiments, but it is not part of the main runtime path of this goal-driven submission.

## Execution Policy

The execution policy is intentionally conservative and portable.

Default policy:

- Configure with `cmake -S <repo> -B <build>`
- Build with `cmake --build <build>`
- Test with `ctest --test-dir <build>`

Additional non-mutating commands supported when the query explicitly asks for them:

- Show configured CMake cache/options with `cmake -LAH -N <build>`
- List configured build targets with `cmake --build <build> --target help`
- List discovered tests with `ctest --test-dir <build> -N`
- Run a filtered test subset with `ctest --test-dir <build> -R <pattern> --output-on-failure`

## Command Safety Model

Commands are classified before execution by `command_safety.py`.

Safety levels:

- `READ_ONLY`
- `BUILD_ARTIFACT_WRITE`
- `SOURCE_MODIFY`

Default policy:

- allow read-only inspection commands
- allow build/test commands only when writes stay under generated artifact directories
- block source-modifying commands by default

Examples:

- read-only:
  - `rg`
  - `grep`
  - `find`
  - `cat`
  - `cmake -LAH -N <build>`
  - `ctest --test-dir <build> -N`
- build-artifact-write:
  - `cmake -S <repo> -B <build>`
  - `cmake --build <build>`
- `ctest --test-dir <build> --output-on-failure`
- `make -C <build>`
- `ninja -C <build>`
- `make -f <repo_makefile>`
- blocked by default:
  - `git reset --hard`
  - `rm -rf`
  - `mv` or `cp` into source folders
  - patching or formatters that rewrite tracked files

Execution safeguards:

- whitelist allowed command forms
- fixed working directory
- timeout on subprocess execution
- stdout/stderr capture
- `shell=False`
- no writes outside build or temporary artifact directories

Raw `make` is not preferred.
It is treated as an optional backend-specific path only when:

- there is no CMake project, and
- a backend-specific build directory is being targeted, and
- the environment actually provides `make`

An explicitly named repository makefile such as `make -f ci.make` is allowed only when the user directly asks for that file to be run.

This is the safer policy for the assignment and for Windows.

Repo-local scripts under `tools/` are intentionally not part of the default command surface for the assignment agent.
Most of them generate files, format sources, or produce release artifacts, which would mutate the target repository.
For the same reason, top-level Makefile targets such as `amalgamate`, `pretty`, `release`, `run_benchmarks`, and `serve_header` are not executed by default.

## Agentic Design Choice

This implementation uses a constrained agentic pattern.

- The model may propose the next action.
- The controller validates the proposal against an allowed action list.
- The task router stays deterministic first and only uses the OpenAI API as a last-resort intent fallback for ambiguous action phrasing.
- Retrieval, token budgeting, command execution, output parsing, and stop conditions remain deterministic.
- The model is used for reasoning and optionally action selection, not unconstrained repository search.

This is more reliable than letting the model freely choose arbitrary searches or commands.

## Token Strategy

The 5,000-token cap is enforced programmatically by `context_manager.py`.
The budget includes:

- prompt instructions
- user query
- retrieved code
- external memory inserted into working memory
- command output included in working memory
- reserved output budget

The code keeps retrieval budget-aware before retrieval runs, not only after prompt assembly.

## Goal-Driven Retrieval Design

This copy replaces the older top-level `path / symbol / keyword / literal` planning abstraction with a goal-driven planner:

1. classify the task objective
2. define the minimum evidence needed
3. choose ordered backend retrieval operators

Task kinds used in this variant:

- `definition_lookup`
- `location_lookup`
- `responsibility_analysis`
- `literal_search`
- `build_failure_analysis`
- `test_failure_analysis`
- `mixed`

The backend retrieval operators still exist, but they are now implementation details of the plan rather than the main planning abstraction.

This aligns better with the assignment because the planner is centered on:

- what evidence is needed before answering
- when exact retrieval is enough
- when to fall back to broader lexical retrieval
- how many files or chunks should be loaded under the token budget

## Install

```powershell
cd C:\Users\Yuliya\Documents\Playground
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

If you want OpenAI-backed action proposal and final reasoning:

```powershell
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_MODEL="gpt-4.1-mini"
```

## Demo Commands

### Fixture demo

```powershell
$env:PYTHONPATH="src"
python -m assignment_agent.command_line --repo tests\fixtures\mini_cpp --query "What does the json_pointer class do and where is it defined?" --show-prompt-report
python -m assignment_agent.command_line --repo tests\fixtures\mini_cpp --query "Build the project and run the tests. If any test fails, explain why." --show-prompt-report
```

### Real local repo demo

```powershell
$env:PYTHONPATH="src"
python -m assignment_agent.command_line --repo C:\Users\Yuliya\source\repos\jsonOpenSource --query "What does the json_pointer class do and where is it defined?" --show-prompt-report
python -m assignment_agent.command_line --repo C:\Users\Yuliya\source\repos\jsonOpenSource --query "Build the project and run the tests. If any test fails, explain why." --show-prompt-report
```

## Verification

Implemented verification paths:

- `python -m compileall src\assignment_agent tests\test_assignment_agent.py`
- dedicated assertion-based tests in `tests/test_assignment_agent.py`
- optional real-repo test path in `tests/test_assignment_agent_real_repo.py`

## Assignment Deliverables

The assignment PDF requires:

- a GitHub repository with working code and a clear README
- a short demo video showing at least one understanding query and one build or execution query
- a design document describing the architecture, context strategy, and design decisions

This project copy already covers the code and README portions.
The design explanation is included in this README and can be supported with the companion docs below.

Still required outside the repo:

- record the demo video
- push the final project to GitHub

## Presentation Resources

Use these companion docs when presenting or packaging the submission:

- `PRESENTATION.md`
- `DELIVERABLES.md`

## Known Limitations

- The retrieval strategy is lexical and structure-aware, not semantic-first.
- If `cmake` or `ctest` is missing, the agent reports the environment problem instead of fixing it.
- `pytest` is not installed in this environment, so direct `pytest` execution is unavailable here.
- The real `nlohmann/json` test suite may depend on local toolchain details and environment setup.
