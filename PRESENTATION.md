# Goal-Driven Assignment Agent Presentation

## What This Version Emphasizes

Retrieval planning is driven by the task objective and required evidence, 
not primarily by low-level search modes.

## One-Sentence Pitch

"This agent works on a local C++ repo, answers code questions,
runs real build and test commands, 
and stays inside a hard 5,000-token context limit by planning retrieval around evidence requirements."

## Core Presentation Points

### Problem

- repository is too large for the prompt window
- build failures require both command execution and source retrieval
- the agent must decide what to load instead of loading everything

### Architecture

- `TaskRouter` classifies the query
- `AssignmentAgentController` runs the constrained loop
- `RetrievalPlanner` selects task kind, evidence requirements, and retrieval steps
- `RepositoryService` executes the retrieval chain
- `BuildRunner` and `TestRunner` execute safe local commands
- `ContextManager` enforces the token budget
- `ReasoningEngine` synthesizes the answer from selected evidence

### Retrieval Story


"The planner first decides what kind of question this is, 
then decides what minimum evidence is needed, 
and only after that chooses retrieval operators 
like symbol lookup, path lookup, literal search, keyword search, or execution-guided lookup."

### Command Safety Story

Say:

"Commands are classified before execution.
Read-only commands and build-artifact-write commands are allowed. 
Source-modifying commands are blocked by policy."

Safety levels:

- `READ_ONLY`
- `BUILD_ARTIFACT_WRITE`
- `SOURCE_MODIFY`

### Patterns To Name

- Strategy Pattern for task-specific retrieval planning
- lightweight state-machine-style workflow for the agent loop
- Policy Pattern for command safety and token budgeting
- Chain of Responsibility for retrieval fallback ordering



## Deliverables Summary

The assignment requires:

- GitHub repository
- demo video
- design document


## Recommended Demo Queries

- "What does the `json_pointer` class do and where is it defined?"
- "Build the project and run the tests. If any test fails, explain why."
- "Which files are responsible for JSON serialization?"

## Honest Limitations To Mention

- retrieval is still structured and lexical, not semantic embedding retrieval
- repository and execution services are not wrapped behind a true facade yet


