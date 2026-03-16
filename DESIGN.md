# Design Document

## Objective

Build a local AI agent that can understand the `nlohmann/json` C++ repository and execute real local build and test commands under a strict 5,000-token context budget.

## High-Level Architecture

The final architecture is organized into five blocks.

### 1. Input and routing

- User / UI
- Agent Controller
- Task Router
- Tool Orchestrator

### 2. Retrieval and repository understanding

- Retrieval Planner
- Repository Index
- Repository Service
- Chunk Retrieval
- External Memory Store

### 3. Context control

- Context Manager
- Token budgeting
- Summarization / compression
- Working memory versus external memory

### 4. Execution and analysis

- Build Runner
- Test Runner
- Command Output Capture
- Output Parser
- Build/Test Analyzer
- Stop/Retry Controller

### 5. Reasoning and output

- Reasoning Engine
- Response Generator
- UI

## Why This Design Fits the Assignment

The assignment requires two task families:

1. Codebase understanding
2. Build and execution

The implementation handles both through one controller while keeping deterministic evidence-handling in the center.

## Constrained Agent Loop

The design uses a constrained agent loop:

- the model may propose the next action
- the controller validates the proposal against an allowlist
- retrieval, token budgeting, command execution, parsing, and stopping remain deterministic

This approach gives agent-like flexibility without sacrificing control over correctness-critical operations.

## Execution Policy

The execution policy is intentionally conservative:

- configure with CMake
- build with `cmake --build`
- test with `ctest`
- use raw backend tools such as `make` only when CMake is unavailable and the environment actually provides them

This is the safest and most portable policy for the assignment, especially on Windows.

## Token Strategy

The 5,000-token limit is enforced programmatically.
The context manager:

- reserves output budget
- reserves instruction budget
- limits retrieval before retrieval runs
- compresses long command output
- drops lower-priority prompt sections when necessary

## Retrieval Strategy

The retrieval strategy is hybrid but deterministic-first:

- path search
- symbol search
- keyword search
- execution-guided chunk retrieval
- structure-aware chunks instead of arbitrary fixed slices when possible

Critical rule: execution evidence and fresh source chunks outrank stored summaries.

## Inconsistency Handling

The design explicitly handles the practical inconsistencies discussed earlier:

- Router/planner mismatch is avoided through explicit contracts.
- Retrieval/context conflicts are reduced by budget-aware retrieval planning.
- Header/source disagreement is reduced by expanding from headers to matching implementation files.
- Build-output noise is reduced by parsing and ranking root-cause candidates.
- Model/tool inconsistency is reduced because the model cannot execute arbitrary actions directly.
- Stop conditions are explicit instead of ad hoc.

## Key Tradeoff

A full agent SDK is not required for this assignment.
A constrained controller-driven implementation is enough, and is often more reliable for:

- hard token-budget enforcement
- deterministic parsing
- safe command execution
- assignment explainability

The chosen design therefore uses optional OpenAI-backed action proposal and reasoning, but not unrestricted model-controlled tool execution.
