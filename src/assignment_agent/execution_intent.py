"""Token-based execution intent classification for local command planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# Normalize punctuation into whitespace so simple token checks stay predictable.
NORMALIZATION_TABLE = str.maketrans({character: " " for character in ",.;:!?()[]{}"})

# These token groups drive deterministic command classification. Keep them small and
# readable so the command path does not turn into a regex-heavy parser.
ACTION_TOKENS = {"run", "build", "execute", "configure", "compile", "show", "list", "rerun"}
BUILD_TOKENS = {"build", "compile"}
TEST_TOKENS = {"test", "tests", "ctest"}
SHOW_TOKENS = {"show", "list"}
OPTION_TOKENS = {"option", "options", "cache", "variable", "variables"}
TARGET_TOKENS = {"target", "targets"}
DEBUG_TOKENS = {"debug", "failure", "failing"}
STOP_TOKENS = {"and", "then", "with", "using", "after", "for", "the", "a", "an", "project"}
GENERIC_TARGET_TOKENS = {"project", "build", "test", "tests", "cmake", "make", "ninja"}


class ExecutionRequestKind(str, Enum):
    """High-level execution requests supported by the agent."""

    NONE = "none"
    CONFIGURE = "configure"
    SHOW_CMAKE_OPTIONS = "show_cmake_options"
    SHOW_BUILD_TARGETS = "show_build_targets"
    LIST_TESTS = "list_tests"
    BUILD = "build"
    TEST = "test"
    BUILD_AND_TEST = "build_and_test"
    FILTERED_TEST = "filtered_test"
    BUILD_TARGET = "build_target"
    MAKEFILE = "makefile"
    DEBUG_FAILURE = "debug_failure"


class BackendPreference(str, Enum):
    """Preferred build backend when the user explicitly requests one."""

    DEFAULT = "default"
    MAKE = "make"
    NINJA = "ninja"


@dataclass(frozen=True)
class ExecutionIntent:
    """Parse execution-oriented user intent once and reuse it everywhere."""

    raw_query: str
    lowered_query: str
    normalized_query: str
    tokens: tuple[str, ...]
    token_set: frozenset[str]

    @classmethod
    def from_query(cls, query_text: str) -> "ExecutionIntent":
        """Normalize one raw query into token data reused across the command path."""
        lowered_query = query_text.lower()
        normalized_query = " ".join(lowered_query.translate(NORMALIZATION_TABLE).split())
        tokens = tuple(normalized_query.split()) if normalized_query else ()
        return cls(query_text, lowered_query, normalized_query, tokens, frozenset(tokens))

    def request_kind(self) -> ExecutionRequestKind:
        """Classify the request into one execution plan kind."""
        if self.extract_makefile_name():
            return ExecutionRequestKind.MAKEFILE

        direct_tool_name = self.direct_tool_name()
        if direct_tool_name:
            direct_kind = self._classify_direct_tool_request(direct_tool_name)
            if direct_kind is not ExecutionRequestKind.NONE:
                return direct_kind

        if self._looks_like_cmake_options_request():
            return ExecutionRequestKind.SHOW_CMAKE_OPTIONS
        if self._looks_like_build_target_listing_request():
            return ExecutionRequestKind.SHOW_BUILD_TARGETS
        if self._looks_like_test_listing_request():
            return ExecutionRequestKind.LIST_TESTS
        if self._looks_like_debug_request():
            return ExecutionRequestKind.DEBUG_FAILURE
        if self._looks_like_configure_only_request():
            return ExecutionRequestKind.CONFIGURE

        test_filter = self.extract_test_filter()
        if test_filter:
            return ExecutionRequestKind.FILTERED_TEST

        build_target = self.extract_build_target_name()
        if build_target:
            return ExecutionRequestKind.BUILD_TARGET

        mentions_build = self._mentions_build_words()
        mentions_tests = self._mentions_test_words()
        mentions_run = self._mentions_action("run", "rerun")
        action_oriented = self._looks_like_action_oriented_request()
        if self._looks_like_build_and_test_request(mentions_build, mentions_tests, mentions_run):
            return ExecutionRequestKind.BUILD_AND_TEST
        if mentions_tests and action_oriented:
            return ExecutionRequestKind.TEST
        if mentions_build and action_oriented:
            return ExecutionRequestKind.BUILD

        if self.backend_preference() is not BackendPreference.DEFAULT and action_oriented:
            return ExecutionRequestKind.BUILD
        return ExecutionRequestKind.NONE

    def requests_execution(self) -> bool:
        """Return True when the user is asking for local command execution."""
        return self.request_kind() is not ExecutionRequestKind.NONE

    def requests_build_execution(self) -> bool:
        """Return True when the query explicitly asks to build or compile."""
        return self.request_kind() in {
            ExecutionRequestKind.BUILD,
            ExecutionRequestKind.BUILD_AND_TEST,
        }

    def requests_test_execution(self) -> bool:
        """Return True when the query explicitly asks to run tests."""
        return self.request_kind() in {
            ExecutionRequestKind.TEST,
            ExecutionRequestKind.BUILD_AND_TEST,
            ExecutionRequestKind.FILTERED_TEST,
        }

    def requests_show_cmake_options(self) -> bool:
        """Return True when the query asks for CMake cache or option output."""
        return self.request_kind() is ExecutionRequestKind.SHOW_CMAKE_OPTIONS

    def requests_build_targets(self) -> bool:
        """Return True when the query asks to list build targets."""
        return self.request_kind() is ExecutionRequestKind.SHOW_BUILD_TARGETS

    def requests_list_tests(self) -> bool:
        """Return True when the query asks to enumerate discovered tests."""
        return self.request_kind() is ExecutionRequestKind.LIST_TESTS

    def requests_configure_only(self) -> bool:
        """Return True when the query asks to configure without building or testing."""
        return self.request_kind() is ExecutionRequestKind.CONFIGURE

    def requests_debug_failure_help(self) -> bool:
        """Return True when the query asks for execution-oriented failure triage."""
        return self.request_kind() is ExecutionRequestKind.DEBUG_FAILURE

    def wants_make_backend(self) -> bool:
        """Return True when the user explicitly prefers a make backend."""
        return self.backend_preference() is BackendPreference.MAKE

    def wants_ninja_backend(self) -> bool:
        """Return True when the user explicitly prefers a Ninja backend."""
        return self.backend_preference() is BackendPreference.NINJA

    def backend_preference(self) -> BackendPreference:
        """Return the preferred backend requested by the user, if any."""
        if "make" in self.token_set:
            return BackendPreference.MAKE
        if "ninja" in self.token_set:
            return BackendPreference.NINJA
        return BackendPreference.DEFAULT

    def direct_tool_name(self) -> str:
        """Return one directly invoked build tool name from the query, if present."""
        if not self.tokens:
            return ""
        first_token = self.tokens[0]
        if first_token in {"cmake", "ctest", "make", "ninja"}:
            return first_token
        return ""

    def extract_makefile_name(self) -> str:
        """Return an explicitly named Makefile or `*.make` file from the query."""
        quoted_value = self._extract_quoted_value()
        if self._looks_like_makefile_name(quoted_value):
            return quoted_value
        for raw_token in self.raw_query.split():
            cleaned_token = raw_token.strip(" `\"'.,;:()[]{}")
            if self._looks_like_makefile_name(cleaned_token):
                return cleaned_token
        return ""

    def extract_build_target_name(self) -> str:
        """Return an explicitly named build target from the query."""
        if self.extract_makefile_name():
            return ""

        direct_tool_name = self.direct_tool_name()
        if direct_tool_name == "cmake":
            tool_target = self._extract_tool_option_value("--target")
            if self._looks_like_build_target_name(tool_target):
                return tool_target

        if not self.tokens or self.tokens[0] not in {"run", "build", "execute"}:
            return ""

        start_index = 2 if len(self.tokens) > 1 and self.tokens[1] == "target" else 1
        candidate = self._extract_named_value(start_index)
        if self._looks_like_build_target_name(candidate):
            return candidate

        quoted_value = self._extract_quoted_value()
        if self._looks_like_build_target_name(quoted_value):
            return quoted_value
        return ""

    def wants_valgrind(self) -> bool:
        """Return True when the user requests Valgrind-related CMake options."""
        return "valgrind" in self.token_set

    def wants_fast_tests(self) -> bool:
        """Return True when the user asks to enable the fast-test option."""
        return {"fast", "skip"} & self.token_set and {"test", "tests"} & self.token_set

    def extract_test_filter(self) -> str:
        """Return one named test filter from the query, if present."""
        direct_tool_name = self.direct_tool_name()
        if direct_tool_name == "ctest":
            tool_filter = self._extract_tool_option_value("-R")
            if tool_filter:
                return tool_filter

        if not self._mentions_test_words():
            return ""

        if not self._looks_like_filtered_test_request():
            return ""

        quoted_value = self._extract_quoted_value()
        if quoted_value:
            return quoted_value

        for prefix_tokens in (
            ("run", "only", "test"),
            ("run", "only", "tests"),
            ("run", "test", "named"),
            ("run", "tests", "named"),
            ("run", "test", "called"),
            ("run", "tests", "called"),
            ("run", "test", "matching"),
            ("run", "tests", "matching"),
            ("run", "test"),
            ("run", "tests"),
            ("rerun", "test"),
            ("rerun", "tests"),
        ):
            if self._starts_with(*prefix_tokens):
                return self._extract_named_value(len(prefix_tokens))
        return ""

    def looks_like_direct_tool_command(self) -> bool:
        """Return True when the query starts with a supported tool name."""
        return bool(self.direct_tool_name())

    def _looks_like_build_and_test_request(self, mentions_build: bool, mentions_tests: bool, mentions_run: bool) -> bool:
        """Return True when the query is clearly asking for both build and test work."""
        return mentions_build and mentions_tests and mentions_run

    def _looks_like_filtered_test_request(self) -> bool:
        """Return True when the query names or scopes one test selection."""
        if self._starts_with("run", "only", "test") or self._starts_with("run", "only", "tests"):
            return True
        if self._starts_with("run", "test") or self._starts_with("run", "tests"):
            return True
        if self._starts_with("rerun", "test") or self._starts_with("rerun", "tests"):
            return True
        return bool({"matching", "named", "called"} & self.token_set)

    def _classify_direct_tool_request(self, tool_name: str) -> ExecutionRequestKind:
        """Classify raw tool-command style queries."""
        if tool_name == "ctest":
            if "-n" in self.token_set:
                return ExecutionRequestKind.LIST_TESTS
            if self._extract_tool_option_value("-R"):
                return ExecutionRequestKind.FILTERED_TEST
            return ExecutionRequestKind.TEST

        if tool_name == "make":
            if self.extract_makefile_name():
                return ExecutionRequestKind.MAKEFILE
            if "test" in self.token_set or "tests" in self.token_set:
                return ExecutionRequestKind.TEST
            return ExecutionRequestKind.BUILD

        if tool_name == "ninja":
            if "test" in self.token_set or "tests" in self.token_set:
                return ExecutionRequestKind.TEST
            return ExecutionRequestKind.BUILD

        if tool_name != "cmake":
            return ExecutionRequestKind.NONE
        if "-lah" in self.token_set and "-n" in self.token_set:
            return ExecutionRequestKind.SHOW_CMAKE_OPTIONS
        if "--build" in self.token_set:
            target_name = self._extract_tool_option_value("--target")
            if target_name == "help":
                return ExecutionRequestKind.SHOW_BUILD_TARGETS
            if target_name:
                return ExecutionRequestKind.BUILD_TARGET
            return ExecutionRequestKind.BUILD
        if "-s" in self.token_set and "-b" in self.token_set:
            return ExecutionRequestKind.CONFIGURE
        return ExecutionRequestKind.CONFIGURE

    def _looks_like_cmake_options_request(self) -> bool:
        """Return True when the query asks for CMake options or cache values."""
        return self._mentions_show_words() and self._mentions_any("cmake", "cache", "variable", "variables") and self._mentions_any(*OPTION_TOKENS)

    def _looks_like_build_target_listing_request(self) -> bool:
        """Return True when the query asks to list available build targets."""
        return self._mentions_show_words() and self._mentions_any(*TARGET_TOKENS)

    def _looks_like_test_listing_request(self) -> bool:
        """Return True when the query asks to enumerate tests."""
        return self._mentions_show_words() and self._mentions_test_words()

    def _looks_like_configure_only_request(self) -> bool:
        """Return True when the query asks only for configure-time work."""
        if self._mentions_test_words() or self._mentions_build_words():
            return False
        if self.extract_build_target_name() or self.extract_test_filter():
            return False
        if self.backend_preference() is not BackendPreference.DEFAULT:
            return False
        return "configure" in self.token_set or ("cmake" in self.token_set and "run" in self.token_set) or "valgrind" in self.token_set

    def _looks_like_debug_request(self) -> bool:
        """Return True when the query is asking for build or test failure diagnosis."""
        return bool(DEBUG_TOKENS & self.token_set) and (self._mentions_build_words() or self._mentions_test_words())

    def _looks_like_action_oriented_request(self) -> bool:
        """Return True when the query is asking the agent to perform a command action."""
        if self.looks_like_direct_tool_command():
            return True
        if self._mentions_any(*ACTION_TOKENS):
            return True
        if self.backend_preference() is not BackendPreference.DEFAULT:
            return True
        return bool(self.extract_makefile_name())

    def _mentions_action(self, *tokens: str) -> bool:
        """Return True when any action token is present."""
        return any(token in self.token_set for token in tokens)

    def _mentions_any(self, *tokens: str) -> bool:
        """Return True when any token is present."""
        return any(token in self.token_set for token in tokens)

    def _mentions_show_words(self) -> bool:
        """Return True when the request is in a show/list form."""
        return bool(SHOW_TOKENS & self.token_set)

    def _mentions_build_words(self) -> bool:
        """Return True when the request is about building or compiling."""
        return bool(BUILD_TOKENS & self.token_set)

    def _mentions_test_words(self) -> bool:
        """Return True when the request is about tests."""
        return bool(TEST_TOKENS & self.token_set)

    def _starts_with(self, *tokens: str) -> bool:
        """Return True when the normalized token sequence begins with the given tokens."""
        if len(self.tokens) < len(tokens):
            return False
        return self.tokens[: len(tokens)] == tokens

    def _extract_named_value(self, start_index: int) -> str:
        """Extract one named token value after an action prefix."""
        if start_index >= len(self.tokens):
            return ""
        for token in self.tokens[start_index:]:
            if token in STOP_TOKENS or token in ACTION_TOKENS:
                continue
            cleaned_token = token.strip(" `\"'")
            if cleaned_token and not cleaned_token.startswith("-"):
                return cleaned_token
        return ""

    def _extract_quoted_value(self) -> str:
        """Return the first quoted or backticked value."""
        for quote_character in ('"', "'", "`"):
            start_index = self.raw_query.find(quote_character)
            if start_index == -1:
                continue
            end_index = self.raw_query.find(quote_character, start_index + 1)
            if end_index <= start_index + 1:
                continue
            value = self.raw_query[start_index + 1 : end_index].strip()
            if value:
                return value
        return ""

    def _extract_tool_option_value(self, option_name: str) -> str:
        """Extract one raw option value from a direct tool command query."""
        raw_tokens = self.raw_query.split()
        lowered_tokens = [token.lower() for token in raw_tokens]
        try:
            option_index = lowered_tokens.index(option_name.lower())
        except ValueError:
            return ""
        if option_index + 1 >= len(raw_tokens):
            return ""
        return raw_tokens[option_index + 1].strip(" `\"'")

    def _looks_like_makefile_name(self, value: str) -> bool:
        """Return True when a token names a Makefile or makefile fragment."""
        if not value:
            return False
        lowered_value = value.lower()
        return lowered_value == "makefile" or lowered_value.endswith(".make")

    def _looks_like_build_target_name(self, value: str) -> bool:
        """Return True when a token looks like a CMake or backend build target name."""
        if not value:
            return False
        lowered_value = value.lower()
        if lowered_value in GENERIC_TARGET_TOKENS:
            return False
        return all(character.isalnum() or character in {"_", "-", ".", ":"} for character in value)
