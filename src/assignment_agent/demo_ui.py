"""Simple desktop UI for demoing the assignment agent."""

from __future__ import annotations

from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from .agent_controller import AssignmentAgentController


class AssignmentAgentDemoUi:
    """Small Tkinter UI for interactive demo runs."""

    DEFAULT_QUERY = "What does the json_pointer class do and where is it defined?"

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Assignment Agent Goal-Driven Demo")
        self.root.geometry("1480x940")
        self.root.minsize(1240, 820)

        self.repo_var = tk.StringVar(value=self._default_repo_path())
        self.model_var = tk.StringVar(value="gpt-4.1-mini")
        self.tokens_var = tk.StringVar(value="5000")
        self.status_var = tk.StringVar(value="Ready")
        self.controller_cache = {}
        self.active_session_key = None
        self.session_turns = []
        self.run_sequence = 0
        self.active_run_id = None
        self.active_run_started_at = 0.0

        self._configure_style()
        self._build_layout()
        self._set_running_state(False)

    def run(self) -> None:
        """Start the desktop UI event loop."""
        self.root.mainloop()

    def _build_layout(self) -> None:
        """Create the form and result panes."""
        frame = ttk.Frame(self.root, padding=16, style="App.TFrame")
        frame.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(frame, style="HeroPanel.TFrame", padding=(18, 18, 18, 16))
        header.pack(fill=tk.X, pady=(0, 14))
        ttk.Label(header, text="Assignment Agent Goal-Driven", style="Hero.TLabel").pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Goal-driven retrieval, controlled build/test execution, diagnostics, and prompt-budget reports.",
            style="HeroSubtle.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))

        info_strip = ttk.Frame(frame, style="InfoStrip.TFrame", padding=(12, 10))
        info_strip.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(info_strip, text="Local repo only", style="InfoTitle.TLabel").pack(side=tk.LEFT)
        ttk.Label(info_strip, text="Real command execution, prompt budget reports, and session memory stay enabled.", style="InfoBody.TLabel").pack(side=tk.LEFT, padx=(12, 0))

        self.main_pane = ttk.Panedwindow(frame, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True)

        control_panel = ttk.Frame(self.main_pane, padding=(0, 0, 16, 0), style="App.TFrame")
        result_panel = ttk.Frame(self.main_pane, style="App.TFrame")
        self.main_pane.add(control_panel, weight=0)
        self.main_pane.add(result_panel, weight=1)

        setup_frame = ttk.LabelFrame(control_panel, text="Run Setup", padding=14, style="Sidebar.TLabelframe")
        setup_frame.pack(fill=tk.X)

        ttk.Label(setup_frame, text="Repository", style="Section.TLabel").grid(row=0, column=0, sticky=tk.W)
        repo_entry = ttk.Entry(setup_frame, textvariable=self.repo_var, width=52)
        repo_entry.grid(row=1, column=0, sticky=tk.EW, pady=(4, 10))

        ttk.Label(setup_frame, text="Model", style="Section.TLabel").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(setup_frame, textvariable=self.model_var, width=28).grid(row=3, column=0, sticky=tk.W, pady=(4, 10))

        ttk.Label(setup_frame, text="Max tokens", style="Section.TLabel").grid(row=4, column=0, sticky=tk.W)
        ttk.Entry(setup_frame, textvariable=self.tokens_var, width=12).grid(row=5, column=0, sticky=tk.W, pady=(4, 4))

        setup_frame.columnconfigure(0, weight=1)

        query_frame = ttk.LabelFrame(control_panel, text="Query", padding=14, style="Sidebar.TLabelframe")
        query_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        ttk.Label(query_frame, text="Use a full question or command. Follow-ups reuse the current session.", style="Subtle.TLabel").pack(anchor=tk.W)
        self.query_box = scrolledtext.ScrolledText(query_frame, wrap=tk.WORD, height=9, font=("Segoe UI", 11))
        self.query_box.pack(fill=tk.BOTH, expand=True, pady=(8, 12))
        self.query_box.insert("1.0", self.DEFAULT_QUERY)
        self._configure_text_widget(self.query_box, background="#fcfbf7", foreground="#17212f", border="#d6dbe6")

        button_row = ttk.Frame(query_frame)
        button_row.pack(fill=tk.X)
        self.run_button = ttk.Button(button_row, text="Run Query", command=self._start_run, style="Accent.TButton")
        self.run_button.pack(side=tk.LEFT)
        self.clear_button = ttk.Button(button_row, text="Clear Output", command=self._clear_output)
        self.clear_button.pack(side=tk.LEFT, padx=(8, 0))
        self.reset_button = ttk.Button(button_row, text="Reset Session", command=self._reset_session)
        self.reset_button.pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(query_frame, textvariable=self.status_var, style="Status.TLabel").pack(anchor=tk.W, pady=(12, 0))

        result_shell = ttk.Frame(result_panel, style="ResultsShell.TFrame", padding=14)
        result_shell.pack(fill=tk.BOTH, expand=True)

        result_header = ttk.Frame(result_shell, style="ResultsShell.TFrame")
        result_header.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(result_header, text="Results", style="Section.TLabel").pack(anchor=tk.W)
        ttk.Label(result_header, text="Answer quality, prompt assembly, session history, and memory usage.", style="Subtle.TLabel").pack(anchor=tk.W, pady=(2, 0))

        notebook = ttk.Notebook(result_shell, style="Dashboard.TNotebook")
        notebook.pack(fill=tk.BOTH, expand=True)

        answer_tab = ttk.Frame(notebook)
        prompt_tab = ttk.Frame(notebook)
        stats_tab = ttk.Frame(notebook)
        commands_tab = ttk.Frame(notebook)
        help_tab = ttk.Frame(notebook)
        session_tab = ttk.Frame(notebook)
        memory_tab = ttk.Frame(notebook)

        notebook.add(answer_tab, text="Answer")
        notebook.add(prompt_tab, text="Prompt Report")
        notebook.add(stats_tab, text="Stats")
        notebook.add(commands_tab, text="Commands")
        notebook.add(help_tab, text="Help")
        notebook.add(session_tab, text="Session")
        notebook.add(memory_tab, text="Memory")

        self.answer_box = scrolledtext.ScrolledText(answer_tab, wrap=tk.WORD, height=20, font=("Segoe UI", 11))
        self.answer_box.pack(fill=tk.BOTH, expand=True)
        self._configure_text_widget(self.answer_box, background="#fffdf8", foreground="#16202d", border="#d8dbe5")

        self.prompt_box = scrolledtext.ScrolledText(prompt_tab, wrap=tk.WORD, height=20, font=("Consolas", 10))
        self.prompt_box.pack(fill=tk.BOTH, expand=True)
        self._configure_text_widget(self.prompt_box, background="#fbfaf7", foreground="#173045", border="#d8dbe5")

        self.stats_box = scrolledtext.ScrolledText(stats_tab, wrap=tk.WORD, height=20, font=("Consolas", 10))
        self.stats_box.pack(fill=tk.BOTH, expand=True)
        self._configure_text_widget(self.stats_box, background="#f8fbfc", foreground="#143042", border="#d8dbe5")

        self.commands_box = scrolledtext.ScrolledText(commands_tab, wrap=tk.WORD, height=20, font=("Consolas", 10))
        self.commands_box.pack(fill=tk.BOTH, expand=True)
        self._configure_text_widget(self.commands_box, background="#fbfaf7", foreground="#173045", border="#d8dbe5")
        self._set_text(self.commands_box, "No commands executed yet.")

        self.help_box = scrolledtext.ScrolledText(help_tab, wrap=tk.WORD, height=20, font=("Consolas", 10))
        self.help_box.pack(fill=tk.BOTH, expand=True)
        self._configure_text_widget(self.help_box, background="#fbfaf7", foreground="#173045", border="#d8dbe5")
        self._set_text(self.help_box, self._supported_commands_text())

        self.session_box = scrolledtext.ScrolledText(session_tab, wrap=tk.WORD, height=20, font=("Segoe UI", 11))
        self.session_box.pack(fill=tk.BOTH, expand=True)
        self._configure_text_widget(self.session_box, background="#fffdf8", foreground="#16202d", border="#d8dbe5")

        self.memory_box = scrolledtext.ScrolledText(memory_tab, wrap=tk.WORD, height=20, font=("Consolas", 10))
        self.memory_box.pack(fill=tk.BOTH, expand=True)
        self._configure_text_widget(self.memory_box, background="#f9fafc", foreground="#183044", border="#d8dbe5")

        self.root.after(50, self._set_initial_pane_width)

    def _start_run(self) -> None:
        """Validate inputs and run the agent in a worker thread."""
        if self.active_run_id is not None:
            return

        repository_path = Path(self.repo_var.get().strip())
        query_text = self.query_box.get("1.0", tk.END).strip()

        if not repository_path.exists():
            messagebox.showerror("Invalid repository", f"Repository does not exist:\n{repository_path}")
            return
        if not query_text:
            messagebox.showerror("Missing query", "Enter a query before running the demo.")
            return

        try:
            max_total_tokens = int(self.tokens_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid token limit", "Max tokens must be an integer.")
            return

        session_key = (str(repository_path.resolve()), self.model_var.get().strip() or "gpt-4.1-mini", max_total_tokens)
        if self.active_session_key != session_key:
            self.active_session_key = session_key
            self.session_turns = []
            self.session_box.delete("1.0", tk.END)

        self.run_sequence += 1
        run_id = self.run_sequence
        self.active_run_id = run_id
        self.active_run_started_at = time.perf_counter()
        self.status_var.set("Processing...")
        self._set_running_state(True)
        self._clear_output()
        self._append_text(self.commands_box, f"Starting query: {query_text}\n")

        worker = threading.Thread(
            target=self._run_controller,
            args=(run_id, repository_path, query_text, max_total_tokens, self.model_var.get().strip() or "gpt-4.1-mini"),
            daemon=True,
        )
        worker.start()

    def _run_controller(self, run_id: int, repository_path: Path, query_text: str, max_total_tokens: int, model_name: str) -> None:
        """Run the controller without blocking the UI thread."""
        try:
            controller = self._get_controller(repository_path, model_name, max_total_tokens)
            def report_progress(message_text: str) -> None:
                self.root.after(0, self._append_command_log, run_id, message_text)

            response = controller.answer_query(
                query_text,
                conversation_turns=list(self.session_turns),
                progress_callback=report_progress,
            )
            self.root.after(0, self._show_result, run_id, query_text, response)
        except Exception as error:  # pragma: no cover - UI runtime path
            self.root.after(0, self._show_error, run_id, str(error))

    def _show_result(self, run_id: int, query_text: str, response) -> None:
        """Render a successful controller result."""
        if run_id != self.active_run_id:
            return

        prompt_report_text = ""
        if response.prompt_report is not None:
            prompt_report_text = response.prompt_report.render_text()
        diagnostics_text = ""
        if response.diagnostics is not None:
            diagnostics_text = response.diagnostics.render_text()
        memory_text = self._render_memory(response)

        self._set_text(self.answer_box, response.answer_text)
        self._set_text(self.prompt_box, prompt_report_text)
        self._set_text(self.stats_box, diagnostics_text)
        self._set_text(self.memory_box, memory_text)
        self._append_execution_summary(run_id, response)

        self.session_turns.append(("user", query_text))
        self.session_turns.append(("assistant", response.answer_text))
        self._render_session_history()
        processing_time_ms = getattr(response.diagnostics, "processing_time_ms", 0) if getattr(response, "diagnostics", None) is not None else 0
        self._finish_run(run_id, f"Completed in {processing_time_ms} ms ({len(self.session_turns) // 2} turn(s))")

    def _show_error(self, run_id: int, error_text: str) -> None:
        """Render a runtime error without crashing the UI."""
        if run_id != self.active_run_id:
            return
        self._set_text(self.answer_box, error_text)
        elapsed_ms = int((time.perf_counter() - self.active_run_started_at) * 1000) if self.active_run_started_at else 0
        self._set_text(self.stats_box, f"request_processing_time_ms={elapsed_ms}")
        self._finish_run(run_id, f"Failed after {elapsed_ms} ms")

    def _clear_output(self) -> None:
        """Clear output panes but keep the current session history."""
        self.answer_box.delete("1.0", tk.END)
        self.prompt_box.delete("1.0", tk.END)
        self.stats_box.delete("1.0", tk.END)
        self.commands_box.delete("1.0", tk.END)
        self.memory_box.delete("1.0", tk.END)

    def _reset_session(self) -> None:
        """Clear the session transcript and controller cache for the current settings."""
        self.session_turns = []
        self.controller_cache.clear()
        self.active_session_key = None
        self.run_sequence += 1
        self.active_run_id = None
        self.active_run_started_at = 0.0
        self._clear_output()
        self.session_box.delete("1.0", tk.END)
        self._set_running_state(False)
        self.status_var.set("Session reset")

    def _get_controller(self, repository_path: Path, model_name: str, max_total_tokens: int) -> AssignmentAgentController:
        """Reuse a controller for the same repo and runtime settings."""
        cache_key = (str(repository_path.resolve()), model_name, max_total_tokens)
        controller = self.controller_cache.get(cache_key)
        if controller is None:
            controller = AssignmentAgentController(
                repository_path=repository_path,
                model_name=model_name,
                max_total_tokens=max_total_tokens,
            )
            self.controller_cache[cache_key] = controller
        return controller

    def _render_session_history(self) -> None:
        """Render all recorded turns in the session tab."""
        self.session_box.delete("1.0", tk.END)
        for role, text in self.session_turns:
            self.session_box.insert(tk.END, f"{role}:\n{text}\n\n")

    def _render_memory(self, response) -> str:
        """Render external memory records shown to the reasoning layer."""
        lines = []
        diagnostics = getattr(response, "diagnostics", None)
        if diagnostics is not None:
            lines.append(f"Matched memory records: {diagnostics.external_memory_hits} of {diagnostics.external_memory_total}")
            lines.append(f"Session turns used: {diagnostics.session_turns_used}")
            lines.append("")
        for record in getattr(response, "external_memory_records", []):
            lines.append(f"{record.source_path}")
            lines.append(f"  {record.summary_text}")
            lines.append("")
        return "\n".join(lines)

    def _supported_commands_text(self) -> str:
        """Render the supported query and command surface for the UI."""
        lines = [
            "Supported query types",
            "",
            "Code understanding:",
            "- What does json_pointer do and where is it defined?",
            "- Which files are responsible for JSON serialization?",
            "- Explain `json_pointer` and where it is defined.",
            "",
            "Build and test:",
            "- Build the project",
            "- Run the tests",
            "- Build the project and run the tests",
            "- Why is the build failing?",
            "",
            "Safe execution requests:",
            "- Show CMake options",
            "- Show build targets",
            "- List tests",
            "- Run only test parser_case",
            "- configure with cmake and then run make (backend-specific)",
            "- Build with ninja (backend-specific)",
            "- run ci.make file",
            "",
            "What it runs internally",
            "",
            "- cmake -S <repo> -B <build>",
            "- cmake --build <build> --config Release --parallel",
            "- ctest --test-dir <build> -C Release --output-on-failure",
            "- ctest --test-dir <build> -C Release -N",
            "- ctest --test-dir <build> -C Release -R <pattern> --output-on-failure",
            "- optional make -C <build> or ninja -C <build> when the matching toolchain is available",
            "- named makefile execution: make -f <repo_makefile>",
            "",
            "Blocked by design",
            "",
            "- arbitrary shell commands",
            "- git reset, git checkout, rm -rf, patching, or source-modifying commands",
            "- commands that write outside build or temp artifact directories",
        ]
        return "\n".join(lines)

    def _set_text(self, text_box: scrolledtext.ScrolledText, text: str) -> None:
        """Replace the full contents of one output box."""
        text_box.delete("1.0", tk.END)
        if text:
            text_box.insert(tk.END, text)

    def _append_text(self, text_box: scrolledtext.ScrolledText, text: str) -> None:
        """Append text and keep the latest output visible."""
        if not text:
            return
        text_box.insert(tk.END, text)
        text_box.see(tk.END)

    def _append_command_log(self, run_id: int, message_text: str) -> None:
        """Append one live command-progress line for the active run."""
        if run_id != self.active_run_id:
            return
        self._append_text(self.commands_box, f"{message_text}\n")
        self.status_var.set(message_text[:140])

    def _append_execution_summary(self, run_id: int, response) -> None:
        """Append a compact execution summary after the run completes."""
        if run_id != self.active_run_id:
            return
        execution_batches = getattr(response, "execution_batches", [])
        if not execution_batches:
            return
        self._append_text(self.commands_box, "\nCommand summary\n")
        for execution_batch in execution_batches:
            self._append_text(self.commands_box, f"[{execution_batch.phase_name}]\n")
            for result in execution_batch.results:
                self._append_text(self.commands_box, f"{result.get_command_text()}\n")
                self._append_text(self.commands_box, f"exit_code={result.exit_code}\n")

    def _finish_run(self, run_id: int, status_text: str) -> None:
        """Mark the active run as complete and re-enable the UI."""
        if run_id != self.active_run_id:
            return
        self.active_run_id = None
        self._set_running_state(False)
        self.status_var.set(status_text)

    def _configure_style(self) -> None:
        """Apply a slightly roomier and more legible ttk style."""
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("App.TFrame", background="#f3efe7")
        style.configure("HeroPanel.TFrame", background="#17324d")
        style.configure("InfoStrip.TFrame", background="#efe5d3")
        style.configure("ResultsShell.TFrame", background="#ece7de")
        style.configure("TLabelframe", background="#f3efe7", padding=10)
        style.configure("Sidebar.TLabelframe", background="#fbfaf7", bordercolor="#d8d5cc", relief="solid")
        style.configure("TLabelframe.Label", background="#fbfaf7", foreground="#17324d", font=("Segoe UI Semibold", 11))
        style.configure("TLabel", background="#f3efe7", foreground="#1e293b", font=("Segoe UI", 10))
        style.configure("Hero.TLabel", background="#17324d", foreground="#f8fafc", font=("Segoe UI Semibold", 22))
        style.configure("HeroSubtle.TLabel", background="#17324d", foreground="#d7e1ec", font=("Segoe UI", 10))
        style.configure("Section.TLabel", background="#f3efe7", foreground="#17324d", font=("Segoe UI Semibold", 12))
        style.configure("Subtle.TLabel", background="#f3efe7", foreground="#576274", font=("Segoe UI", 9))
        style.configure("Status.TLabel", background="#fbfaf7", foreground="#0f172a", font=("Segoe UI Semibold", 10))
        style.configure("InfoTitle.TLabel", background="#efe5d3", foreground="#8a4b10", font=("Segoe UI Semibold", 10))
        style.configure("InfoBody.TLabel", background="#efe5d3", foreground="#5b6573", font=("Segoe UI", 9))
        style.configure("TEntry", fieldbackground="#fffdf9", padding=7)
        style.configure("TButton", background="#f6f2ea", foreground="#16324a", padding=(12, 8), bordercolor="#cfd5df")
        style.map("TButton", background=[("active", "#ece5d8"), ("pressed", "#e2dacb")])
        style.configure(
            "Accent.TButton",
            background="#0f766e",
            foreground="#ffffff",
            padding=(14, 9),
            bordercolor="#0f766e",
            focusthickness=0,
        )
        style.map("Accent.TButton", background=[("active", "#115e59"), ("pressed", "#134e4a")], foreground=[("disabled", "#dbe7e5")])
        style.configure("Dashboard.TNotebook", background="#ece7de", borderwidth=0, tabmargins=(0, 0, 0, 0))
        style.configure("Dashboard.TNotebook.Tab", padding=(16, 8), background="#ddd6c8", foreground="#17324d", font=("Segoe UI Semibold", 10))
        style.map("Dashboard.TNotebook.Tab", background=[("selected", "#fffdf8"), ("active", "#ece5d8")], foreground=[("selected", "#0f172a")])
        self.root.configure(background="#f3efe7")

    def _configure_text_widget(self, widget: scrolledtext.ScrolledText, background: str, foreground: str, border: str) -> None:
        """Apply a cleaner visual treatment to one text widget."""
        widget.configure(
            background=background,
            foreground=foreground,
            insertbackground=foreground,
            relief=tk.FLAT,
            bd=0,
            padx=14,
            pady=12,
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor="#0f766e",
            selectbackground="#dbeafe",
            selectforeground="#0f172a",
        )

    def _set_initial_pane_width(self) -> None:
        """Give the control column a stable default width on launch."""
        try:
            self.main_pane.sashpos(0, 380)
        except tk.TclError:
            return

    def _set_running_state(self, is_running: bool) -> None:
        """Update buttons for the active run state."""
        if is_running:
            self.run_button.state(["disabled"])
            return
        self.active_run_started_at = 0.0
        self.run_button.state(["!disabled"])

    def _default_repo_path(self) -> str:
        """Choose a reasonable default repository for demos."""
        real_repo = Path(r"C:\Users\Yuliya\source\repos\jsonOpenSource")
        if real_repo.exists():
            return str(real_repo)
        return str(Path.cwd() / "tests" / "fixtures" / "mini_cpp")


def main() -> None:
    """Run the desktop demo UI."""
    AssignmentAgentDemoUi().run()


if __name__ == "__main__":
    main()
