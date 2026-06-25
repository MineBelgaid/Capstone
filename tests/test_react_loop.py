"""Stub-LLM tests for the custom ReAct loop -- run WITHOUT Ollama.

Verifies: action parsing/validation, tool dispatch, the reason->act->observe
graph wiring, step-cap termination, and that final_answer flows out. A fake chat
model returns canned JSON so no real LLM backend is needed.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys

# ensure project root importable when run directly
sys.path.insert(0, ".")

from schemas import Task, TaskStatus


class _FakeResp:
    def __init__(self, content: str) -> None:
        self.content = content


class StubModel:
    """Returns a scripted sequence of ReActStep JSON objects."""

    def __init__(self, script: list[dict]) -> None:
        self._script = script
        self._i = 0

    def bind(self, **_kwargs):       # mimic ChatOllama.bind(format="json")
        return self

    def invoke(self, _messages):
        step = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return _FakeResp(json.dumps(step))


def _sample_tasks() -> list[Task]:
    return [
        Task(task_id="T1", title="Login", status=TaskStatus.DONE,
             assignee="Ava", story_points=5, sprint="S1"),
        Task(task_id="T2", title="Docs", status=TaskStatus.BLOCKED,
             assignee="Ben", story_points=3, sprint="S1"),
        Task(task_id="T3", title="DB", status=TaskStatus.IN_PROGRESS,
             assignee="Ava", story_points=2, sprint="S1"),
    ]


def _install_stub(monkeypatch_script: list[dict]) -> None:
    import agent.llm as llm
    import agent.react as react
    stub = StubModel(monkeypatch_script)
    if hasattr(llm.get_chat_model, "cache_clear"):
        llm.get_chat_model.cache_clear()
    # patch the name in every namespace that resolves it
    llm.get_chat_model = lambda: stub          # type: ignore[assignment]
    react.get_chat_model = lambda: stub         # used by _reason_node


def test_dispatch_unknown_tool():
    from agent.react import dispatch_tool
    out = dispatch_tool("nope", "x")
    assert out.startswith("ERROR: unknown tool"), out
    print("dispatch unknown-tool -> OK")


def test_deterministic_tool_dispatch():
    from agent.react import dispatch_tool
    from agent.tools import AgentContext, set_context
    set_context(AgentContext(tasks=_sample_tasks(), sprint="S1",
                             sprint_start=_dt.date(2026, 6, 1),
                             sprint_end=_dt.date(2026, 6, 15),
                             today=_dt.date(2026, 6, 10)))
    kpis = json.loads(dispatch_tool("compute_sprint_kpis", ""))
    assert kpis["completed_tasks"] == 1 and kpis["total_tasks"] == 3, kpis
    risks = json.loads(dispatch_tool("detect_project_risks", ""))
    assert any(r["severity"] == "high" for r in risks), risks  # blocked T2
    print("deterministic tool dispatch -> OK")


def test_full_loop_reaches_final_answer():
    from agent.tools import AgentContext, set_context
    set_context(AgentContext(tasks=_sample_tasks(), sprint="S1",
                             sprint_start=_dt.date(2026, 6, 1),
                             sprint_end=_dt.date(2026, 6, 15),
                             today=_dt.date(2026, 6, 10)))
    # script: call KPIs, then finish
    _install_stub([
        {"thought": "Need the numbers", "tool": "compute_sprint_kpis", "tool_input": ""},
        {"thought": "Done", "tool": "final_answer",
         "tool_input": "Sprint S1: 1/3 tasks done, 1 blocked task."},
    ])
    from agent.react import run_custom_react
    result = run_custom_react("Summarize sprint KPIs and risks")
    assert "S1" in result["answer"], result
    assert result["scratchpad"][0]["tool"] == "compute_sprint_kpis"
    assert "completed_tasks" in result["scratchpad"][0]["observation"]
    print("full loop -> final_answer -> OK")


def test_step_cap_terminates():
    from agent.tools import AgentContext, set_context
    set_context(AgentContext(tasks=_sample_tasks(), sprint="S1",
                             sprint_start=_dt.date(2026, 6, 1),
                             sprint_end=_dt.date(2026, 6, 15),
                             today=_dt.date(2026, 6, 10)))
    # never emits final_answer -> must stop at the step cap
    _install_stub([
        {"thought": "loop", "tool": "detect_project_risks", "tool_input": ""},
    ])
    from agent.react import run_custom_react
    result = run_custom_react("loop forever")
    assert result["answer"].startswith("(step limit reached)"), result["answer"][:60]
    print("step-cap termination -> OK")


if __name__ == "__main__":
    test_dispatch_unknown_tool()
    test_deterministic_tool_dispatch()
    test_full_loop_reaches_final_answer()
    test_step_cap_terminates()
    print("\nALL REACT LOOP TESTS PASSED")
