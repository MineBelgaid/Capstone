"""Explicit, constrained reason -> act -> observe ReAct loop.

Why this exists: local models (qwen2.5:14b via Ollama) emit native tool-call JSON
inconsistently, which makes LangGraph's prebuilt ``create_react_agent`` flaky on
the dev backend. Instead of relying on native tool calling, each step is a single
constrained JSON object (``ReActStep``) validated by Pydantic with retry -- the
same robustness pattern already used for schema'd outputs. A hallucinated tool
name fails ``Literal`` validation and triggers a retry rather than crashing.

The loop deliberately CANNOT choose ``export_report``: the only external action is
reachable solely through the human-approved dashboard path, reinforcing the
"zero autonomous external actions" rule.

The graph nodes (reason / act) are explicit so the ReAct pattern is inspectable
for the capstone write-up and demo.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from agent.llm import get_chat_model
from agent.tools import (
    compute_sprint_kpis,
    detect_project_risks,
    retrieve_context,
    summarize_meeting,
)
from agent.validation import generate_structured
from config import settings

# Tools the autonomous loop may select (export_report intentionally excluded).
_TOOL_REGISTRY: dict[str, Any] = {
    "retrieve_context": retrieve_context,
    "summarize_meeting": summarize_meeting,
    "compute_sprint_kpis": compute_sprint_kpis,
    "detect_project_risks": detect_project_risks,
}

ToolName = Literal[
    "retrieve_context",
    "summarize_meeting",
    "compute_sprint_kpis",
    "detect_project_risks",
    "final_answer",
]


class ReActStep(BaseModel):
    """One constrained step of the loop. The whole reliability story lives here."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    thought: str = Field(..., min_length=1, description="Brief reasoning for this step")
    tool: ToolName = Field(..., description="Tool to call, or 'final_answer' to stop")
    tool_input: str = Field(
        default="",
        description="Input for the tool; for 'final_answer' this is the answer text",
    )


REACT_SYSTEM = (
    "You are a project-coordination assistant for a small software team. "
    "You operate in a strict reason-act-observe loop. At each step output exactly "
    "one JSON object with 'thought', 'tool', and 'tool_input'. "
    "Use tools to ground every answer in real project data; never invent numbers "
    "(compute_sprint_kpis and detect_project_risks return exact figures). "
    "When you have enough information, set tool to 'final_answer' and put the full "
    "answer in 'tool_input'. You cannot take external actions such as exporting; "
    "those require human approval in the dashboard."
)

_TOOL_HELP = (
    "Available tools:\n"
    "- retrieve_context(query): similarity search over project notes/tasks.\n"
    "- summarize_meeting(): structured summary + action items of loaded notes.\n"
    "- compute_sprint_kpis(): exact velocity / completion / workload.\n"
    "- detect_project_risks(): exact blocked / overdue / stale / overload list.\n"
    "- final_answer(answer): finish and return the answer."
)


def dispatch_tool(name: str, tool_input: str) -> str:
    """Run a registered tool by name. Unknown names return an error observation
    (which the model then sees and can correct on the next step)."""
    tool = _TOOL_REGISTRY.get(name)
    if tool is None:
        return f"ERROR: unknown tool '{name}'."
    try:
        return str(tool.invoke(tool_input))
    except Exception as exc:  # noqa: BLE001 - surface the error as an observation
        return f"ERROR running {name}: {exc}"


def _reason_prompt(query: str, scratchpad: list[dict]) -> str:
    lines = [_TOOL_HELP, "", f"User request: {query}", ""]
    if scratchpad:
        lines.append("Steps so far:")
        for i, s in enumerate(scratchpad, 1):
            obs = s["observation"]
            if len(obs) > 1200:  # keep context bounded for local models
                obs = obs[:1200] + " …[truncated]"
            lines.append(f"{i}. thought: {s['thought']}")
            lines.append(f"   action: {s['tool']}({s['tool_input']!r})")
            lines.append(f"   observation: {obs}")
        lines.append("")
    lines.append("Decide the next step as a single JSON object.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# LangGraph state + nodes
# --------------------------------------------------------------------------- #
class ReActState(TypedDict, total=False):
    query: str
    scratchpad: list[dict]
    steps: int
    answer: str
    proposed: dict


def _reason_node(state: ReActState) -> ReActState:
    model = get_chat_model()
    step = generate_structured(
        chat_model=model,
        schema=ReActStep,
        system_prompt=REACT_SYSTEM,
        user_prompt=_reason_prompt(state["query"], state.get("scratchpad", [])),
    )
    return {"proposed": step.model_dump()}


def _act_node(state: ReActState) -> ReActState:
    action = state["proposed"]
    scratchpad = list(state.get("scratchpad", []))
    observation = dispatch_tool(action["tool"], action["tool_input"])
    scratchpad.append({
        "thought": action["thought"],
        "tool": action["tool"],
        "tool_input": action["tool_input"],
        "observation": observation,
    })
    return {"scratchpad": scratchpad, "steps": state.get("steps", 0) + 1}


def _route_after_reason(state: ReActState) -> str:
    if state["proposed"]["tool"] == "final_answer":
        return "finalize"
    return "act"


def _route_after_act(state: ReActState) -> str:
    if state.get("steps", 0) >= settings.agent.max_react_steps:
        return "finalize"
    return "reason"


def _finalize_node(state: ReActState) -> ReActState:
    action = state.get("proposed", {})
    if action.get("tool") == "final_answer" and action.get("tool_input"):
        return {"answer": action["tool_input"]}
    # Hit the step cap without an explicit answer: summarize observations.
    obs = "\n".join(s["observation"] for s in state.get("scratchpad", []))
    return {"answer": f"(step limit reached) Collected findings:\n{obs}"}


def build_custom_react_graph():
    """Compile the explicit reason/act/observe graph."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(ReActState)
    g.add_node("reason", _reason_node)
    g.add_node("act", _act_node)
    g.add_node("finalize", _finalize_node)
    g.add_edge(START, "reason")
    g.add_conditional_edges("reason", _route_after_reason,
                            {"act": "act", "finalize": "finalize"})
    g.add_conditional_edges("act", _route_after_act,
                            {"reason": "reason", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


def run_custom_react(query: str) -> dict:
    """Run the custom loop. Assumes the per-run AgentContext is already set."""
    graph = build_custom_react_graph()
    final = graph.invoke(
        {"query": query, "scratchpad": [], "steps": 0},
        config={"recursion_limit": settings.agent.max_react_steps * 3 + 5},
    )
    return {"answer": final.get("answer", ""), "scratchpad": final.get("scratchpad", [])}
