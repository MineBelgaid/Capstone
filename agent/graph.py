"""LangGraph single-agent ReAct loop.

A reason -> act -> observe cycle: the model reasons over retrieved context,
selects a tool, the tool executes, the model observes the result and either calls
another tool or finalizes. The loop is hard-capped at ``max_react_steps``.

We use LangGraph's prebuilt tool-calling ReAct agent (works with Qwen2.5 / Llama
3.1 tool calling via Ollama, and with Claude for the demo). The deterministic
analytics/report builders are also callable directly (see dashboard / eval) so
exact numbers never depend on the LLM choosing the right tool.
"""

from __future__ import annotations

from agent.llm import get_chat_model
from agent.tools import ALL_TOOLS, AgentContext, set_context
from config import settings

SYSTEM_PROMPT = (
    "You are a project-coordination assistant for a small software team. "
    "Reason step by step. Use the available tools to ground every answer in real "
    "project data: retrieve_context for relevant notes, compute_sprint_kpis and "
    "detect_project_risks for metrics (these are exact -- never estimate numbers "
    "yourself), and summarize_meeting for meeting notes. "
    "You must NOT take any external action (such as export_report) on your own. "
    "Exports require explicit human approval and will be blocked otherwise. "
    "When you have enough information, give a clear final answer."
)


def build_agent():
    """Construct the compiled ReAct graph bound to the configured tools."""
    from langgraph.prebuilt import create_react_agent

    model = get_chat_model()
    return create_react_agent(model, ALL_TOOLS, prompt=SYSTEM_PROMPT)


def run_agent(query: str, context: AgentContext) -> dict:
    """Run one query against the agent with a per-run context.

    Dispatches on ``settings.agent.react_mode``:
      * "custom"   -> explicit constrained reason/act/observe loop (robust on
                      local Ollama models).
      * "prebuilt" -> LangGraph native tool-calling ReAct agent (best on Claude).

    Returns a dict with the final answer text and the step trace (the trace shape
    differs by mode but both expose ``answer``), handy for the dashboard and for
    LangSmith review.
    """
    set_context(context)

    if settings.agent.react_mode == "custom":
        from agent.react import run_custom_react

        result = run_custom_react(query)
        return {"answer": result["answer"], "scratchpad": result["scratchpad"],
                "mode": "custom"}

    # prebuilt native tool-calling path
    from langchain_core.messages import HumanMessage

    agent = build_agent()
    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": settings.agent.max_react_steps * 2 + 1},
    )
    messages = result.get("messages", [])
    final = messages[-1].content if messages else ""
    return {"answer": final, "messages": messages, "mode": "prebuilt"}
