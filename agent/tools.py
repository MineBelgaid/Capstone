"""Agent tools -- each capability is a distinct tool the ReAct loop can select.

Tools operate over a per-run ``AgentContext`` (the normalized tasks, the ChromaDB
store, sprint metadata). Deterministic work (KPIs, risk signals) is done in
``analytics``; the LLM is used only for narration/summarization through the
validation wrapper.

HARD RULE: the only tool that performs an *external* action -- ``export_report``
-- refuses to write anything unless the run has been explicitly human-approved.
This enforces the brief's "zero autonomous external actions" requirement.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.tools import tool

from agent.analytics import compute_kpis, detect_risks
from agent.llm import get_chat_model
from agent.validation import generate_structured
from config import settings
from schemas import MeetingSummary, Task


# --------------------------------------------------------------------------- #
# Per-run context (set before invoking the agent)
# --------------------------------------------------------------------------- #
@dataclass
class AgentContext:
    tasks: list[Task] = field(default_factory=list)
    store: object | None = None          # retrieval.KnowledgeStore (optional)
    meeting_text: str | None = None
    meeting_source: str = "meeting_notes"
    meeting_date_hint: str | None = None
    sprint: str = "Sprint"
    sprint_start: _dt.date = field(default_factory=_dt.date.today)
    sprint_end: _dt.date = field(default_factory=_dt.date.today)
    today: _dt.date = field(default_factory=_dt.date.today)
    # Human-in-the-loop gate. The dashboard flips this to True on approval.
    human_approved: bool = False


_CTX: AgentContext = AgentContext()


def set_context(ctx: AgentContext) -> None:
    global _CTX
    _CTX = ctx


def get_context() -> AgentContext:
    return _CTX


# --------------------------------------------------------------------------- #
# RAG tool
# --------------------------------------------------------------------------- #
@tool
def retrieve_context(query: str) -> str:
    """Retrieve the most relevant project notes/tasks for a query via similarity
    search over the local ChromaDB store. Use before summarizing or reporting to
    ground answers in actual project data."""
    if _CTX.store is None:
        return "No knowledge store configured for this run."
    chunks = _CTX.store.query(query)
    if not chunks:
        return "No relevant context found."
    return "\n\n".join(
        f"[{c.metadata.get('doc_type', '?')}] {c.text}" for c in chunks
    )


# --------------------------------------------------------------------------- #
# Meeting summarizer + action-item extractor
# --------------------------------------------------------------------------- #
@tool
def summarize_meeting(_: str = "") -> str:
    """Summarize the current meeting notes into a structured MeetingSummary with
    key decisions and action items (owner + deadline + source). Returns JSON."""
    if not _CTX.meeting_text:
        return json.dumps({"error": "No meeting notes loaded for this run."})
    model = get_chat_model()
    user_prompt = (
        f"Meeting source: {_CTX.meeting_source}\n"
        f"Date hint (may be empty): {_CTX.meeting_date_hint or ''}\n\n"
        f"Meeting notes:\n{_CTX.meeting_text}\n\n"
        "Extract attendees, key decisions, and concrete action items. "
        "Each action item needs a real owner (a named person) and, if stated, a "
        f"deadline as YYYY-MM-DD. Use '{_CTX.meeting_source}' as each item's source."
    )
    summary = generate_structured(
        chat_model=model,
        schema=MeetingSummary,
        system_prompt="You are a precise meeting-notes analyst.",
        user_prompt=user_prompt,
    )
    return summary.model_dump_json()


# --------------------------------------------------------------------------- #
# KPI / sprint analytics  (deterministic)
# --------------------------------------------------------------------------- #
@tool
def compute_sprint_kpis(_: str = "") -> str:
    """Compute sprint KPIs (velocity, completion rate, per-member workload) from
    the current tasks. Deterministic -- numbers are calculated, not estimated.
    Returns JSON matching SprintKPIReport."""
    report = compute_kpis(
        tasks=_CTX.tasks,
        sprint=_CTX.sprint,
        sprint_start=_CTX.sprint_start,
        sprint_end=_CTX.sprint_end,
    )
    return report.model_dump_json()


# --------------------------------------------------------------------------- #
# Risk / bottleneck detector  (deterministic)
# --------------------------------------------------------------------------- #
@tool
def detect_project_risks(_: str = "") -> str:
    """Detect risks/bottlenecks (blocked, overdue/at-deadline, stale, overloaded)
    from the current tasks. Deterministic. Returns a JSON list of RiskAlert."""
    risks = detect_risks(_CTX.tasks, today=_CTX.today)
    return json.dumps([r.model_dump(mode="json") for r in risks])


# --------------------------------------------------------------------------- #
# External action -- HUMAN-APPROVAL GATED
# --------------------------------------------------------------------------- #
@tool
def export_report(markdown: str) -> str:
    """Export an approved report to a Markdown file on disk. This is an EXTERNAL
    action: it will refuse to write unless the run has been human-approved in the
    review dashboard."""
    if settings.require_human_approval and not _CTX.human_approved:
        return (
            "BLOCKED: export requires human approval. The report is staged in the "
            "review dashboard and was NOT written. Approve it there to export."
        )
    out_dir = Path("data/exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"status_report-{stamp}.md"
    path.write_text(markdown, encoding="utf-8")
    return f"Exported approved report to {path}"


# Registry the graph binds to the model.
ALL_TOOLS = [
    retrieve_context,
    summarize_meeting,
    compute_sprint_kpis,
    detect_project_risks,
    export_report,
]
