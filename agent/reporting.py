"""Assemble a weekly StatusReport from deterministic KPIs/risks + an LLM narrative.

The structured numbers come from ``analytics`` (always schema-valid); only the
headline and prose narrative are LLM-generated, through the validation wrapper.
This keeps SprintKPIReport validity at 100% while still demonstrating generation.
"""

from __future__ import annotations

import datetime as _dt

from pydantic import BaseModel, ConfigDict, Field

from agent.analytics import compute_kpis, detect_risks
from agent.llm import get_chat_model
from agent.validation import generate_structured
from schemas import ActionItem, StatusReport, Task


class _Narrative(BaseModel):
    """Small LLM-only payload merged into the deterministic StatusReport."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    headline: str = Field(..., min_length=10)
    narrative: str = Field(..., min_length=20)


def build_status_report(
    tasks: list[Task],
    sprint: str,
    sprint_start: _dt.date,
    sprint_end: _dt.date,
    recent_action_items: list[ActionItem] | None = None,
    today: _dt.date | None = None,
    use_llm: bool = True,
) -> StatusReport:
    today = today or _dt.date.today()
    kpis = compute_kpis(tasks, sprint, sprint_start, sprint_end)
    risks = detect_risks(tasks, today=today)
    action_items = recent_action_items or []

    if use_llm:
        narrative = _llm_narrative(kpis, risks, action_items)
    else:
        narrative = _Narrative(
            headline=f"{sprint}: {kpis.completion_rate:.0%} complete, {len(risks)} risk(s).",
            narrative=(
                f"Velocity {kpis.velocity:.0f} pts across {kpis.total_tasks} tasks "
                f"({kpis.completed_tasks} done). {len(risks)} active risk(s) flagged."
            ),
        )

    return StatusReport(
        title=f"Weekly Status -- {sprint}",
        period_start=sprint_start,
        period_end=sprint_end,
        headline=narrative.headline,
        kpis=kpis,
        risks=risks,
        recent_action_items=action_items,
        narrative=narrative.narrative,
    )


def _llm_narrative(kpis, risks, action_items) -> _Narrative:
    model = get_chat_model()
    risks_txt = "\n".join(f"- [{r.severity.value}] {r.area}: {r.reason}" for r in risks) or "None"
    ai_txt = "\n".join(f"- {a.description} (owner {a.owner})" for a in action_items) or "None"
    user_prompt = (
        f"Sprint: {kpis.sprint} ({kpis.sprint_start} to {kpis.sprint_end})\n"
        f"Velocity: {kpis.velocity} pts | Completion: {kpis.completion_rate:.0%} "
        f"({kpis.completed_tasks}/{kpis.total_tasks} tasks)\n"
        f"Risks:\n{risks_txt}\n"
        f"Recent action items:\n{ai_txt}\n\n"
        "Write a one-line executive 'headline' and a short prose 'narrative' "
        "(3-5 sentences) summarizing status, progress and the most important risks "
        "for a small team's weekly digest. Do not invent numbers."
    )
    return generate_structured(
        chat_model=model,
        schema=_Narrative,
        system_prompt="You are a concise engineering project manager.",
        user_prompt=user_prompt,
    )


def status_report_to_markdown(report: StatusReport) -> str:
    """Render a StatusReport as Markdown (used by the gated export tool / dashboard)."""
    lines = [
        f"# {report.title}",
        f"_{report.period_start} to {report.period_end}_",
        "",
        f"**{report.headline}**",
        "",
        report.narrative,
        "",
        "## KPIs",
        f"- Velocity: {report.kpis.velocity:.0f} pts",
        f"- Completion: {report.kpis.completion_rate:.0%} "
        f"({report.kpis.completed_tasks}/{report.kpis.total_tasks} tasks)",
        "",
        "### Workload",
    ]
    for w in report.kpis.workload:
        lines.append(f"- {w.member}: {w.assigned_tasks} tasks, "
                     f"{w.assigned_points:.0f} pts, {w.done_tasks} done")
    lines += ["", "## Risks"]
    if report.risks:
        for r in report.risks:
            lines.append(f"- **[{r.severity.value}]** {r.area} — {r.reason} "
                         f"_→ {r.recommended_action}_")
    else:
        lines.append("- None flagged.")
    lines += ["", "## Recent Action Items"]
    if report.recent_action_items:
        for a in report.recent_action_items:
            due = f" (due {a.deadline})" if a.deadline else ""
            lines.append(f"- {a.description} — **{a.owner}**{due}")
    else:
        lines.append("- None.")
    return "\n".join(lines)
