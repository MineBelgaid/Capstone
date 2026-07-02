"""Deterministic project analytics computed in pure Python.

KPIs and risk signals are computed from the normalized tasks here -- NOT
hallucinated by the LLM. The LLM's job is to narrate/explain these numbers, which
keeps the "KPI report schema validity = 100%" and "risk precision > 85%" targets
achievable and reproducible. Tools wrap these functions.
"""

from __future__ import annotations

import datetime as _dt
from collections import defaultdict

from schemas import (
    RiskAlert,
    RiskSeverity,
    SprintKPIReport,
    Task,
    TaskStatus,
    WorkloadEntry,
)

# Tunable thresholds for risk detection.
STALE_DAYS = 5             # no update in N days while not done -> stale
DUE_SOON_DAYS = 3          # due within N days and not done -> deadline risk
OVERLOAD_POINT_RATIO = 1.5  # member points > ratio * team mean -> overloaded


def compute_kpis(
    tasks: list[Task],
    sprint: str,
    sprint_start: _dt.date,
    sprint_end: _dt.date,
) -> SprintKPIReport:
    total = len(tasks)
    completed = [t for t in tasks if t.status is TaskStatus.DONE]
    velocity = sum((t.story_points or 0.0) for t in completed)
    completion_rate = (len(completed) / total) if total else 0.0

    by_member: dict[str, dict[str, float]] = defaultdict(
        lambda: {"assigned": 0, "points": 0.0, "done": 0}
    )
    for t in tasks:
        member = t.assignee or "Unassigned"
        by_member[member]["assigned"] += 1
        by_member[member]["points"] += t.story_points or 0.0
        if t.status is TaskStatus.DONE:
            by_member[member]["done"] += 1

    workload = [
        WorkloadEntry(
            member=m,
            assigned_tasks=int(v["assigned"]),
            assigned_points=float(v["points"]),
            done_tasks=int(v["done"]),
        )
        for m, v in sorted(by_member.items())
    ] or [WorkloadEntry(member="Unassigned", assigned_tasks=0, assigned_points=0.0, done_tasks=0)]

    return SprintKPIReport(
        sprint=sprint,
        sprint_start=sprint_start,
        sprint_end=sprint_end,
        velocity=float(velocity),
        completion_rate=round(completion_rate, 4),
        total_tasks=total,
        completed_tasks=len(completed),
        workload=workload,
    )


def open_points_by_member(tasks: list[Task]) -> dict[str, float]:
    """Open (not done/cancelled) story points per member. Deterministic."""
    out: dict[str, float] = defaultdict(float)
    for t in tasks:
        if t.status in {TaskStatus.DONE, TaskStatus.CANCELLED}:
            continue
        out[t.assignee or "Unassigned"] += t.story_points or 0.0
    return dict(out)


def analyze_workload_balance(tasks: list[Task]) -> dict:
    """Deterministic workload analysis feeding the rebalancing proposal.

    Identifies who is overloaded (open points well above the team mean), who has
    spare capacity, and which open tasks are candidates to move (owned by an
    overloaded member, or unassigned). The LLM proposes moves ONLY from this
    candidate set -- so proposals stay grounded in real, movable work."""
    open_pts = open_points_by_member(tasks)
    open_cnt: dict[str, int] = defaultdict(int)
    for t in tasks:
        if t.status in {TaskStatus.DONE, TaskStatus.CANCELLED}:
            continue
        open_cnt[t.assignee or "Unassigned"] += 1

    real = {m: p for m, p in open_pts.items() if m != "Unassigned"}
    mean = (sum(real.values()) / len(real)) if real else 0.0
    overloaded = sorted(
        [m for m, p in real.items() if mean > 0 and p > OVERLOAD_POINT_RATIO * mean],
        key=lambda m: -real[m],
    )
    underloaded = sorted([m for m, p in real.items() if p < mean], key=lambda m: real[m])

    movable: list[dict] = []
    for t in tasks:
        if t.status in {TaskStatus.DONE, TaskStatus.CANCELLED}:
            continue
        owner = t.assignee or "Unassigned"
        if owner in overloaded or owner == "Unassigned":
            movable.append({
                "task_id": t.task_id, "title": t.title, "owner": owner,
                "points": float(t.story_points or 0.0), "status": t.status.value,
            })
    return {
        "mean_open_points": round(mean, 2),
        "load": {m: {"open_points": round(open_pts[m], 2), "open_tasks": open_cnt[m]}
                 for m in sorted(open_pts)},
        "overloaded": overloaded,
        "underloaded": underloaded,
        "movable_tasks": movable,
    }


def detect_risks(tasks: list[Task], today: _dt.date | None = None) -> list[RiskAlert]:
    """Flag blocked tasks, stale/at-deadline tasks, and overloaded members."""
    today = today or _dt.date.today()
    alerts: list[RiskAlert] = []

    # 1) explicitly blocked tasks
    for t in tasks:
        if t.status is TaskStatus.BLOCKED:
            alerts.append(RiskAlert(
                area=f"{t.task_id}: {t.title}",
                severity=RiskSeverity.HIGH,
                reason="Task is marked blocked.",
                recommended_action="Identify and remove the blocker; escalate if external.",
                related_task_ids=[t.task_id],
            ))

    # 2) deadline risk: due soon / overdue and not done
    for t in tasks:
        if t.status in {TaskStatus.DONE, TaskStatus.CANCELLED} or not t.due_date:
            continue
        days_left = (t.due_date - today).days
        if days_left < 0:
            alerts.append(RiskAlert(
                area=f"{t.task_id}: {t.title}",
                severity=RiskSeverity.CRITICAL,
                reason=f"Overdue by {abs(days_left)} day(s) and not done.",
                recommended_action="Reassess scope/owner; replan or de-scope immediately.",
                related_task_ids=[t.task_id],
            ))
        elif days_left <= DUE_SOON_DAYS:
            alerts.append(RiskAlert(
                area=f"{t.task_id}: {t.title}",
                severity=RiskSeverity.MEDIUM,
                reason=f"Due in {days_left} day(s) and not yet done.",
                recommended_action="Confirm it is on track; surface blockers early.",
                related_task_ids=[t.task_id],
            ))

    # 3) stale tasks: in progress but no recent activity
    for t in tasks:
        if t.status is not TaskStatus.IN_PROGRESS or not t.updated_at:
            continue
        idle = (today - t.updated_at).days
        if idle >= STALE_DAYS:
            alerts.append(RiskAlert(
                area=f"{t.task_id}: {t.title}",
                severity=RiskSeverity.MEDIUM,
                reason=f"In progress but no activity for {idle} day(s).",
                recommended_action="Check in with the owner; confirm it is not silently stuck.",
                related_task_ids=[t.task_id],
            ))

    # 4) overloaded members (open-work points well above team mean)
    open_points: dict[str, float] = defaultdict(float)
    for t in tasks:
        if t.status in {TaskStatus.DONE, TaskStatus.CANCELLED}:
            continue
        open_points[t.assignee or "Unassigned"] += t.story_points or 0.0
    real = {m: p for m, p in open_points.items() if m != "Unassigned"}
    if real:
        mean = sum(real.values()) / len(real)
        for member, pts in real.items():
            if mean > 0 and pts > OVERLOAD_POINT_RATIO * mean:
                alerts.append(RiskAlert(
                    area=f"Workload: {member}",
                    severity=RiskSeverity.MEDIUM,
                    reason=f"{member} holds {pts:.0f} open pts vs team mean {mean:.1f}.",
                    recommended_action="Rebalance assignments across the team.",
                    related_task_ids=[],
                ))

    severity_rank = {RiskSeverity.CRITICAL: 0, RiskSeverity.HIGH: 1,
                     RiskSeverity.MEDIUM: 2, RiskSeverity.LOW: 3}
    alerts.sort(key=lambda a: severity_rank[a.severity])
    return alerts
