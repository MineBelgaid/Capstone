"""Curated, documented test-scenario catalog.

Unlike the random generator (``eval/generate.py``), these scenarios are
hand-designed so each one has a *clear, explainable purpose* -- ideal for the
demo ("here is the absentee-developer case, and the agent catches it") and for
defensible grading. Every scenario carries by-construction ground truth:

  * ``ground_truth.risky_tasks`` -- the tasks deliberately made risky, with the
    reason. Assigned independently of the detector, so precision/recall are real.
  * ``ground_truth.expected_kpis`` -- recomputed here independently of the
    production analytics, so KPI numeric accuracy is a genuine cross-check.

Normal ("clean") tasks are kept genuinely healthy (recent activity, far
deadline, not blocked) so ONLY the intended conditions read as risky. This keeps
risk precision honest and is what the negative-control scenarios verify.

Output format is identical to the generator's, so ``eval/score.py`` reads both.

Run:  python -m eval.catalog          # writes data/scenarios/ and data/meetings/
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path

from config import DATA_DIR

SCENARIO_DIR = DATA_DIR / "scenarios"
MEETING_DIR = DATA_DIR / "meetings"

# A fixed reference "today" makes the catalog fully reproducible.
TODAY = _dt.date(2026, 6, 22)

NAMES = [
    "Ava", "Ben", "Carla", "Dmitri", "Elena", "Farid", "Grace", "Hassan",
    "Ines", "Jonas", "Kira", "Leo", "Mara", "Noah", "Omar",
]


def _iso(d: _dt.date) -> str:
    return d.isoformat()


# --------------------------------------------------------------------------- #
# Sprint scenario builder
# --------------------------------------------------------------------------- #
@dataclass
class SprintBuilder:
    scenario_id: str
    intent: str
    team_size: int
    today: _dt.date = TODAY
    days_into_sprint: int = 7
    sprint_len: int = 14
    tasks: list[dict] = field(default_factory=list)
    risky: list[dict] = field(default_factory=list)
    _n: int = 0

    def __post_init__(self) -> None:
        self.team = NAMES[: self.team_size]
        self.sprint = self.scenario_id
        self.sprint_start = self.today - _dt.timedelta(days=self.days_into_sprint)
        self.sprint_end = self.sprint_start + _dt.timedelta(days=self.sprint_len)

    # -- internal helper -------------------------------------------------- #
    def _add(self, title: str, assignee: str, status: str, points: float,
             created: _dt.date, updated: _dt.date, due: _dt.date,
             labels: list[str], risk: str | None) -> str:
        self._n += 1
        tid = f"{self.sprint}-T{self._n:02d}"
        updated = max(updated, created)  # schema invariant
        self.tasks.append({
            "task_id": tid, "title": title, "status": status, "assignee": assignee,
            "story_points": float(points), "created_at": _iso(created),
            "updated_at": _iso(updated), "due_date": _iso(due), "sprint": self.sprint,
            "labels": labels, "source_system": "curated",
        })
        if risk:
            self.risky.append({"task_id": tid, "reason": risk})
        return tid

    # -- public, readable task constructors ------------------------------- #
    def clean(self, title: str, assignee: str, status: str = "in_progress",
              points: float = 3, labels: list[str] | None = None) -> str:
        """A healthy task: recent activity, far deadline, not flagged."""
        return self._add(
            title, assignee, status, points,
            created=self.sprint_start,
            updated=self.today - _dt.timedelta(days=1),
            due=self.sprint_end, labels=labels or [], risk=None,
        )

    def done(self, title: str, assignee: str, points: float = 3) -> str:
        return self._add(
            title, assignee, "done", points,
            created=self.sprint_start,
            updated=self.today - _dt.timedelta(days=2),
            due=self.sprint_end, labels=[], risk=None,
        )

    def blocked(self, title: str, assignee: str, points: float = 3) -> str:
        return self._add(
            title, assignee, "blocked", points,
            created=self.sprint_start,
            updated=self.today - _dt.timedelta(days=1),
            due=self.sprint_end, labels=["bug"], risk="blocked",
        )

    def overdue(self, title: str, assignee: str, days_over: int = 2,
                points: float = 3) -> str:
        return self._add(
            title, assignee, "in_progress", points,
            created=self.sprint_start,
            updated=self.today - _dt.timedelta(days=1),
            due=self.today - _dt.timedelta(days=days_over),
            labels=[], risk="overdue",
        )

    def due_soon(self, title: str, assignee: str, days_left: int = 2,
                 points: float = 3, status: str = "todo") -> str:
        return self._add(
            title, assignee, status, points,
            created=self.sprint_start,
            updated=self.today - _dt.timedelta(days=1),
            due=self.today + _dt.timedelta(days=days_left),
            labels=[], risk="due_soon",
        )

    def stale(self, title: str, assignee: str, idle_days: int = 7,
              points: float = 3) -> str:
        """In-progress but untouched -- the absentee/stalled signal."""
        return self._add(
            title, assignee, "in_progress", points,
            created=self.today - _dt.timedelta(days=idle_days + 5),
            updated=self.today - _dt.timedelta(days=idle_days),
            due=self.sprint_end, labels=[], risk="stale",
        )

    def scope_creep(self, title: str, assignee: str, points: float = 5) -> str:
        """Unplanned task added late in the sprint (still clean re: risk flags)."""
        return self._add(
            "Unplanned: " + title, assignee, "todo", points,
            created=self.today - _dt.timedelta(days=1),
            updated=self.today - _dt.timedelta(days=1),
            due=self.sprint_end, labels=["scope-creep"], risk=None,
        )

    # -- serialization ---------------------------------------------------- #
    def to_dict(self) -> dict:
        done = [t for t in self.tasks if t["status"] == "done"]
        total = len(self.tasks)
        expected_kpis = {
            "velocity": sum(t["story_points"] for t in done),
            "completed_tasks": len(done),
            "total_tasks": total,
            "completion_rate": round(len(done) / total, 4) if total else 0.0,
        }
        return {
            "scenario_id": self.scenario_id,
            "intent": self.intent,
            "today": _iso(self.today),
            "team_size": self.team_size,
            "team": self.team,
            "sprint": self.sprint,
            "sprint_start": _iso(self.sprint_start),
            "sprint_end": _iso(self.sprint_end),
            "tasks": self.tasks,
            "meeting_note": "",          # sprint scenarios: notes live in data/meetings
            "ground_truth": {
                "risky_tasks": self.risky,
                "expected_kpis": expected_kpis,
                "action_items": [],
            },
        }


# --------------------------------------------------------------------------- #
# The curated scenarios (team sizes span 3..15)
# --------------------------------------------------------------------------- #
def _healthy_small() -> dict:
    b = SprintBuilder("CUR-01-healthy-small", "Negative control: a healthy 3-person "
                      "sprint with no risks. Verifies the detector does not cry wolf.", 3)
    b.done("Set up CI", "Ava", 3); b.done("Auth scaffolding", "Ben", 5)
    b.clean("Login UI", "Carla", points=3); b.clean("API contract", "Ava", points=2)
    b.clean("Unit tests", "Ben", "todo", 2)
    return b.to_dict()


def _single_blocker() -> dict:
    b = SprintBuilder("CUR-02-single-blocker", "One hard-blocked task in an otherwise "
                      "healthy 4-person sprint.", 4)
    b.done("Payment model", "Dmitri", 5); b.clean("Checkout flow", "Ava", points=5)
    b.blocked("3rd-party payment integration", "Ben", 8)
    b.clean("Receipts email", "Carla", "todo", 3); b.done("Pricing config", "Ava", 2)
    return b.to_dict()


def _blocker_chain() -> dict:
    b = SprintBuilder("CUR-03-bottleneck", "Bottleneck: several blocked tasks stalling "
                      "a 5-person team.", 5)
    b.blocked("DB migration", "Elena", 8); b.blocked("Schema review", "Ben", 3)
    b.blocked("Data backfill", "Carla", 5); b.clean("Frontend polish", "Ava", points=3)
    b.done("Docs update", "Dmitri", 2); b.clean("Logging", "Elena", "todo", 2)
    return b.to_dict()


def _deadline_slip_minor() -> dict:
    b = SprintBuilder("CUR-04-deadline-slip-minor", "A couple of tasks due in 1-2 days "
                      "and not done -- early-warning deadline risk.", 4)
    b.done("Settings page", "Ava", 3); b.clean("Profile API", "Ben", points=5)
    b.due_soon("Export feature", "Carla", days_left=1, points=5)
    b.due_soon("Email templates", "Dmitri", days_left=2, points=3, status="in_progress")
    b.clean("Analytics wiring", "Ava", "todo", 2)
    return b.to_dict()


def _deadline_slip_severe() -> dict:
    b = SprintBuilder("CUR-05-deadline-slip-severe", "Multiple overdue tasks: the "
                      "'simulated delays' the risk-precision target is about.", 6)
    b.overdue("Search backend", "Elena", days_over=3, points=8)
    b.overdue("Cache layer", "Farid", days_over=1, points=5)
    b.overdue("Rate limiting", "Ben", days_over=2, points=3)
    b.done("Health checks", "Ava", 2); b.clean("Dashboard", "Carla", points=5)
    b.clean("Onboarding", "Dmitri", "todo", 3)
    return b.to_dict()


def _scope_creep() -> dict:
    b = SprintBuilder("CUR-06-scope-creep", "Unplanned tasks injected mid-sprint, "
                      "depressing completion rate for a 5-person team.", 5)
    b.done("Core feature A", "Ava", 5); b.done("Core feature B", "Ben", 5)
    b.clean("Core feature C", "Carla", points=5)
    b.scope_creep("urgent client request", "Dmitri", 8)
    b.scope_creep("compliance checkbox", "Elena", 5)
    b.scope_creep("hotfix follow-up", "Ava", 3)
    return b.to_dict()


def _absentee_dev() -> dict:
    b = SprintBuilder("CUR-07-absentee", "A developer goes inactive: their in-progress "
                      "tasks all go stale (no recent activity).", 6)
    b.stale("Notifications service", "Farid", idle_days=8, points=8)
    b.stale("Webhook handler", "Farid", idle_days=7, points=5)
    b.stale("Retry logic", "Farid", idle_days=9, points=3)
    b.clean("Frontend", "Ava", points=3); b.done("Infra", "Ben", 3)
    b.clean("QA pass", "Carla", "todo", 2)
    return b.to_dict()


def _absentee_plus_deadline() -> dict:
    b = SprintBuilder("CUR-08-absentee-deadline", "Combo: an absentee's stale work is "
                      "ALSO overdue -- compounded risk on a 7-person team.", 7)
    b.stale("Billing reconciliation", "Grace", idle_days=8, points=8)
    b.overdue("Invoice export", "Grace", days_over=2, points=5)
    b.blocked("Tax rules engine", "Ben", 5)
    b.done("Currency support", "Ava", 3); b.clean("Refunds", "Carla", points=5)
    b.clean("Audit log", "Dmitri", "todo", 2); b.clean("Reports", "Elena", points=3)
    return b.to_dict()


def _overloaded_lead() -> dict:
    b = SprintBuilder("CUR-09-overloaded-lead", "Workload imbalance: one member holds "
                      "most of the open points on an 8-person team.", 8)
    for i in range(5):
        b.clean(f"Lead task {i+1}", "Ava", "todo" if i % 2 else "in_progress", points=8)
    b.done("Helper task", "Ben", 2); b.clean("Small fix", "Carla", points=2)
    b.clean("Tiny tweak", "Dmitri", "todo", 1)
    return b.to_dict()


def _large_team_mixed() -> dict:
    b = SprintBuilder("CUR-10-large-mixed", "Larger 12-person team with a realistic "
                      "mix: a blocker, an overdue task, and scope creep.", 12)
    b.blocked("SSO integration", "Kira", 8)
    b.overdue("Migration scripts", "Leo", days_over=2, points=5)
    b.scope_creep("exec demo request", "Mara", 5)
    for i, who in enumerate(["Ava", "Ben", "Carla", "Dmitri", "Elena", "Farid"]):
        b.done(f"Module {i+1}", who, 3)
    b.clean("Polish pass", "Grace", points=3); b.clean("Docs", "Hassan", "todo", 2)
    return b.to_dict()


def _very_large_team() -> dict:
    b = SprintBuilder("CUR-11-very-large", "Scale test: 15-person team, mostly healthy "
                      "with a single overdue task. Also a workflow-time check.", 15)
    for i, who in enumerate(NAMES[:12]):
        b.done(f"Workstream {i+1}", who, 3)
    b.overdue("Cross-team dependency", "Mara", days_over=1, points=5)
    b.clean("Integration", "Noah", points=3); b.clean("Release notes", "Omar", "todo", 2)
    return b.to_dict()


def _end_of_sprint_crunch() -> dict:
    # sprint ends tomorrow -> every open task is 'due soon'
    b = SprintBuilder("CUR-12-crunch", "End-of-sprint crunch: sprint ends tomorrow and "
                      "many tasks are still open -- broad deadline risk.", 5,
                      days_into_sprint=13)
    b.done("Feature A", "Ava", 5)
    b.due_soon("Feature B", "Ben", days_left=1, points=5, status="in_progress")
    b.due_soon("Feature C", "Carla", days_left=1, points=3, status="todo")
    b.due_soon("Feature D", "Dmitri", days_left=1, points=8, status="in_progress")
    b.due_soon("QA sign-off", "Elena", days_left=1, points=2, status="todo")
    return b.to_dict()


def _blocked_and_scope_creep() -> dict:
    b = SprintBuilder("CUR-13-blocked-scope", "Combo: a blocker plus mid-sprint scope "
                      "creep on a 6-person team.", 6)
    b.blocked("Vendor API access", "Farid", 8)
    b.scope_creep("security finding", "Ava", 5); b.scope_creep("urgent bugfix", "Ben", 3)
    b.done("Baseline feature", "Carla", 5); b.clean("Refactor", "Dmitri", points=3)
    b.clean("Tests", "Elena", "todo", 2)
    return b.to_dict()


CATALOG = [
    _healthy_small, _single_blocker, _blocker_chain, _deadline_slip_minor,
    _deadline_slip_severe, _scope_creep, _absentee_dev, _absentee_plus_deadline,
    _overloaded_lead, _large_team_mixed, _very_large_team, _end_of_sprint_crunch,
    _blocked_and_scope_creep,
]


def write_catalog(out_dir: Path = SCENARIO_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for builder in CATALOG:
        scenario = builder()
        path = out_dir / f"{scenario['scenario_id']}.json"
        path.write_text(json.dumps(scenario, indent=2), encoding="utf-8")
        paths.append(path)
    return paths


def write_meetings(out_dir: Path = MEETING_DIR) -> list[Path]:
    from eval.meetings import MEETINGS

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for m in MEETINGS:
        path = out_dir / f"{m['scenario_id']}.json"
        path.write_text(json.dumps(m, indent=2), encoding="utf-8")
        paths.append(path)
    return paths


def main() -> None:
    sp = write_catalog()
    mp = write_meetings()
    print(f"Wrote {len(sp)} curated scenarios to {SCENARIO_DIR}")
    print(f"Wrote {len(mp)} meeting-note tests to {MEETING_DIR}")


if __name__ == "__main__":
    main()
