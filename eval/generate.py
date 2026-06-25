"""Synthetic sprint scenario generator with known ground truth.

Each scenario contains:
  * a list of normalized tasks (varying team size 3-15, with injected edge cases),
  * a meeting note whose action items are known verbatim,
  * a ``ground_truth`` block: the tasks deliberately made risky (with the reason),
    the expected KPI figures, and the planted action items.

Because risk labels are assigned *by construction* (independently of the
detector), the scoring harness can measure real precision/recall, not a tautology.
Deterministic seeding makes the whole suite reproducible.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
from pathlib import Path

from config import SYNTHETIC_DIR

NAMES = [
    "Ava", "Ben", "Carla", "Dmitri", "Elena", "Farid", "Grace", "Hassan",
    "Ines", "Jonas", "Kira", "Leo", "Mara", "Noah", "Omar",
]

TASK_TITLES = [
    "Implement login flow", "Refactor payment module", "Write API docs",
    "Fix flaky test suite", "Design onboarding screens", "Set up CI pipeline",
    "Optimize DB queries", "Add audit logging", "Migrate to Pydantic v2",
    "Build settings page", "Patch security advisory", "Integrate analytics",
    "Improve error handling", "Add rate limiting", "Create admin dashboard",
    "Localize UI strings", "Upgrade dependencies", "Write load tests",
]

# Edge-case categories injected per scenario.
EDGE_CASES = ["blocked", "overdue", "stale", "overloaded", "scope_creep", "absenteeism"]


def _iso(d: _dt.date) -> str:
    return d.isoformat()


def generate_scenario(idx: int, rng: random.Random, today: _dt.date) -> dict:
    team_size = rng.randint(3, 15)
    team = rng.sample(NAMES, team_size)
    # Keep sprint_end comfortably beyond today so a normal task due at sprint_end
    # is NOT flagged "due soon" -- only injected delays should read as risky.
    sprint_start = today - _dt.timedelta(days=rng.randint(2, 5))
    sprint_end = sprint_start + _dt.timedelta(days=14)
    sprint = f"Sprint-{idx:02d}"

    n_tasks = rng.randint(team_size, team_size * 3)
    tasks: list[dict] = []
    risky_truth: list[dict] = []

    # pick which edge cases this scenario exhibits (always >= 2 for signal)
    active_edges = set(rng.sample(EDGE_CASES, rng.randint(2, 4)))

    overloaded_member = rng.choice(team) if "overloaded" in active_edges else None
    absent_member = rng.choice(team) if "absenteeism" in active_edges else None

    for t in range(n_tasks):
        tid = f"{sprint}-T{t+1:02d}"
        title = rng.choice(TASK_TITLES)
        assignee = overloaded_member if (overloaded_member and rng.random() < 0.5) else rng.choice(team)
        points = rng.choice([1, 2, 3, 5, 8])
        created = sprint_start + _dt.timedelta(days=rng.randint(0, 2))
        # normal tasks: recent activity (idle < STALE_DAYS) so they read as clean
        updated = today - _dt.timedelta(days=rng.randint(0, 3))
        status = rng.choice(["todo", "in_progress", "in_progress", "done", "done"])
        due = sprint_end                     # far future -> not "due soon"
        labels = rng.sample(["frontend", "backend", "infra", "bug", "feature"], rng.randint(0, 2))

        risk_reason = None

        # Inject edge cases on a subset of tasks (these are the labeled risks)
        if "blocked" in active_edges and rng.random() < 0.12:
            status = "blocked"
            risk_reason = "blocked"
        elif "overdue" in active_edges and rng.random() < 0.12:
            status = rng.choice(["todo", "in_progress"])
            due = today - _dt.timedelta(days=rng.randint(1, 4))
            risk_reason = "overdue"
        elif "stale" in active_edges and rng.random() < 0.14:
            status = "in_progress"
            created = today - _dt.timedelta(days=rng.randint(11, 14))
            updated = today - _dt.timedelta(days=rng.randint(5, 9))
            due = sprint_end
            risk_reason = "stale"

        if absent_member and assignee == absent_member and status == "in_progress" \
                and risk_reason is None:
            # absentee's in-progress work goes stale
            created = today - _dt.timedelta(days=rng.randint(11, 14))
            updated = today - _dt.timedelta(days=rng.randint(6, 9))
            risk_reason = "stale"

        if status == "done":
            updated = min(updated, sprint_end)

        # invariant: updated_at never precedes created_at (schema enforces this)
        updated = max(updated, created)

        task = {
            "task_id": tid, "title": title, "status": status, "assignee": assignee,
            "story_points": float(points), "created_at": _iso(created),
            "updated_at": _iso(updated), "due_date": _iso(due), "sprint": sprint,
            "labels": labels, "source_system": "synthetic",
        }
        tasks.append(task)
        if risk_reason:
            risky_truth.append({"task_id": tid, "reason": risk_reason})

    # scope creep: add extra unplanned tasks late
    if "scope_creep" in active_edges:
        for k in range(rng.randint(2, 4)):
            tid = f"{sprint}-SC{k+1}"
            tasks.append({
                "task_id": tid, "title": "Unplanned: " + rng.choice(TASK_TITLES),
                "status": "todo", "assignee": rng.choice(team),
                "story_points": float(rng.choice([3, 5, 8])),
                "created_at": _iso(today - _dt.timedelta(days=1)),
                "updated_at": _iso(today - _dt.timedelta(days=1)),
                "due_date": _iso(sprint_end), "sprint": sprint,
                "labels": ["scope-creep"], "source_system": "synthetic",
            })

    # KPI ground truth (recomputed independently of the production analytics)
    done = [t for t in tasks if t["status"] == "done"]
    velocity = sum(t["story_points"] for t in done)
    total = len(tasks)
    expected_kpis = {
        "velocity": velocity,
        "completed_tasks": len(done),
        "total_tasks": total,
        "completion_rate": round(len(done) / total, 4) if total else 0.0,
    }

    # Planted action items in a meeting note (known ground truth for extraction)
    action_items = _plant_action_items(team, rng, today)
    meeting_note = _render_meeting_note(sprint, team, action_items, today)

    return {
        "scenario_id": sprint,
        "today": _iso(today),
        "team_size": team_size,
        "team": team,
        "sprint": sprint,
        "sprint_start": _iso(sprint_start),
        "sprint_end": _iso(sprint_end),
        "edge_cases": sorted(active_edges),
        "tasks": tasks,
        "meeting_note": meeting_note,
        "ground_truth": {
            "risky_tasks": risky_truth,
            "expected_kpis": expected_kpis,
            "action_items": action_items,
        },
    }


def _plant_action_items(team, rng, today) -> list[dict]:
    verbs = [
        "Follow up on the staging deploy", "Review the security patch PR",
        "Schedule the retro", "Update the API documentation",
        "Investigate the failing payment test", "Rebalance the sprint board",
    ]
    n = rng.randint(2, 4)
    items = []
    for v in rng.sample(verbs, n):
        owner = rng.choice(team)
        deadline = today + _dt.timedelta(days=rng.randint(1, 7))
        items.append({"description": v, "owner": owner, "deadline": _iso(deadline)})
    return items


def _render_meeting_note(sprint, team, action_items, today) -> str:
    lines = [
        f"Date: {_iso(today)}",
        f"Sprint: {sprint}",
        f"Attendees: {', '.join(team)}",
        "",
        "Discussion:",
        "- The team reviewed sprint progress and blockers.",
        "- Decided to prioritize the security patch this week.",
        "- Agreed to tighten the definition of done for QA.",
        "",
        "Action items:",
    ]
    for a in action_items:
        lines.append(f"- {a['owner']} to {a['description'].lower()} by {a['deadline']}.")
    return "\n".join(lines)


def generate_suite(n: int, seed: int, out_dir: Path, today: _dt.date | None = None) -> list[Path]:
    rng = random.Random(seed)
    today = today or _dt.date.today()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(1, n + 1):
        scenario = generate_scenario(i, rng, today)
        path = out_dir / f"{scenario['scenario_id']}.json"
        path.write_text(json.dumps(scenario, indent=2), encoding="utf-8")
        paths.append(path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic sprint scenarios")
    ap.add_argument("-n", "--count", type=int, default=24, help="number of scenarios (>=20)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=SYNTHETIC_DIR)
    args = ap.parse_args()
    paths = generate_suite(max(args.count, 20), args.seed, args.out)
    print(f"Generated {len(paths)} scenarios in {args.out}")


if __name__ == "__main__":
    main()
