"""Scoring harness: run the system over synthetic scenarios and measure against
the Section 7 success criteria.

Deterministic metrics run with NO LLM (so the suite works offline / in CI):
  * KPI report schema validity      -> target 100%
  * KPI numeric correctness          -> matches independently-computed ground truth
  * Risk detection precision/recall  -> against by-construction risk labels
  * Workflow execution time          -> target < 60s/scenario

LLM-dependent metrics run only when ``--with-llm`` is passed and a backend is up:
  * Action-item extraction accuracy  -> target > 90%
  * Meeting summary validity         -> structured output validates

Usage:
    python -m eval.score                # deterministic metrics only
    python -m eval.score --with-llm     # also run summarizer (needs Ollama/Claude)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from config import SYNTHETIC_DIR
from agent.analytics import compute_kpis, detect_risks
from schemas import SprintKPIReport, Task


# --------------------------------------------------------------------------- #
# Targets (from the brief, Section 7)
# --------------------------------------------------------------------------- #
TARGETS = {
    "kpi_schema_validity": 1.00,
    "kpi_numeric_accuracy": 1.00,
    "risk_precision": 0.85,
    "action_item_accuracy": 0.90,
    "max_workflow_seconds": 60.0,
}


@dataclass
class ScenarioResult:
    scenario_id: str
    kpi_valid: bool
    kpi_numeric_ok: bool
    risk_precision: float
    risk_recall: float
    action_item_accuracy: float | None  # None when LLM not run
    elapsed_s: float
    notes: list[str] = field(default_factory=list)


def _load_tasks(scenario: dict) -> list[Task]:
    return [Task.model_validate(t) for t in scenario["tasks"]]


def _risk_task_ids(alerts) -> set[str]:
    ids: set[str] = set()
    for a in alerts:
        ids.update(a.related_task_ids)
    return ids


def score_scenario(scenario: dict, with_llm: bool) -> ScenarioResult:
    start = time.perf_counter()
    notes: list[str] = []
    today = _dt.date.fromisoformat(scenario["today"])
    tasks = _load_tasks(scenario)

    # --- KPIs ---
    kpis = compute_kpis(
        tasks,
        scenario["sprint"],
        _dt.date.fromisoformat(scenario["sprint_start"]),
        _dt.date.fromisoformat(scenario["sprint_end"]),
    )
    kpi_valid = isinstance(kpis, SprintKPIReport)
    gt = scenario["ground_truth"]["expected_kpis"]
    kpi_numeric_ok = (
        kpis.total_tasks == gt["total_tasks"]
        and kpis.completed_tasks == gt["completed_tasks"]
        and abs(kpis.velocity - gt["velocity"]) < 1e-6
        and abs(kpis.completion_rate - gt["completion_rate"]) < 1e-3
    )
    if not kpi_numeric_ok:
        notes.append(f"KPI mismatch: got v={kpis.velocity} cr={kpis.completion_rate}")

    # --- Risk precision/recall vs by-construction labels ---
    truth_ids = {r["task_id"] for r in scenario["ground_truth"]["risky_tasks"]}
    flagged_ids = _risk_task_ids(detect_risks(tasks, today=today))
    flagged_known = {i for i in flagged_ids if i in truth_ids or i in {t.task_id for t in tasks}}
    tp = len(flagged_ids & truth_ids)
    precision = tp / len(flagged_ids) if flagged_ids else 1.0
    recall = tp / len(truth_ids) if truth_ids else 1.0

    # --- Action-item extraction (LLM) -- only for scenarios that carry notes ---
    ai_accuracy: float | None = None
    if with_llm and scenario.get("meeting_note", "").strip():
        ai_accuracy = _score_action_items(scenario, notes)

    elapsed = time.perf_counter() - start
    return ScenarioResult(
        scenario_id=scenario["scenario_id"],
        kpi_valid=kpi_valid,
        kpi_numeric_ok=kpi_numeric_ok,
        risk_precision=round(precision, 4),
        risk_recall=round(recall, 4),
        action_item_accuracy=ai_accuracy,
        elapsed_s=round(elapsed, 3),
        notes=notes,
    )


def _score_action_items(scenario: dict, notes: list[str]) -> float:
    """Run the summarizer tool and match extracted action items to ground truth."""
    from agent.tools import AgentContext, set_context, summarize_meeting

    set_context(AgentContext(
        meeting_text=scenario["meeting_note"],
        meeting_source=f"{scenario['scenario_id']}-notes",
    ))
    try:
        payload = json.loads(summarize_meeting.invoke({"_": ""}))
        extracted = payload.get("action_items", [])
    except Exception as exc:  # noqa: BLE001
        notes.append(f"summarizer failed: {exc}")
        return 0.0

    truth = scenario["ground_truth"]["action_items"]
    return match_action_items(extracted, truth)


def match_action_items(extracted: list[dict], truth: list[dict]) -> float:
    """Fraction of ground-truth action items recovered (owner match + description
    token overlap). Pure function -- unit-testable without any LLM."""
    if not truth:
        return 1.0
    matched = 0
    extracted_owners = [(e.get("owner", ""), e.get("description", "")) for e in extracted]
    for t in truth:
        for owner, desc in extracted_owners:
            if owner.lower() == t["owner"].lower() and _overlap(desc, t["description"]):
                matched += 1
                break
    return round(matched / len(truth), 4)


def _overlap(a: str, b: str) -> bool:
    """Loose token-overlap match between two action-item descriptions."""
    sa = {w for w in a.lower().split() if len(w) > 3}
    sb = {w for w in b.lower().split() if len(w) > 3}
    if not sb:
        return False
    return len(sa & sb) / len(sb) >= 0.4


# --------------------------------------------------------------------------- #
# Aggregation / reporting
# --------------------------------------------------------------------------- #
def summarize(results: list[ScenarioResult], with_llm: bool) -> dict:
    n = len(results)
    agg = {
        "scenarios": n,
        "kpi_schema_validity": round(sum(r.kpi_valid for r in results) / n, 4),
        "kpi_numeric_accuracy": round(sum(r.kpi_numeric_ok for r in results) / n, 4),
        "risk_precision_mean": round(sum(r.risk_precision for r in results) / n, 4),
        "risk_recall_mean": round(sum(r.risk_recall for r in results) / n, 4),
        "max_workflow_seconds": round(max(r.elapsed_s for r in results), 3),
        "mean_workflow_seconds": round(sum(r.elapsed_s for r in results) / n, 3),
    }
    if with_llm:
        ai = [r.action_item_accuracy for r in results if r.action_item_accuracy is not None]
        agg["action_item_accuracy_mean"] = round(sum(ai) / len(ai), 4) if ai else None

    agg["pass"] = {
        "kpi_schema_validity": agg["kpi_schema_validity"] >= TARGETS["kpi_schema_validity"],
        "kpi_numeric_accuracy": agg["kpi_numeric_accuracy"] >= TARGETS["kpi_numeric_accuracy"],
        "risk_precision": agg["risk_precision_mean"] >= TARGETS["risk_precision"],
        "workflow_time": agg["max_workflow_seconds"] <= TARGETS["max_workflow_seconds"],
    }
    if with_llm and agg.get("action_item_accuracy_mean") is not None:
        agg["pass"]["action_item_accuracy"] = (
            agg["action_item_accuracy_mean"] >= TARGETS["action_item_accuracy"]
        )
    return agg


def run(scenario_dir: Path, with_llm: bool) -> dict:
    files = sorted(scenario_dir.glob("*.json"))
    if not files:
        raise SystemExit(
            f"No scenarios in {scenario_dir}. Run: python -m eval.generate -n 24"
        )
    results = [score_scenario(json.loads(p.read_text()), with_llm) for p in files]
    agg = summarize(results, with_llm)
    agg["per_scenario"] = [r.__dict__ for r in results]
    return agg


def run_meetings(meeting_dir: Path) -> dict:
    """Score action-item extraction over the standalone meeting-note set (needs LLM)."""
    files = sorted(meeting_dir.glob("*.json"))
    if not files:
        raise SystemExit(
            f"No meeting notes in {meeting_dir}. Run: python -m eval.catalog"
        )
    per = []
    for p in files:
        m = json.loads(p.read_text())
        notes: list[str] = []
        acc = _score_action_items(m, notes)
        per.append({"scenario_id": m["scenario_id"], "action_item_accuracy": acc,
                    "notes": notes})
    mean = round(sum(r["action_item_accuracy"] for r in per) / len(per), 4)
    return {
        "meetings": len(per),
        "action_item_accuracy_mean": mean,
        "pass": {"action_item_accuracy": mean >= TARGETS["action_item_accuracy"]},
        "per_meeting": per,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the agent over test scenarios")
    ap.add_argument("--dir", type=Path, default=SYNTHETIC_DIR,
                    help="sprint scenario dir (use data/scenarios for the curated catalog)")
    ap.add_argument("--with-llm", action="store_true",
                    help="also score action-item extraction (needs Ollama/Claude)")
    ap.add_argument("--meetings", type=Path, default=None,
                    help="score the standalone meeting-note set in this dir (needs LLM)")
    ap.add_argument("--out", type=Path, default=None, help="write full JSON report here")
    args = ap.parse_args()

    if args.meetings:
        agg = run_meetings(args.meetings)
        print(json.dumps({k: v for k, v in agg.items() if k != "per_meeting"}, indent=2))
    else:
        agg = run(args.dir, args.with_llm)
        print(json.dumps({k: v for k, v in agg.items() if k != "per_scenario"}, indent=2))
    if args.out:
        args.out.write_text(json.dumps(agg, indent=2), encoding="utf-8")
        print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
