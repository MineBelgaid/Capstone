"""Separate meeting-note test set for the summarizer / action-item eval.

Each entry is a realistic meeting note whose action items are KNOWN ground truth
(owner + deadline + description). The summarizer must extract them; the scorer
matches extracted items against this truth to measure the >90% accuracy target.

Design rules so the truth is fair and checkable:
  * Every action item's owner appears in ``attendees``.
  * Every deadline is a real ISO date.
  * The note text states each action item explicitly (a named owner + the task),
    so a competent model can extract it -- this isn't a trick set.

Format matches what ``eval/score.py`` reads: ``meeting_note`` (text) plus
``ground_truth.action_items``.
"""

from __future__ import annotations


def _entry(scenario_id: str, attendees: list[str], note: str,
           action_items: list[dict]) -> dict:
    # sanity: owners must be attendees (kept as an assertion for catalog integrity)
    for ai in action_items:
        assert ai["owner"] in attendees, f"{ai['owner']} not in attendees of {scenario_id}"
    return {
        "scenario_id": scenario_id,
        "attendees": attendees,
        "meeting_note": note,
        "ground_truth": {"action_items": action_items},
    }


MEETINGS = [
    _entry(
        "MTG-01-standup",
        ["Ava", "Ben", "Carla"],
        (
            "Date: 2026-06-22\n"
            "Daily standup.\n"
            "Attendees: Ava, Ben, Carla.\n\n"
            "- Ava finished the login flow and will write the integration tests by "
            "2026-06-24.\n"
            "- Ben is blocked on the payment vendor; Carla will email the vendor for "
            "API access by 2026-06-23.\n"
            "- We agreed to demo the checkout flow on Friday.\n"
        ),
        [
            {"description": "Write integration tests for the login flow",
             "owner": "Ava", "deadline": "2026-06-24"},
            {"description": "Email the payment vendor for API access",
             "owner": "Carla", "deadline": "2026-06-23"},
        ],
    ),
    _entry(
        "MTG-02-sprint-planning",
        ["Dmitri", "Elena", "Farid", "Grace"],
        (
            "Date: 2026-06-22\n"
            "Sprint planning for Sprint 14.\n"
            "Attendees: Dmitri, Elena, Farid, Grace.\n\n"
            "Decisions: prioritize the search backend; defer the admin dashboard.\n\n"
            "Action items:\n"
            "- Dmitri to break down the search backend into subtasks by 2026-06-23.\n"
            "- Elena to set up the staging environment by 2026-06-25.\n"
            "- Farid to review the API design doc by 2026-06-24.\n"
            "- Grace to schedule the stakeholder review by 2026-06-26.\n"
        ),
        [
            {"description": "Break down the search backend into subtasks",
             "owner": "Dmitri", "deadline": "2026-06-23"},
            {"description": "Set up the staging environment",
             "owner": "Elena", "deadline": "2026-06-25"},
            {"description": "Review the API design doc",
             "owner": "Farid", "deadline": "2026-06-24"},
            {"description": "Schedule the stakeholder review",
             "owner": "Grace", "deadline": "2026-06-26"},
        ],
    ),
    _entry(
        "MTG-03-retro",
        ["Ava", "Ben", "Hassan"],
        (
            "Date: 2026-06-22\n"
            "Sprint retrospective.\n"
            "Attendees: Ava, Ben, Hassan.\n\n"
            "What went well: faster code review. What to improve: flaky CI.\n\n"
            "- Hassan will stabilize the flaky test suite by 2026-06-27.\n"
            "- Ava will document the new review checklist by 2026-06-25.\n"
            "- Ben mentioned morale is good; no action needed there.\n"
        ),
        [
            {"description": "Stabilize the flaky test suite",
             "owner": "Hassan", "deadline": "2026-06-27"},
            {"description": "Document the new review checklist",
             "owner": "Ava", "deadline": "2026-06-25"},
        ],
    ),
    _entry(
        "MTG-04-stakeholder-review",
        ["Grace", "Ines", "Jonas", "Kira"],
        (
            "Date: 2026-06-22\n"
            "Stakeholder review.\n"
            "Attendees: Grace, Ines, Jonas, Kira.\n\n"
            "Stakeholders requested a clearer onboarding and an export feature.\n\n"
            "Action items:\n"
            "- Ines to draft the onboarding redesign by 2026-06-29.\n"
            "- Jonas to scope the CSV export feature by 2026-06-26.\n"
            "- Kira to follow up with stakeholders on pricing by 2026-06-30.\n"
        ),
        [
            {"description": "Draft the onboarding redesign",
             "owner": "Ines", "deadline": "2026-06-29"},
            {"description": "Scope the CSV export feature",
             "owner": "Jonas", "deadline": "2026-06-26"},
            {"description": "Follow up with stakeholders on pricing",
             "owner": "Kira", "deadline": "2026-06-30"},
        ],
    ),
    _entry(
        "MTG-05-postmortem",
        ["Ben", "Elena", "Leo"],
        (
            "Date: 2026-06-22\n"
            "Incident postmortem: API outage.\n"
            "Attendees: Ben, Elena, Leo.\n\n"
            "Root cause: an unbounded query under load. Blameless review.\n\n"
            "- Leo will add a query timeout and index by 2026-06-24.\n"
            "- Elena will add load-test coverage for the endpoint by 2026-06-28.\n"
            "- Ben will write the public incident summary by 2026-06-23.\n"
        ),
        [
            {"description": "Add a query timeout and index",
             "owner": "Leo", "deadline": "2026-06-24"},
            {"description": "Add load-test coverage for the endpoint",
             "owner": "Elena", "deadline": "2026-06-28"},
            {"description": "Write the public incident summary",
             "owner": "Ben", "deadline": "2026-06-23"},
        ],
    ),
]
