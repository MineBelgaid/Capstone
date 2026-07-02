"""Strictly-typed Pydantic v2 models.

These are the validation contract: every LLM output must parse into one of these
models before it is allowed to surface in the review dashboard. No loose ``dict``
or ``Any`` fields -- that is the whole point. Malformed LLM output should raise a
``ValidationError`` so the agent can retry (see agent/validation.py).
"""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    field_validator,
    model_validator,
)

# A shared strict base: forbid unknown keys so the LLM can't smuggle junk through.
_STRICT = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=False)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class TaskStatus(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"
    CANCELLED = "cancelled"


class RiskSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# --------------------------------------------------------------------------- #
# Internal normalized task (output of the ingestion layer)
# --------------------------------------------------------------------------- #
class Task(BaseModel):
    """One project task, normalized across Jira / Trello / GitHub exports."""

    model_config = _STRICT

    task_id: str = Field(..., min_length=1, description="Source-stable identifier")
    title: str = Field(..., min_length=1)
    status: TaskStatus
    assignee: str | None = Field(None, description="Team member name; None = unassigned")
    story_points: NonNegativeFloat | None = None
    created_at: _dt.date | None = None
    updated_at: _dt.date | None = None
    due_date: _dt.date | None = None
    sprint: str | None = None
    labels: list[str] = Field(default_factory=list)
    source_system: str | None = Field(None, description="jira | trello | github | ...")

    @model_validator(mode="after")
    def _check_dates(self) -> "Task":
        if self.created_at and self.updated_at and self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        return self


# --------------------------------------------------------------------------- #
# Meeting summarization outputs
# --------------------------------------------------------------------------- #
class ActionItem(BaseModel):
    """A single extracted action item with clear ownership and provenance."""

    model_config = _STRICT

    description: str = Field(..., min_length=3, description="What needs to be done")
    owner: str = Field(..., min_length=1, description="Person responsible")
    deadline: _dt.date | None = Field(None, description="Due date if stated, else None")
    source: str = Field(
        ..., min_length=1, description="Which meeting/doc this came from (provenance)"
    )

    @field_validator("owner")
    @classmethod
    def _owner_not_placeholder(cls, v: str) -> str:
        if v.strip().lower() in {"tbd", "n/a", "unknown", "someone", "?"}:
            raise ValueError("owner must be a concrete person, not a placeholder")
        return v


class MeetingSummary(BaseModel):
    model_config = _STRICT

    meeting_date: _dt.date
    attendees: Annotated[list[str], Field(min_length=1)]
    key_decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    summary: str = Field(..., min_length=10, description="Concise prose summary")
    source: str = Field(..., min_length=1, description="Source document name/id")


# --------------------------------------------------------------------------- #
# Risk detection output
# --------------------------------------------------------------------------- #
class RiskAlert(BaseModel):
    model_config = _STRICT

    area: str = Field(..., min_length=1, description="Task id/title or team member at risk")
    severity: RiskSeverity
    reason: str = Field(..., min_length=5, description="Why this is flagged")
    recommended_action: str = Field(..., min_length=5)
    related_task_ids: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Workload rebalancing proposal (human-approval gated; proposes, never applies)
# --------------------------------------------------------------------------- #
class ReassignmentSuggestion(BaseModel):
    """A single proposed move of one task to a member with spare capacity."""

    model_config = _STRICT

    task_id: str = Field(..., min_length=1)
    task_title: str = Field(..., min_length=1)
    from_member: str = Field(..., min_length=1, description="Current owner (or 'Unassigned')")
    to_member: str = Field(..., min_length=1, description="Proposed new owner with capacity")
    points: NonNegativeFloat = Field(..., description="Story points moved")
    reason: str = Field(..., min_length=5, description="Why this move helps balance load")


class RebalanceProposal(BaseModel):
    model_config = _STRICT

    summary: str = Field(..., min_length=10, description="One-paragraph rationale")
    suggestions: list[ReassignmentSuggestion] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# KPI / sprint analytics output
# --------------------------------------------------------------------------- #
class WorkloadEntry(BaseModel):
    model_config = _STRICT

    member: str = Field(..., min_length=1)
    assigned_tasks: NonNegativeInt
    assigned_points: NonNegativeFloat
    done_tasks: NonNegativeInt

    @model_validator(mode="after")
    def _done_le_assigned(self) -> "WorkloadEntry":
        if self.done_tasks > self.assigned_tasks:
            raise ValueError("done_tasks cannot exceed assigned_tasks")
        return self


class SprintKPIReport(BaseModel):
    model_config = _STRICT

    sprint: str = Field(..., min_length=1)
    sprint_start: _dt.date
    sprint_end: _dt.date
    velocity: NonNegativeFloat = Field(..., description="Completed story points")
    completion_rate: float = Field(..., ge=0.0, le=1.0, description="done / total tasks")
    total_tasks: NonNegativeInt
    completed_tasks: NonNegativeInt
    workload: Annotated[list[WorkloadEntry], Field(min_length=1)]

    @model_validator(mode="after")
    def _coherent(self) -> "SprintKPIReport":
        if self.sprint_end < self.sprint_start:
            raise ValueError("sprint_end cannot precede sprint_start")
        if self.completed_tasks > self.total_tasks:
            raise ValueError("completed_tasks cannot exceed total_tasks")
        return self


# --------------------------------------------------------------------------- #
# Weekly status report (combines everything)
# --------------------------------------------------------------------------- #
class StatusReport(BaseModel):
    model_config = _STRICT

    title: str = Field(..., min_length=1)
    period_start: _dt.date
    period_end: _dt.date
    headline: str = Field(..., min_length=10, description="One-line executive summary")
    kpis: SprintKPIReport
    risks: list[RiskAlert] = Field(default_factory=list)
    recent_action_items: list[ActionItem] = Field(default_factory=list)
    narrative: str = Field(..., min_length=20, description="Prose weekly digest")

    @model_validator(mode="after")
    def _check_period(self) -> "StatusReport":
        if self.period_end < self.period_start:
            raise ValueError("period_end cannot precede period_start")
        return self
