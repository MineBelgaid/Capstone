"""Normalize heterogeneous task exports into the internal ``Task`` schema.

Jira, Trello and GitHub Projects all use different column names for the same
concepts. We map them onto the canonical fields of ``schemas.Task``. Unknown
columns are ignored; missing optional fields become ``None``.
"""

from __future__ import annotations

import datetime as _dt
from typing import Iterable

import pandas as pd
from dateutil import parser as dateparser

from schemas import Task, TaskStatus

# Canonical internal field  ->  list of accepted source column names (lowercased)
FIELD_MAPS: dict[str, list[str]] = {
    "task_id": ["task_id", "id", "key", "issue key", "card id", "number"],
    "title": ["title", "summary", "name", "card name", "issue summary"],
    "status": ["status", "state", "list", "column", "issue status"],
    "assignee": ["assignee", "assigned to", "owner", "member", "members"],
    "story_points": ["story points", "points", "estimate", "sp", "story_points"],
    "created_at": ["created", "created at", "created_at", "date created"],
    "updated_at": ["updated", "updated at", "updated_at", "last activity", "modified"],
    "due_date": ["due", "due date", "due_date", "deadline"],
    "sprint": ["sprint", "iteration", "milestone"],
    "labels": ["labels", "label", "tags", "components"],
}

# Map raw status strings (lowercased) onto the canonical TaskStatus enum.
_STATUS_MAP: dict[str, TaskStatus] = {
    "todo": TaskStatus.TODO, "to do": TaskStatus.TODO, "backlog": TaskStatus.TODO,
    "open": TaskStatus.TODO, "new": TaskStatus.TODO, "selected for development": TaskStatus.TODO,
    "in progress": TaskStatus.IN_PROGRESS, "in-progress": TaskStatus.IN_PROGRESS,
    "doing": TaskStatus.IN_PROGRESS, "started": TaskStatus.IN_PROGRESS,
    "in review": TaskStatus.IN_PROGRESS, "review": TaskStatus.IN_PROGRESS,
    "blocked": TaskStatus.BLOCKED, "impediment": TaskStatus.BLOCKED, "on hold": TaskStatus.BLOCKED,
    "done": TaskStatus.DONE, "closed": TaskStatus.DONE, "resolved": TaskStatus.DONE,
    "complete": TaskStatus.DONE, "completed": TaskStatus.DONE, "merged": TaskStatus.DONE,
    "cancelled": TaskStatus.CANCELLED, "canceled": TaskStatus.CANCELLED, "wont do": TaskStatus.CANCELLED,
}


def detect_source_system(columns: Iterable[str]) -> str:
    cols = {c.strip().lower() for c in columns}
    if {"issue key", "issue status"} & cols or "issue summary" in cols:
        return "jira"
    if {"card id", "card name", "list"} & cols:
        return "trello"
    if {"number"} & cols and {"title"} & cols and {"milestone", "labels"} & cols:
        return "github"
    return "unknown"


def _build_reverse_map(columns: list[str]) -> dict[str, str]:
    """Map each canonical field to the actual column name present in the file."""
    lower_to_actual = {c.strip().lower(): c for c in columns}
    resolved: dict[str, str] = {}
    for canonical, aliases in FIELD_MAPS.items():
        for alias in aliases:
            if alias in lower_to_actual:
                resolved[canonical] = lower_to_actual[alias]
                break
    return resolved


def _parse_date(value: str | None) -> _dt.date | None:
    if not value or not str(value).strip():
        return None
    try:
        return dateparser.parse(str(value)).date()
    except (ValueError, OverflowError, TypeError):
        return None


def _parse_points(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        pts = float(str(value).strip())
        return pts if pts >= 0 else None
    except ValueError:
        return None


def _parse_status(value: str | None) -> TaskStatus:
    key = (value or "").strip().lower()
    return _STATUS_MAP.get(key, TaskStatus.TODO)


def _parse_labels(value: str | None) -> list[str]:
    if not value:
        return []
    raw = str(value).replace(";", ",").replace("|", ",")
    return [x.strip() for x in raw.split(",") if x.strip()]


def normalize_tasks(df: pd.DataFrame, source_system: str | None = None) -> list[Task]:
    """Convert a raw export DataFrame into a list of validated ``Task`` objects.

    Rows that cannot satisfy the schema (e.g. missing id/title) are skipped rather
    than aborting the whole batch; counts are available via the returned list.
    """
    columns = list(df.columns)
    source_system = source_system or detect_source_system(columns)
    rev = _build_reverse_map(columns)

    def cell(row: pd.Series, canonical: str) -> str | None:
        col = rev.get(canonical)
        if col is None:
            return None
        val = row.get(col)
        return None if val is None else str(val).strip() or None

    tasks: list[Task] = []
    for idx, row in df.iterrows():
        task_id = cell(row, "task_id") or f"row-{idx}"
        title = cell(row, "title")
        if not title:
            continue  # a task with no title is unusable
        try:
            tasks.append(
                Task(
                    task_id=task_id,
                    title=title,
                    status=_parse_status(cell(row, "status")),
                    assignee=cell(row, "assignee"),
                    story_points=_parse_points(cell(row, "story_points")),
                    created_at=_parse_date(cell(row, "created_at")),
                    updated_at=_parse_date(cell(row, "updated_at")),
                    due_date=_parse_date(cell(row, "due_date")),
                    sprint=cell(row, "sprint"),
                    labels=_parse_labels(cell(row, "labels")),
                    source_system=source_system,
                )
            )
        except Exception:  # noqa: BLE001 - skip malformed rows, keep the batch
            continue
    return tasks
