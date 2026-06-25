"""Raw file loaders. These just read bytes into memory; normalization happens
in normalize.py. Supports CSV/Excel task exports and plain-text/Markdown notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def load_tabular(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel task export into a raw DataFrame (no normalization).

    Column names are left untouched here so the normalizer can map them.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    elif suffix in {".xlsx", ".xls", ".xlsm"}:
        df = pd.read_excel(path, dtype=str, engine="openpyxl")
        df = df.fillna("")
    else:
        raise ValueError(f"Unsupported tabular format: {suffix}")

    # Normalize header whitespace/case for robust mapping later.
    df.columns = [str(c).strip() for c in df.columns]
    return df


@dataclass
class MeetingNote:
    """A loaded but unparsed meeting note."""

    source: str       # filename or id, used as provenance
    text: str
    meeting_date: str | None = None   # ISO string if parseable from filename/header


def load_meeting_notes(path: str | Path) -> MeetingNote:
    """Load a plain-text or Markdown meeting note.

    A leading ``Date:`` line or an ISO date in the filename is captured as a hint
    for the summarizer; otherwise meeting_date stays None and the LLM infers it.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in {".txt", ".md", ".markdown"}:
        raise ValueError(f"Unsupported note format: {path.suffix}")

    text = path.read_text(encoding="utf-8")
    date_hint = _extract_date_hint(text, path.stem)
    return MeetingNote(source=path.name, text=text, meeting_date=date_hint)


def _extract_date_hint(text: str, stem: str) -> str | None:
    import re

    iso = re.compile(r"(\d{4}-\d{2}-\d{2})")
    # 1) explicit "Date:" header line
    for line in text.splitlines()[:10]:
        if line.lower().startswith("date:"):
            m = iso.search(line)
            if m:
                return m.group(1)
    # 2) ISO date anywhere in filename
    m = iso.search(stem)
    return m.group(1) if m else None
