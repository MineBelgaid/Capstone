# AI Project Workflow Intelligence Agent

A single-agent system that reduces project-coordination overhead for small teams.
It ingests project data (CSV/Excel task exports, meeting notes), reasons over it
with a LangGraph ReAct loop, and produces structured outputs — meeting summaries,
KPI reports, risk alerts, status reports — that **a human reviews and approves
before anything is exported**.

> HAW Hamburg Agentic AI capstone. Everything runs locally and free during
> development; the LLM backend swaps to the Claude API only for the final demo
> via a one-line config change.

## Hard design rule

Zero autonomous external actions. The agent never writes to an external system
or file without explicit human confirmation in the Streamlit review dashboard.
This is enforced in `agent/tools.py::export_report`, which refuses to write unless
the run is `human_approved`.

## Architecture — 5-stage pipeline

```
Ingest → Retrieve → Reason → Validate → Review
```

| Stage | Module | What it does |
|-------|--------|--------------|
| Ingest | `ingestion/` | Load CSV/Excel/notes; normalize Jira/Trello/GitHub exports into the internal `Task` schema |
| Retrieve | `retrieval/` | Embed tasks & notes into persistent ChromaDB (local embeddings); similarity search with metadata filtering |
| Reason | `agent/graph.py`, `agent/react.py` | LangGraph ReAct loop selects tools (summarize, KPIs, risks) |
| Validate | `agent/validation.py` | Every LLM output validated against a Pydantic v2 schema, with retry on failure |
| Review | `dashboard/app.py` | Streamlit UI: view, edit, approve/reject. Nothing exports without approval |

Agentic patterns demonstrated: **ReAct**, **Tool Use** (each capability is a
distinct tool), **RAG** (ChromaDB retrieval), **Human-in-the-loop** (approval
gate), and an optional **MCP/file export** tool extension.

### ReAct loop: two modes

`REACT_MODE` selects the loop implementation:

- **`custom`** (default, recommended for local Ollama) — an explicit
  reason→act→observe graph (`agent/react.py`) where each step is a single
  constrained `ReActStep` JSON object validated by Pydantic with retry. Robust on
  local models that emit native tool-call JSON inconsistently; the loop also
  *cannot* select `export_report`, so external actions are unreachable
  autonomously.
- **`prebuilt`** — LangGraph `create_react_agent` native tool calling, ideal on
  Claude for the final demo.

Same tools and schemas either way; switching is one env var.

A key design choice: **KPIs and risk signals are computed deterministically in
Python** (`agent/analytics.py`), not hallucinated by the LLM. The model is used
only to narrate/summarize. This is what keeps KPI schema validity at 100% and
risk precision high and reproducible.

## Tech stack (100% free / local)

LangGraph · Ollama (qwen2.5:14b / llama3.1:8b) · ChromaDB · Pydantic v2 ·
Streamlit · Plotly · Pandas · SQLite · local embeddings (nomic-embed-text or
sentence-transformers all-MiniLM-L6-v2) · LangSmith free tier. Claude API is used
**only** for the final demo.

## Project layout

```
config.py            # single LLM backend switch (ollama | claude) + all settings
schemas/             # Pydantic v2 models (validation contract)
ingestion/           # loaders + Jira/Trello/GitHub field-mapping normalizer
retrieval/           # ChromaDB store + local embeddings (with fallback)
agent/               # LLM factory, deterministic analytics, tools, ReAct graph, validation
dashboard/           # Streamlit human-review UI (Plotly charts + approval gate)
eval/                # synthetic scenario generator + scoring harness
data/synthetic/      # generated scenarios (gitignored)
```

## Setup

```bash
# 1. Install Ollama, then pull the models (local, free)
ollama pull qwen2.5:14b          # or: ollama pull llama3.1:8b  (RAM < 16GB)
ollama pull nomic-embed-text

# 2. Python env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Config
cp .env.example .env             # edit if needed; defaults to local Ollama
```

## Usage

```bash
# Curated test catalog: 13 named scenarios (team 3–15) + 5 meeting notes
python -m eval.catalog
python -m eval.score --dir data/scenarios            # deterministic, no LLM
python -m eval.score --meetings data/meetings        # action-item extraction (needs LLM)

# Random synthetic suite (for volume / regression)
python -m eval.generate -n 24
python -m eval.score --out data/eval_report.json     # deterministic metrics
python -m eval.score --with-llm                      # + action-item extraction (needs LLM)

# Launch the review dashboard
streamlit run dashboard/app.py
```

### Two scenario sources

- **Curated catalog** (`eval/catalog.py` → `data/scenarios/`) — hand-designed,
  documented scenarios with a clear purpose each (single blocker, bottleneck,
  minor/severe deadline slips, scope creep, absentee dev, absentee+deadline combo,
  overloaded lead, large/very-large teams, end-of-sprint crunch, plus a healthy
  negative control). Version-controlled; ideal for the demo and grading.
- **Random generator** (`eval/generate.py` → `data/synthetic/`) — fully synthetic
  scenarios for volume and regression. Gitignored.
- **Meeting notes** (`eval/meetings.py` → `data/meetings/`) — 5 realistic notes
  with planted action items (owner + deadline) as ground truth for the >90%
  extraction target.

Scope creep and overload show up as KPI signals (lower completion, skewed
workload) rather than per-task `RiskAlert`s; the blocker/deadline/stale cases are
the task-level risks the precision target measures.

### Switching to Claude for the final demo

One change, no refactor:

```bash
export LLM_BACKEND=claude
export ANTHROPIC_API_KEY=sk-ant-...
```

`config.py` and `agent/llm.py` route everything off `LLM_BACKEND`; all other code
is backend-agnostic.

## Pydantic schemas (validation contract)

`Task` (internal normalized), `ActionItem`, `MeetingSummary`, `RiskAlert`,
`SprintKPIReport` (+ `WorkloadEntry`), `StatusReport`. All strictly typed
(`extra="forbid"`, no `Any`) so malformed LLM output raises `ValidationError` and
triggers a retry rather than reaching the dashboard.

## Evaluation & success criteria (Section 7)

`eval/score.py` measures, over 20+ synthetic scenarios:

| Criterion | Target | How it's measured |
|-----------|--------|-------------------|
| KPI report schema validity | 100% | every report parses as `SprintKPIReport` |
| KPI numeric accuracy | 100% | vs independently-computed ground truth |
| Risk detection precision | > 85% | vs by-construction risk labels |
| Action-item extraction accuracy | > 90% | LLM output matched to planted items (`--with-llm`) |
| Full workflow time | < 60 s | wall-clock per scenario |
| Autonomous external actions | 0 | enforced by approval gate |

Risk labels are assigned **by construction** in the generator, independently of
the detector, so precision/recall are real measurements rather than a tautology.

## Notion MCP (optional / stretch)

Not required. The export tool already demonstrates the Tool-Use pattern by writing
approved reports to Markdown. A Notion MCP push tool can be added later as another
gated external action without touching the core pipeline.

## Status

Week 1–2 scaffold complete and verified: schemas, ingestion, retrieval, agent
graph + deterministic analytics, validation, Streamlit dashboard, and the eval
harness. Deterministic eval passes all offline targets. Next: wire LangSmith
tracing, expand scenario edge-cases, and (week 5) swap in Claude for the demo.
