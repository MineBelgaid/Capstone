# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

**AI Project Workflow Intelligence Agent** — a single-agent system that ingests
project data (task exports + meeting notes), reasons over it with a LangGraph
ReAct loop, and produces structured, human-reviewed outputs (KPI reports, risk
alerts, meeting summaries, status reports, workload-rebalancing proposals).

HAW Hamburg Agentic AI capstone. Runs **fully local and free** on Ollama during
development; the LLM backend swaps to Claude for the final demo via one env var.

Read these for depth (don't duplicate them — link):
- [README.md](README.md) — overview + usage
- [ARCHITECTURE.md](ARCHITECTURE.md) — design decisions + "why" (use for any
  design question or defense)
- [PRESENTATION.md](PRESENTATION.md) — the 15-minute live-demo script

## How to run

**One command does everything** (preflight checks Ollama, models, Python, venv,
`.env`, starts the Ollama server if needed, then launches the dashboard):

- Windows: `.\run.ps1`   (verify only: `.\run.ps1 -Check`)
- macOS/Linux: `./run.sh`   (verify only: `./run.sh --check`)

Dashboard serves at **http://localhost:8501**.

> Windows execution-policy error the first time? Run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Manual / individual commands

Always use the project virtualenv interpreter, not the system Python:
- Windows: `.\.venv\Scripts\python.exe`
- Unix: `.venv/bin/python`

```bash
# Deterministic eval (NO LLM needed — offline proof, good for CI)
python -m eval.score --dir data/scenarios

# LLM-backed eval (needs the Ollama server up)
python -m eval.score --meetings data/meetings

# Regenerate the curated scenarios + meeting-note set
python -m eval.catalog

# Dashboard
python -m streamlit run dashboard/app.py

# Tests
python -m pytest tests/
```

## Prerequisites

- **Python 3.10+**
- **Ollama** installed and running. The server must be up (the launchers start it;
  otherwise the app fails with a connection-refused error on port 11434).
- Models: `qwen2.5:14b` (chat) and `nomic-embed-text` (embeddings). `run.ps1`/`run.sh`
  pull them if missing. `llama3.1:8b` is the documented fallback for <16GB RAM
  (set `OLLAMA_MODEL`).
- First run pulls ~9GB (qwen2.5:14b) and installs a large dependency set (torch,
  sentence-transformers, chromadb) — expect several minutes.

## Configuration

All in [config.py](config.py), env-overridable (see [.env.example](.env.example)):
- `LLM_BACKEND` — `ollama` (default) or `claude`. **The whole system switches on
  this one variable; never branch on backend elsewhere** — call
  `agent/llm.py::get_chat_model()`.
- `REACT_MODE` — `custom` (default, robust on local models) or `prebuilt` (native
  tool calling, best on Claude).
- `REACT_REFLECTION` — `true` (default): self-critique pass after the loop.
- `OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL`, `CLAUDE_MODEL`, `ANTHROPIC_API_KEY`.

## Architecture (map)

5-stage pipeline: **Ingest → Retrieve → Reason → Validate → Review**

| Module | Role |
|--------|------|
| `config.py` | Single backend switch + all settings |
| `schemas/models.py` | Pydantic v2 contract for every output (`extra="forbid"`, no `Any`) |
| `ingestion/` | Loaders + Jira/Trello/GitHub → internal `Task` normalizer |
| `retrieval/` | ChromaDB store + local embeddings (Ollama, ST fallback) |
| `agent/analytics.py` | **Deterministic** KPIs, risk detection, workload balance |
| `agent/tools.py` | Tools; `export_report` is approval-gated |
| `agent/react.py` | Custom ReAct loop + `stream_custom_react` + reflection |
| `agent/graph.py` | Prebuilt LangGraph tool-calling agent (Claude path) |
| `agent/validation.py` | `generate_structured`: LLM → Pydantic with retry |
| `agent/reporting.py` | Assembles the weekly `StatusReport` |
| `dashboard/app.py` | Streamlit review UI (all features surface here) |
| `eval/` | Curated catalog, random generator, meeting set, scoring |

## Design rules (do NOT violate these when editing)

1. **Zero autonomous external actions.** The only external action
   (`export_report`) must stay (a) excluded from the ReAct loop's tool registry in
   `agent/react.py`, and (b) gated on `human_approved` in `agent/tools.py`.
   Same rule for the rebalancing feature: it *proposes*, a human applies.
2. **Numbers are computed, not generated.** KPIs, risk signals, and workload
   analysis are deterministic Python in `agent/analytics.py`. The LLM only
   narrates/interprets. Do not move numeric computation into an LLM prompt.
3. **All structured LLM output goes through `generate_structured`** against a
   strict Pydantic schema (validate + retry). New structured outputs must add a
   schema in `schemas/models.py` and export it from `schemas/__init__.py`.
4. **LLM-proposal features need a deterministic guardrail.** Validate model output
   against ground truth and drop/replace invalid fields (see `propose_rebalance`).
5. **Eval ground truth stays independent of the code it measures** (assigned by
   construction in `eval/catalog.py` / `eval/generate.py`).

## Conventions

- Match the surrounding style: `from __future__ import annotations`, type hints,
  module docstrings explaining the *why*, `# noqa: BLE001` on intentional
  broad excepts that degrade gracefully.
- New dashboard features should be **visible in the Streamlit UI** and persist
  across reruns via `st.session_state` (cleared on scenario change).
- After editing agent/dashboard code, byte-compile to catch errors before running:
  `python -m py_compile <files>`.

## Gotchas

- The Ollama server stops between machine sessions — if an LLM call fails with
  "connection refused", start it (`run.ps1 -Check`) before retrying.
- First LLM call after idle is a **cold start** (model loads into VRAM, ~10-20s);
  warm it once before a live demo.
- The dashboard reads the curated catalog from `data/scenarios/` and the random
  set from `data/synthetic/`; run `python -m eval.catalog` if `data/scenarios/`
  is empty.
- `.env` is gitignored; copy from `.env.example` (the launchers do this).
