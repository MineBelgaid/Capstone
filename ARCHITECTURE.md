# Architecture & Design Decisions

> **Audience:** the project author, for presentation prep and defense.
> **Goal:** explain *what* the system is, *how* it is built, and — most importantly —
> *why* each significant choice was made, so any question in the demo has an answer.
>
> Companion to the [README](README.md). The README is the "how to run it"; this is
> the "why it is the way it is."

---

## 1. The problem & the goal

**Problem.** Small software teams lose time to coordination overhead: chasing
status, spotting blockers late, writing the same reports every week, digging
action items out of meeting notes.

**Goal.** A single agent that ingests real project data (task exports + meeting
notes), reasons over it, and produces **structured, reviewable outputs** —
KPI reports, risk alerts, meeting summaries, weekly status reports — where a
**human approves anything before it leaves the system.**

**Hard rule (from the brief):** *zero autonomous external actions.* The agent
never writes to an external system or file without explicit human confirmation.

Everything runs **locally and free** during development (Ollama); the LLM backend
swaps to the Claude API for the final demo with a **one-line change**.

---

## 2. System at a glance — the 5-stage pipeline

```
   Ingest          Retrieve         Reason          Validate         Review
 ┌─────────┐    ┌───────────┐   ┌────────────┐   ┌───────────┐   ┌──────────┐
 │ CSV /   │    │ ChromaDB  │   │ LangGraph  │   │ Pydantic  │   │ Streamlit│
 │ Excel / │ →  │ embeddings│ → │ ReAct loop │ → │ v2 schema │ → │ approval │
 │ notes   │    │ + search  │   │ + tools    │   │ + retry   │   │  gate    │
 └─────────┘    └───────────┘   └────────────┘   └───────────┘   └──────────┘
   ingestion/     retrieval/       agent/           schemas/        dashboard/
```

| Stage | Module | Responsibility |
|-------|--------|----------------|
| **Ingest** | `ingestion/` | Load CSV/Excel/notes; normalize Jira/Trello/GitHub exports into one internal `Task` schema |
| **Retrieve** | `retrieval/` | Embed tasks & notes into persistent ChromaDB; similarity search with metadata filtering (RAG) |
| **Reason** | `agent/` | LangGraph ReAct loop selects tools (summarize, KPIs, risks) and narrates results |
| **Validate** | `schemas/`, `agent/validation.py` | Every LLM output parsed into a strict Pydantic model, with retry on failure |
| **Review** | `dashboard/` | Streamlit UI: view, edit, approve/reject. Nothing exports without approval |

This maps directly to the dashboard tabs and to the agentic patterns required by
the brief.

---

## 3. Agentic patterns demonstrated

The brief asks for recognizable agentic patterns. Each is a concrete, pointable
part of the system:

| Pattern | Where it lives | What to show |
|---------|----------------|--------------|
| **ReAct** (reason→act→observe) | `agent/react.py` | The live trace in the **🔬 Agent Process** tab |
| **Tool use** | `agent/tools.py` | Each capability (summarize, KPIs, risks, retrieve, export) is a distinct tool |
| **RAG** | `retrieval/` | The vector store built per scenario; `retrieve_context` tool |
| **Human-in-the-loop** | `agent/tools.py::export_report` + dashboard | The approval gate; export is *blocked* until approved |
| **Structured output / validation** | `agent/validation.py` + `schemas/` | Malformed LLM output raises `ValidationError` → retry, never reaches the user |
| **Reflection / self-critique** | `agent/react.py::reflect_on_answer` | Draft answer re-checked against tool observations; unsupported claims corrected before finalizing |

---

## 4. Key design decisions (the "why")

This is the heart of the defense. Each decision states the choice, the
reasoning, the trade-off accepted, and the alternative rejected.

### 4.1 Deterministic analytics, LLM only for narration

**Choice.** KPIs (velocity, completion, workload) and risk signals (blocked,
overdue, stale, overloaded) are **computed in plain Python** (`agent/analytics.py`).
The LLM is used **only** to narrate/summarize those numbers — never to produce them.

**Why.** Numbers must be correct and reproducible. An LLM asked "what's the
velocity?" can hallucinate or drift run-to-run. By computing them deterministically:
- KPI **numeric accuracy is 100%** (it's arithmetic, cross-checked against
  independently-computed ground truth).
- KPI **schema validity is 100%** (the object is constructed, not parsed from text).
- Results are **identical every run** — essential for a credible eval.

**Trade-off.** Less "magic" — the model isn't doing the math. That's the point:
**the math shouldn't be probabilistic.** The intelligence is in *interpretation*
(what's risky, what to say), not *calculation*.

**Rejected alternative.** Let the LLM compute metrics via tool-free reasoning —
rejected because it makes accuracy unmeasurable and non-reproducible.

> **Anticipated Q: "Isn't this barely using the LLM then?"**
> The LLM does the genuinely hard, fuzzy parts: extracting action items + owners +
> deadlines from free-form notes, deciding what matters, and writing the prose
> narrative. We deliberately *don't* outsource arithmetic to a probabilistic
> system — that's an engineering choice, not a limitation.

### 4.2 Two ReAct implementations, switchable by one env var

**Choice.** `REACT_MODE` selects between:
- **`custom`** (default for local Ollama) — an explicit reason→act→observe loop
  in `agent/react.py` where each step is a single `ReActStep` JSON object,
  Pydantic-validated, with retry.
- **`prebuilt`** — LangGraph's native `create_react_agent` tool-calling agent
  (`agent/graph.py`), ideal on Claude for the final demo.

**Why.** Local models (qwen2.5:14b via Ollama) emit native tool-call JSON
**inconsistently**, which makes the prebuilt agent flaky on the dev backend.
The custom loop constrains the model to emit **one validated JSON object per
step**; a hallucinated tool name fails `Literal` validation and triggers a retry
rather than crashing. Same tools, same schemas either way.

**Trade-off.** Two code paths to maintain. Justified because it lets the project
develop reliably for free locally, then use the stronger Claude tool-calling for
the polished demo — without a rewrite.

> **Anticipated Q: "Why not just always use the prebuilt LangGraph agent?"**
> It's less robust on local models. The custom loop is the reliability story:
> every step is schema-validated, and the loop is *inspectable* (that's what the
> live Process tab shows).

### 4.3 The export tool is the only external action — and it's gated

**Choice.** Of all tools, only `export_report` performs an external action
(writing a file). It **refuses to write** unless `human_approved` is True.
Moreover, the **autonomous ReAct loop cannot even select it** — `export_report`
is excluded from the loop's tool registry (`agent/react.py::_TOOL_REGISTRY`).

**Why.** This enforces the brief's "zero autonomous external actions" rule at
**two layers**:
1. The agent loop literally has no path to export.
2. Even via the dashboard, export is blocked until a human ticks approve.

**Trade-off.** The agent can't "finish the job" by itself. That's intentional and
is the safety guarantee.

> **Anticipated Q: "How do you *know* the agent can't act on its own?"**
> Two independent guarantees: (a) the tool isn't in the loop's registry, so it's
> unreachable in reasoning; (b) the tool checks `human_approved` and returns a
> BLOCKED message otherwise. You can demo (b) live by clicking export without
> approving — it refuses and writes nothing.

### 4.4 Strict Pydantic schemas as the validation contract

**Choice.** Every structured output is a Pydantic v2 model with
`extra="forbid"`, no `Any`, and field/model validators (`schemas/models.py`).
LLM output that doesn't conform raises `ValidationError`.

**Why.** This is the gate that keeps malformed/hallucinated output from ever
reaching the dashboard. Examples of what the schema *enforces*, not just hopes for:
- `ActionItem.owner` cannot be a placeholder ("TBD", "someone", "?") — a validator
  rejects it, so extracted action items have a real, named owner.
- `WorkloadEntry.done_tasks` cannot exceed `assigned_tasks`.
- `SprintKPIReport`: `completed_tasks ≤ total_tasks`, `sprint_end ≥ sprint_start`.

On failure, the validation wrapper **retries** (up to `max_validation_retries`)
with the error fed back to the model.

**Trade-off.** Strictness can cause retries (latency) on a weak model. Worth it:
a retry is cheap; a wrong report reaching a human as "validated" is not.

### 4.5 Local-first, free, with a one-line swap to Claude

**Choice.** `LLM_BACKEND` (`config.py`) switches the whole system between Ollama
(local) and Claude (API). Callers **never branch on backend** — they call
`get_chat_model()` (`agent/llm.py`) and get a LangChain chat model either way.
Embeddings stay **local even on the Claude path** (no paid embeddings).

**Why.** Develop and evaluate for free, unlimited, offline. Switch to Claude's
stronger reasoning only for the final live demo, where a few dollars is fine.
The abstraction means the swap is genuinely one line — no refactor, no risk.

**Trade-off.** Maintaining backend-agnostic interfaces. Cheap, and it's good
design anyway.

> **Anticipated Q: "Why qwen2.5:14b specifically?"**
> Best tool-use / reasoning quality in the size that fits 16 GB VRAM and runs
> fully on GPU. `llama3.1:8b` is the documented fallback for RAM-constrained
> machines. Both are configured via `OLLAMA_MODEL`.

### 4.6 Curated catalog *and* random generator for evaluation

**Choice.** Two scenario sources:
- **Curated catalog** (`eval/catalog.py` → `data/scenarios/`) — 13 hand-designed,
  documented scenarios (single blocker, bottleneck, deadline slips, scope creep,
  absentee dev, overloaded lead, large teams, end-of-sprint crunch, healthy
  control).
- **Random generator** (`eval/generate.py` → `data/synthetic/`) — volume/regression.

Both share an output format, so `eval/score.py` reads either.

**Why.** The curated set gives **explainable** scenarios for the demo and
defensible grading ("here's the absentee case, and the agent catches it"). The
random set gives **volume** for the 20+ scenario requirement and regression
safety.

**The credibility detail:** ground-truth risk labels are assigned
**by construction** in the generator, *independently of the detector*. So risk
precision/recall are **real measurements, not a tautology** — the thing being
measured didn't define its own answer key.

### 4.8 Reflection loop to reduce hallucination

**Choice.** After the ReAct loop drafts a final answer, a **self-critique pass**
(`agent/react.py::reflect_on_answer`) re-reads the answer against the exact tool
**observations** and produces a `ReflectionResult` — `grounded` (bool),
`issues` (list of unsupported claims), and a `revised_answer`. The revised answer
is what the user sees. Gated by `REACT_REFLECTION` (default on).

**Why.** The narrative is the one place the LLM writes freely, so it's where
hallucination can creep in (a wrong number, an invented risk). Because KPIs and
risks are **deterministic**, the observations are an authoritative answer key: any
claim they don't support is, by definition, unsupported. The critic catches and
rewrites those before finalizing. It's the **structured-output pattern applied to
the model's own output** — the check itself is a validated Pydantic object.

**Trade-off.** One extra LLM call per run (latency). It never blocks the run: any
failure degrades gracefully to the original draft (`reflect_on_answer` catches all
exceptions). Reflection *reduces*, not *eliminates*, hallucination — the hard
numbers are already guaranteed by the deterministic tools; this guards the prose.

> **Anticipated Q: "How does reflection actually reduce hallucination here?"**
> The tool observations are exact (deterministic KPIs/risks). The reflection step
> compares each claim in the draft narrative to those observations and rewrites
> anything unsupported. It's visible live in the Process tab as a
> "🔍 Reflection" step: either "✅ grounded" or "⚠️ corrected N claims".

### 4.9 Workload rebalancing: propose → validate → human applies

**Choice.** A `propose_rebalance` tool suggests moving open tasks off overloaded
members (or placing unassigned ones) onto teammates with spare capacity. It's a
**three-stage pipeline**, not a single LLM call:
1. **Deterministic analysis** (`agent/analytics.py::analyze_workload_balance`) —
   computes open points per member, the team mean, who is over/under-loaded, and
   the exact set of *movable* tasks.
2. **LLM proposal** — the model proposes specific moves *from that candidate set
   only*, each with a one-line reason, as a validated `RebalanceProposal`.
3. **Deterministic guardrail** — every suggestion is re-checked against the real
   tasks: unknown task ids or non-members are dropped, and `from_member`/`points`
   are **overwritten from ground truth** so the model can't misstate them.

Then the dashboard shows a **before/after workload chart** and a **human approval**
step. Nothing is ever auto-applied — it *proposes*, a human enacts.

**Why.** This is the same philosophy as the rest of the system: **let the LLM do
the judgment (which move makes sense, why), keep the facts deterministic (who's
overloaded, what's movable, how many points).** Stage 3 is an explicit
anti-hallucination guardrail — the LLM literally cannot propose a task that
doesn't exist or a point value that's wrong, because those fields are replaced
with the real ones after generation.

**Trade-off.** The LLM's freedom is deliberately narrow (it picks among vetted
moves, not arbitrary ones). That's the safety property.

> **Anticipated Q: "What if the model suggests a nonsensical or fake move?"**
> It's filtered out. Suggestions referencing a non-movable/unknown task or a
> non-member are dropped before display; the numeric fields come from the real
> task, not the model. And nothing is applied without the approval click.

### 4.7 Persistent ChromaDB + local embeddings with a fallback

**Choice.** RAG uses a persistent ChromaDB store. Embeddings prefer Ollama's
`nomic-embed-text`; if Ollama or the model is unavailable, it **falls back** to
`sentence-transformers` (`all-MiniLM-L6-v2`) — same interface
(`retrieval/embeddings.py`).

**Why.** Free, local, and **never blocks** dev/eval on Ollama being up. The
fallback means the pipeline degrades gracefully instead of failing.

---

## 5. Data flow (one request, end to end)

Tracing a "give me a status briefing" request through the custom ReAct loop:

```
1. Ingest    Scenario tasks → Task[] (validated Pydantic objects)
2. Retrieve  Tasks (+ notes) embedded into ChromaDB
                ↓
3. Reason    LangGraph loop:
                reason → "I should compute KPIs"      (ReActStep, validated)
                act    → compute_sprint_kpis()         (deterministic Python)
                observe→ exact KPI JSON
                reason → "Now detect risks"
                act    → detect_project_risks()         (deterministic Python)
                observe→ exact RiskAlert JSON
                reason → final_answer (narrative)
                reflect→ critique draft vs observations, revise unsupported claims
4. Validate  Each structured output parsed into its Pydantic model (retry on fail)
5. Review    Surfaced in the dashboard; status report editable;
             export BLOCKED until human approves
```

The deterministic builders are also callable **directly** (dashboard/eval), so
exact numbers never depend on the LLM choosing the right tool.

---

## 6. Component reference

| Module | Role |
|--------|------|
| `config.py` | Single source of truth: backend switch, model names, thresholds, paths. All env-overridable. |
| `schemas/models.py` | Pydantic v2 contract: `Task`, `ActionItem`, `MeetingSummary`, `RiskAlert`, `SprintKPIReport` (+`WorkloadEntry`), `StatusReport`. |
| `ingestion/` | Tabular loaders + Jira/Trello/GitHub field-mapping normalizer → internal `Task`. |
| `retrieval/store.py` | ChromaDB wrapper: embed tasks/notes, similarity query with metadata filter. |
| `retrieval/embeddings.py` | Local embedder (Ollama) with sentence-transformers fallback. |
| `agent/llm.py` | Backend-agnostic chat-model factory (Ollama \| Claude). |
| `agent/analytics.py` | **Deterministic** KPI computation, risk detection, and workload-balance analysis. |
| `agent/tools.py` | The tools (retrieve, summarize, KPIs, risks, `propose_rebalance`); `export_report` is approval-gated. |
| `agent/react.py` | Custom constrained ReAct loop, the `stream_custom_react` live generator, and the `reflect_on_answer` self-critique pass. |
| `agent/graph.py` | Prebuilt LangGraph tool-calling agent (Claude demo path). |
| `agent/validation.py` | `generate_structured`: LLM → Pydantic with retry. |
| `agent/reporting.py` | Assembles the weekly `StatusReport` + Markdown rendering. |
| `dashboard/app.py` | Streamlit review UI: scenario gallery, live agent trace, KPIs/risks, approval gate, export log. |
| `eval/` | Curated catalog, random generator, meeting-note set, scoring harness. |

---

## 7. Evaluation & success criteria

Measured by `eval/score.py` over the scenarios (Section 7 of the brief):

| Criterion | Target | How it's measured | Status |
|-----------|--------|-------------------|--------|
| KPI report schema validity | 100% | every report parses as `SprintKPIReport` | ✅ 100% |
| KPI numeric accuracy | 100% | vs independently-computed ground truth | ✅ 100% |
| Risk detection precision | > 85% | vs by-construction risk labels | ✅ (100% on curated) |
| Action-item extraction accuracy | > 90% | LLM output matched to planted items | ✅ 100% on meeting set |
| Full workflow time | < 60 s | wall-clock per scenario | ✅ deterministic path < 1s |
| Autonomous external actions | 0 | enforced by the approval gate | ✅ 0 (gated + unreachable) |

**Two measurement modes:**
- **Deterministic** (no LLM): runs offline/in CI — KPI validity/accuracy, risk
  precision/recall, timing. `python -m eval.score --dir data/scenarios`
- **LLM-dependent** (needs a backend up): action-item extraction, summary
  validity. `python -m eval.score --meetings data/meetings`

---

## 8. How to run it (quick reference)

```bash
# Models (local, free)
ollama pull qwen2.5:14b          # or llama3.1:8b on <16GB RAM
ollama pull nomic-embed-text

# Python env
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -r requirements.txt
cp .env.example .env              # defaults to local Ollama

# Eval (proof it works)
python -m eval.score --dir data/scenarios     # deterministic, no LLM
python -m eval.score --meetings data/meetings # action items (needs LLM)

# Dashboard (the demo)
streamlit run dashboard/app.py
```

For the final demo only:
```bash
export LLM_BACKEND=claude
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## 9. Anticipated questions — quick answers

**Q. Where's the "agent"? Looks like a pipeline.**
The agent is the ReAct loop in `agent/react.py`: it *reasons* about which tool to
call next, *acts*, *observes* the result, and decides the next step or to finish.
The deterministic builders are tools it *chooses to use* — watch the live trace in
the Process tab to see it decide.

**Q. Why constrain the loop so much? Real agents are more open-ended.**
Reliability on a local model. An open-ended loop on qwen2.5 produces inconsistent
tool-call JSON. Constraining each step to a validated `ReActStep` makes it robust
and inspectable. The `prebuilt` mode (open-ended native tool calling) exists for
Claude, switchable by one env var.

**Q. What stops it from doing something dangerous?**
The only external action (`export_report`) is (a) excluded from the agent's tool
registry, so it can't be chosen autonomously, and (b) hard-gated on
`human_approved`. Two independent layers.

**Q. How is the eval not circular?**
Risk labels are assigned by construction in the scenario generator, independently
of the detector that's being scored. The KPI ground truth is computed separately
from the production analytics. Neither answer key is written by the thing it grades.

**Q. Why local models instead of just using Claude?**
Free, unlimited, offline development and evaluation. The backend swap to Claude is
one line for the final demo. Good separation regardless.

**Q. What would you improve with more time?**
LangSmith tracing wired through both loops; more risk heuristics (dependency
chains, velocity trend); a Notion MCP push as a second *gated* external action;
broader edge-case scenarios.

---

## 10. Tech stack

LangGraph · Ollama (qwen2.5:14b / llama3.1:8b) · ChromaDB · Pydantic v2 ·
Streamlit · Plotly · Pandas · SQLite · local embeddings (nomic-embed-text /
sentence-transformers) · LangSmith (free tier) · Claude API (final demo only).

100% free and local during development.
