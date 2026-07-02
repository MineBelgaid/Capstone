# Live Demo Script (~15 minutes)

> Companion to [ARCHITECTURE.md](ARCHITECTURE.md) (the "why" for Q&A) and the
> [README](README.md). This is the **click-path + talking points** for the demo.

---

## Before you present (2 min, do this offstage)

1. **Launch everything with one command:**
   - Windows: `.\run.ps1`
   - macOS/Linux: `./run.sh`

   This verifies Python, Ollama, the models, the venv, and `.env`, starts the
   Ollama server if needed, then opens the dashboard. To only verify without
   launching: `.\run.ps1 -Check` (or `./run.sh --check`).

   > If Windows blocks the script with an execution-policy error, run once:
   > `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` — then `.\run.ps1`.

2. Open **http://localhost:8501** in your browser.
3. **Warm up the model** so the first live call isn't a cold start: go to
   **🔬 Agent Process**, pick a scenario, click **Run agent** once, let it finish,
   then click **🧹 Clear run**. (Keeps the model resident in VRAM.)
4. Have [ARCHITECTURE.md](ARCHITECTURE.md) open in a second tab for Q&A.

**One-line pitch to open with:**
> "A single agent that turns raw project data into reviewed, structured
> insights — where the numbers are guaranteed correct because they're computed,
> not guessed, and nothing leaves the system without a human approving it."

---

## Timeline

| # | Segment | Time | Tab |
|---|---------|------|-----|
| 1 | Problem & framing | 1.5 min | — |
| 2 | Architecture overview | 2 min | ARCHITECTURE.md |
| 3 | Scenario catalog | 1.5 min | 📂 Scenarios |
| 4 | Live agent reasoning + reflection | 3.5 min | 🔬 Agent Process |
| 5 | KPIs & risks vs ground truth | 2 min | 📊 KPIs / ⚠️ Risks |
| 6 | Workload rebalancing | 2 min | ⚖️ Rebalance |
| 7 | Meeting summary | 1 min | 📝 Meeting |
| 8 | Status report + approval gate | 1.5 min | 📤 Status Report |
| 9 | Backend swap + close | 1 min | sidebar |
|   | **Total** | **~15 min** | |

---

## 1 · Problem & framing (1.5 min)

**Say:**
- Small teams lose hours to coordination: chasing status, spotting blockers late,
  writing the same weekly reports, digging action items out of meeting notes.
- Goal: one agent that ingests project data and produces **structured, reviewable**
  outputs — KPIs, risks, meeting summaries, status reports — under a hard rule:
  **zero autonomous external actions.** A human approves anything before it leaves.
- Everything runs **locally and free** (Ollama + qwen2.5:14b on the GPU); the LLM
  backend swaps to Claude for the final demo with a one-line change.

**Point at the sidebar:** backend badge (🔵 Ollama / model name), "Human approval
required: True".

---

## 2 · Architecture overview (2 min)

**Show** the ASCII pipeline in ARCHITECTURE.md (§2) and say:

- Five stages: **Ingest → Retrieve → Reason → Validate → Review** (point out the
  banner that appears on each dashboard tab).
- The one design choice that ties it together: **deterministic analytics, LLM only
  for narration.** KPIs and risks are computed in Python; the model interprets and
  writes prose. That's why KPI accuracy is 100% and reproducible.
- Agentic patterns present: **ReAct, tool use, RAG, human-in-the-loop, structured
  output/validation, and reflection.** Each maps to a specific part of the code
  (table in ARCHITECTURE.md §3).

---

## 3 · Scenario catalog (1.5 min)  → tab: 📂 Scenarios

**Do:** Scroll the card grid.

**Say:**
- 13 hand-designed sprints, each probing one failure mode — single blocker,
  bottleneck, deadline slips, scope creep, **absentee developer**, **overloaded
  lead**, large teams, end-of-sprint crunch, plus a healthy control.
- Each carries **ground truth assigned by construction** — so when we measure risk
  detection later, it's a real measurement, not the system grading its own homework.

**Do:** On the **`CUR-07-absentee`** card, click **Use this scenario**.

**Say:** "Its purpose card says a developer went inactive and their work went
stale — let's see if the agent catches exactly that."

---

## 4 · Live agent reasoning + reflection (3.5 min)  → tab: 🔬 Agent Process

**This is the centerpiece — slow down here.**

**Do:** Keep RAG enabled. Click **▶️ Run agent**.

**Narrate as the steps stream in:**
- "This is an explicit **reason → act → observe** loop. Watch — the model *thinks*,
  then *chooses a tool*, then *sees the result*, then decides the next step."
- On a `compute_sprint_kpis` / `detect_project_risks` step: "It's calling a
  **deterministic tool** — these numbers are computed in Python, so they're exact.
  The model decided to use them; it isn't inventing them."
- Expand one **👁️ Observation** to show the real JSON it reasoned over.
- On the **🔍 Reflection** step: "Before finalizing, the agent **self-critiques**:
  it re-checks every claim in its answer against those exact observations and
  corrects anything unsupported. This is our anti-hallucination guard on the
  free-text — either '✅ grounded' or 'corrected N claims'."
- Read the **✅ Final answer**.

**Say (payoff):** "Every 'Thought' you saw was a schema-validated object — if the
model emitted a bad tool name or malformed JSON, it fails validation and retries
instead of crashing. That's the reliability story on a local model."

**Then:** Switch to another tab and back to show the run **persists** — "I don't
have to re-run it; the trace is kept."

---

## 5 · KPIs & risks vs ground truth (2 min)  → tabs: 📊 KPIs, ⚠️ Risks

**📊 KPIs — Do:** Show the metric cards + workload bar + status pie.
**Say:** "All computed deterministically — velocity is summed completed points,
not an estimate."

**⚠️ Risks — Do:** Point at the top row: **Precision / Recall vs ground truth**.
**Say:** "The detector flagged the absentee's stale tasks. Precision and recall are
measured against the planted labels — here it catches exactly what the scenario
planted. Because the labels were assigned independently of the detector, this is a
genuine measurement."

---

## 6 · Workload rebalancing (2 min)  → tab: ⚖️ Rebalance

**Do:** Switch scenario to **`CUR-09-overloaded-lead`** (sidebar or Scenarios tab),
then open ⚖️ Rebalance.

**Say:**
- "Deterministic analysis shows one member — Ava — holding most of the open points,
  way above the team mean (the dashed line)."
- Click **🤖 Propose rebalancing**. "The LLM proposes **specific moves** off the
  overloaded member onto teammates with capacity, each with a reason."
- "Crucially, every suggestion is **validated against the real tasks** — it can't
  invent a task or misstate points; those fields come from ground truth. The AI
  **proposes**, it never reassigns on its own."
- Toggle a couple of **Accept** boxes and show the **before/after chart** rebalance.
- Tick **I approve** → **Approve plan**: "A human enacts it. Same
  zero-autonomous-action rule as everywhere else."

---

## 7 · Meeting summary (1 min)  → tab: 📝 Meeting

**Do:** In the sidebar, with a curated scenario selected, use **Attach meeting note**
to pick e.g. `MTG-05-postmortem`. Open 📝 Meeting → **Summarize with agent**.

**Say:** "Here the LLM does the genuinely hard, fuzzy work: pulling **action items
with a named owner and deadline** out of free-form notes. The schema *forbids*
placeholder owners like 'TBD' — so every action item is actionable. On our meeting
set this hits 100% extraction accuracy."

---

## 8 · Status report + approval gate (1.5 min)  → tab: 📤 Status Report

**Do:**
1. Toggle **Generate narrative with LLM** on. Click **Build report**.
2. Scroll to the **🔒 Approval gate**. **Leave the box unchecked** and point at the
   blue **BLOCKED** message: "Export is refused — nothing is written."
3. Tick **I approve** → **Approve & export**. Show the green success message.
4. Scroll to **📁 Exported reports**: "The file appears here *only* after approval,
   with a timestamp. This panel is the visible proof of the human-in-the-loop rule."

**Say:** "Two independent guarantees: the autonomous loop **can't even select** the
export tool, and the tool itself **refuses** without approval."

---

## 9 · Backend swap + close (1 min)

**Say:**
- "Everything you saw ran on a **local, free** model on the GPU. For the final live
  demo, switching to Claude is **one line** — `LLM_BACKEND=claude` — no refactor,
  because every call goes through one backend-agnostic factory."
- **Close:** "So: correct-by-construction numbers, an inspectable reasoning loop, a
  reflection pass against hallucination, and a hard human-approval gate on every
  external action. The intelligence is in the interpretation; the guarantees are in
  the engineering."

---

## Q&A — where to look

The hard questions and crisp answers are in **[ARCHITECTURE.md §9](ARCHITECTURE.md)**
and the per-decision "Anticipated Q" callouts (§4). The five most likely:

- **"Where's the agent / isn't this a pipeline?"** → §9, the ReAct loop in the
  Process tab.
- **"Isn't the LLM barely doing anything?"** → §4.1 — it does extraction + judgment
  + narration; arithmetic is deliberately deterministic.
- **"How is the eval not circular?"** → §4.6 — ground truth assigned by construction.
- **"How do you know it can't act autonomously?"** → §4.3 — two layers.
- **"How does reflection reduce hallucination?"** → §4.8 — critique vs exact
  observations.

---

## If something goes wrong (fallbacks)

- **A live LLM step is slow / hangs:** the model may have cold-started. Mention it's
  loading into VRAM; the deterministic tabs (KPIs, Risks, Rebalance analysis) don't
  need the LLM and are instant. You warmed it up in prep to avoid this.
- **Ollama not responding:** in a terminal, `.\run.ps1 -Check` re-verifies and
  restarts the server.
- **A generation looks off:** that's a fine moment to show the **🔍 Reflection**
  step or the **deterministic guardrail** catching it — the safeguards are the point.
- **Worst case:** the deterministic eval is offline proof:
  `python -m eval.score --dir data/scenarios` → all targets pass, no LLM needed.
