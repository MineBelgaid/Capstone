"""Streamlit human-in-the-loop review dashboard.

Central UI where all agent outputs surface for review. Nothing is exported or
finalized without explicit approval here -- this is the enforcement point for the
"zero autonomous external actions" rule. KPIs are visualized with Plotly.

For demos/grading it also surfaces:
  * the curated scenario catalog (each with its documented *purpose* and the
    by-construction ground truth the agent is measured against), and
  * a live view of the agent's ReAct process (thought -> tool -> observation),
    so the reasoning is inspectable as it happens.

Run:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import datetime as _dt
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from agent.analytics import (
    analyze_workload_balance,
    compute_kpis,
    detect_risks,
    open_points_by_member,
)
from agent.reporting import build_status_report, status_report_to_markdown
from config import DATA_DIR, SYNTHETIC_DIR, settings
from ingestion import load_tabular, normalize_tasks
from schemas import RiskSeverity, Task

st.set_page_config(page_title="Project Workflow Intelligence", layout="wide")

SCENARIO_DIR = DATA_DIR / "scenarios"      # curated catalog (documented purposes)
MEETING_DIR = DATA_DIR / "meetings"        # standalone meeting-note tests

SEVERITY_COLORS = {
    RiskSeverity.CRITICAL.value: "#b91c1c",
    RiskSeverity.HIGH.value: "#ea580c",
    RiskSeverity.MEDIUM.value: "#ca8a04",
    RiskSeverity.LOW.value: "#16a34a",
}

# The 5-stage pipeline from the architecture (README). Surfaced as a banner so a
# viewer can place each tab within the overall flow.
PIPELINE = ["Ingest", "Retrieve", "Reason", "Validate", "Review"]


# --------------------------------------------------------------------------- #
# Data loading helpers
# --------------------------------------------------------------------------- #
def _tasks_from_scenario(scenario: dict) -> list[Task]:
    return [Task.model_validate(t) for t in scenario["tasks"]]


def _load_uploaded(upload) -> list[Task]:
    suffix = upload.name.split(".")[-1].lower()  # noqa: F841 - kept for clarity
    tmp = f"/tmp/{upload.name}"
    with open(tmp, "wb") as fh:
        fh.write(upload.getbuffer())
    df = load_tabular(tmp)
    return normalize_tasks(df)


def _pipeline_banner(active: str) -> None:
    """Render the Ingest->Retrieve->Reason->Validate->Review flow, highlighting
    the stage the current view corresponds to."""
    cells = []
    for stage in PIPELINE:
        if stage == active:
            cells.append(
                f"<span style='background:#2563eb;color:#fff;padding:4px 12px;"
                f"border-radius:6px;font-weight:700'>{stage}</span>"
            )
        else:
            cells.append(
                f"<span style='background:#1f2937;color:#9ca3af;padding:4px 12px;"
                f"border-radius:6px'>{stage}</span>"
            )
    arrow = "<span style='color:#6b7280;margin:0 6px'>&rarr;</span>"
    st.markdown(
        "<div style='margin:4px 0 14px 0'>" + arrow.join(cells) + "</div>",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _load_catalog_summaries() -> list[dict]:
    """Read every curated scenario and extract display metadata for the gallery."""
    out: list[dict] = []
    for f in sorted(SCENARIO_DIR.glob("*.json")):
        s = json.loads(f.read_text())
        gt = s.get("ground_truth", {})
        ek = gt.get("expected_kpis", {})
        out.append({
            "id": s.get("scenario_id", f.stem),
            "stem": f.stem,
            "intent": s.get("intent", ""),
            "team_size": s.get("team_size", len({t.get("assignee") for t in s.get("tasks", [])})),
            "total_tasks": len(s.get("tasks", [])),
            "planted_risks": len(gt.get("risky_tasks", [])),
            "velocity": ek.get("velocity", 0),
            "completion": ek.get("completion_rate", 0),
        })
    return out


@st.cache_resource(show_spinner=False)
def _build_knowledge_store(cache_key: str, _tasks: list[Task], _note: str, _source: str):
    """Build a per-scenario ChromaDB store (tasks + optional meeting note) so the
    live agent's ``retrieve_context`` RAG tool has real data to ground on.

    Cached per scenario (``cache_key``) and wrapped by the caller in try/except so
    a missing embedding backend degrades gracefully to "RAG off" rather than
    breaking the demo. The leading-underscore args are excluded from Streamlit's
    cache hashing (the key already identifies the scenario)."""
    from retrieval.store import KnowledgeStore

    store = KnowledgeStore()
    store.reset()
    store.add_tasks(_tasks)
    if _note.strip():
        store.add_meeting_note(source=_source, text=_note)
    return store


def _render_step(ev: dict) -> None:
    """Render one streamed ReAct event (reason step, observation, or reflection)."""
    if ev["type"] == "reason":
        st.markdown(f"**Step {ev['n']} · \U0001f9e0 Thought**  \n{ev['thought']}")
        st.markdown(f"→ \U0001f527 **Action:** `{ev['tool']}({ev['tool_input']!r})`")
    elif ev["type"] == "observe":
        obs = ev["observation"]
        with st.expander(f"\U0001f441️ Observation (step {ev['n']})", expanded=False):
            try:
                st.json(json.loads(obs))
            except Exception:  # noqa: BLE001
                st.code(obs)
        st.divider()
    elif ev["type"] == "reflect":
        if ev.get("grounded"):
            st.markdown("**\U0001f50d Reflection** — ✅ answer is grounded in the "
                        "tool observations (no unsupported claims).")
        else:
            issues = ev.get("issues", [])
            st.markdown(f"**\U0001f50d Reflection** — ⚠️ corrected "
                        f"{len(issues)} unsupported claim(s) before finalizing:")
            for it in issues:
                st.markdown(f"- {it}")
            with st.expander("Draft answer before reflection", expanded=False):
                st.markdown(ev.get("draft", ""))
        st.divider()


def _render_trace(events: list[dict], answer, meta: dict) -> None:
    """Redraw a persisted agent run from session_state (survives reruns)."""
    if meta:
        st.caption(f"Last run · scenario **{meta.get('scenario', '?')}** · task: "
                   f"_{(meta.get('query', '') or '')[:90]}…_")
    for ev in events:
        _render_step(ev)
    if answer is not None:
        st.success("✅ Final answer")
        st.markdown(answer)


# --------------------------------------------------------------------------- #
# Sidebar: backend banner + data source
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Configuration")
backend_label = "\U0001f7e2 Claude (demo)" if settings.backend == "claude" else "\U0001f535 Ollama (local)"
st.sidebar.metric("LLM backend", backend_label)
st.sidebar.caption(f"Model: `{settings.active_model_name()}`")
st.sidebar.caption(f"ReAct mode: `{settings.agent.react_mode}`")
st.sidebar.caption(f"Human approval required: **{settings.require_human_approval}**")

st.sidebar.divider()
st.sidebar.subheader("Data source")

# Allow the Scenarios gallery to drive the sidebar selection. The gallery (which
# renders later in the script than these widgets) can't set their state directly,
# so it stashes a request in ``_pending_pick`` and reruns; we apply it here BEFORE
# the widgets are instantiated, which Streamlit permits.
if "_pending_pick" in st.session_state:
    st.session_state["data_source"] = "Curated catalog"
    st.session_state["catalog_pick"] = st.session_state.pop("_pending_pick")

source = st.sidebar.radio(
    "Load tasks from",
    ["Curated catalog", "Random synthetic", "Upload CSV/Excel"],
    key="data_source",
    help="Curated = hand-designed scenarios with documented purpose + ground "
         "truth (best for the demo). Random = generated volume/regression set.",
)

tasks: list[Task] = []
sprint = "Sprint"
sprint_start = sprint_end = _dt.date.today()
today = _dt.date.today()
meeting_note = ""
meeting_source = "meeting_notes"
scenario_meta: dict = {}     # intent + ground truth for curated scenarios
scenario_id = ""

if source in ("Curated catalog", "Random synthetic"):
    folder = SCENARIO_DIR if source == "Curated catalog" else SYNTHETIC_DIR
    files = sorted(folder.glob("*.json"))
    if not files:
        if source == "Curated catalog":
            st.sidebar.warning("No catalog. Run: `python -m eval.catalog`")
        else:
            st.sidebar.warning("No scenarios. Run: `python -m eval.generate -n 24`")
    else:
        stems = [f.stem for f in files]
        key = "catalog_pick" if source == "Curated catalog" else "synthetic_pick"
        # If a stale pending pick isn't in this folder, fall back to the first.
        if key in st.session_state and st.session_state[key] not in stems:
            st.session_state[key] = stems[0]
        pick = st.sidebar.selectbox("Scenario", stems, key=key)
        scenario = json.loads((folder / f"{pick}.json").read_text())
        tasks = _tasks_from_scenario(scenario)
        sprint = scenario["sprint"]
        sprint_start = _dt.date.fromisoformat(scenario["sprint_start"])
        sprint_end = _dt.date.fromisoformat(scenario["sprint_end"])
        today = _dt.date.fromisoformat(scenario["today"])
        meeting_note = scenario.get("meeting_note", "")
        meeting_source = f"{sprint}-notes"
        scenario_id = scenario.get("scenario_id", pick)
        scenario_meta = {
            "intent": scenario.get("intent", ""),
            "ground_truth": scenario.get("ground_truth", {}),
        }
        # Curated sprint scenarios keep their meeting notes in data/meetings; let
        # the user attach one so the Meeting tab + RAG have something to chew on.
        if not meeting_note.strip() and MEETING_DIR.exists():
            mfiles = sorted(MEETING_DIR.glob("*.json"))
            if mfiles:
                names = ["(none)"] + [f.stem for f in mfiles]
                mpick = st.sidebar.selectbox("Attach meeting note (optional)", names)
                if mpick != "(none)":
                    m = json.loads((MEETING_DIR / f"{mpick}.json").read_text())
                    meeting_note = m.get("meeting_note", "")
                    meeting_source = m.get("scenario_id", mpick)
else:
    upload = st.sidebar.file_uploader("Task export", type=["csv", "xlsx", "xls", "tsv"])
    if upload:
        tasks = _load_uploaded(upload)
        sprint = tasks[0].sprint if tasks and tasks[0].sprint else "Sprint"
    meeting_note = st.sidebar.text_area("Paste meeting notes (optional)", height=150)
    meeting_source = "uploaded-notes"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.title("\U0001f9ed AI Project Workflow Intelligence")
st.caption("Human-reviewed outputs. Nothing is exported without your approval.")

if not tasks:
    st.info("Load a scenario from the curated catalog (left) or upload a task export to begin.")
    st.stop()

# Switching scenarios must not leave a stale report/summary/trace from the
# previous one cached in session_state. Detect a change and clear those outputs
# so every tab reflects the scenario currently selected in the sidebar.
_active_key = scenario_id or sprint or "uploaded"
if st.session_state.get("_active_scenario") != _active_key:
    st.session_state["_active_scenario"] = _active_key
    for _k in ("report_md", "summary", "process_trace", "process_answer",
               "process_meta", "rebalance"):
        st.session_state.pop(_k, None)

# Scenario briefing: what this scenario is for + what the agent *should* find.
if scenario_meta.get("intent"):
    with st.container(border=True):
        st.markdown(f"**\U0001f3af Scenario purpose** · `{scenario_id}`")
        st.write(scenario_meta["intent"])
        gt = scenario_meta.get("ground_truth", {})
        risky = gt.get("risky_tasks", [])
        ekpis = gt.get("expected_kpis", {})
        gcols = st.columns(3)
        gcols[0].metric("Expected risky tasks", len(risky))
        if ekpis:
            gcols[1].metric("Expected velocity", f"{ekpis.get('velocity', 0):.0f}")
            gcols[2].metric("Expected completion", f"{ekpis.get('completion_rate', 0):.0%}")
        if risky:
            with st.expander("Ground truth — the risks this scenario plants (by construction)"):
                st.dataframe(pd.DataFrame(risky), use_container_width=True, hide_index=True)
        st.caption("Ground truth is assigned independently of the detector, so the "
                   "match below is a real measurement, not a tautology.")

kpis = compute_kpis(tasks, sprint, sprint_start, sprint_end)
risks = detect_risks(tasks, today=today)

(tab_scenarios, tab_process, tab_kpi, tab_risk, tab_rebalance,
 tab_meeting, tab_report) = st.tabs(
    ["\U0001f4c2 Scenarios", "\U0001f52c Agent Process (live)", "\U0001f4ca KPIs",
     "⚠️ Risks", "⚖️ Rebalance", "\U0001f4dd Meeting", "\U0001f4e4 Status Report"]
)

# ---- Scenario gallery ----
with tab_scenarios:
    _pipeline_banner("Ingest")
    st.subheader("Curated scenario catalog")
    st.caption(
        "13 hand-designed sprints (teams of 3–15), each probing a specific "
        "coordination failure mode. Every scenario carries by-construction ground "
        "truth, so the agent's risk detection is a real measurement. Pick one to "
        "load it into every tab."
    )
    summaries = _load_catalog_summaries()
    if not summaries:
        st.warning("No catalog found. Generate it with: `python -m eval.catalog`")
    else:
        cols = st.columns(3)
        for i, s in enumerate(summaries):
            with cols[i % 3]:
                with st.container(border=True):
                    selected = (s["stem"] == scenario_id)
                    badge = " ✅" if selected else ""
                    st.markdown(f"**{s['id']}**{badge}")
                    st.caption(s["intent"])
                    m = st.columns(2)
                    m[0].metric("Team", s["team_size"])
                    m[1].metric("Tasks", s["total_tasks"])
                    m2 = st.columns(2)
                    m2[0].metric("Planted risks", s["planted_risks"])
                    m2[1].metric("Completion", f"{s['completion']:.0%}")
                    if selected:
                        st.success("Loaded", icon="\U0001f4cd")
                    else:
                        if st.button("Use this scenario", key=f"use_{s['stem']}",
                                     use_container_width=True):
                            st.session_state["_pending_pick"] = s["stem"]
                            st.rerun()

# ---- Agent Process (live ReAct trace) ----
with tab_process:
    _pipeline_banner("Reason")
    st.subheader("Watch the agent reason")
    st.caption(
        "The agent runs an explicit reason → act → observe loop on "
        f"`{settings.active_model_name()}`. Each step is one Pydantic-validated "
        "decision; deterministic tools return exact numbers (never hallucinated). "
        "The loop **cannot** export — that stays behind the approval gate."
    )

    default_q = (
        "Give the team a status briefing for this sprint: retrieve relevant "
        "context, compute the KPIs, detect the risks, and summarize what needs "
        "attention."
    )
    query = st.text_area("Agent task", default_q, height=80)
    use_rag = st.toggle(
        "Enable RAG (build a vector store from this scenario)", value=True,
        help="Embeds tasks + any attached meeting note into ChromaDB so the "
             "retrieve_context tool has real data. Off = the agent relies on the "
             "deterministic KPI/risk tools only.",
    )

    run_col, clear_col = st.columns([3, 1])
    run_clicked = run_col.button("▶️ Run agent", type="primary", use_container_width=True)
    if clear_col.button("\U0001f9f9 Clear run", use_container_width=True):
        for _k in ("process_trace", "process_answer", "process_meta"):
            st.session_state.pop(_k, None)
        st.rerun()

    if run_clicked:
        from agent.tools import AgentContext, set_context
        from agent.react import stream_custom_react

        store = None
        if use_rag:
            with st.spinner("Building vector store (embedding tasks + notes)…"):
                try:
                    store = _build_knowledge_store(
                        scenario_id or sprint, tasks, meeting_note, meeting_source
                    )
                    st.caption(f"\U0001f5c2️ Knowledge store ready: {store.count()} chunks "
                               f"embedded via `{settings.embed_model_name()}`.")
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"RAG disabled (embedding backend unavailable): {exc}")
                    store = None

        set_context(AgentContext(
            tasks=tasks, store=store, meeting_text=meeting_note,
            meeting_source=meeting_source, sprint=sprint,
            sprint_start=sprint_start, sprint_end=sprint_end, today=today,
        ))

        st.divider()
        trace_box = st.container()
        events: list[dict] = []
        answer = None
        try:
            with st.spinner("Agent is reasoning… (local model, may take a moment)"):
                for ev in stream_custom_react(query):
                    if ev["type"] == "final":
                        answer = ev["answer"]
                    else:
                        events.append(ev)
                        with trace_box:
                            _render_step(ev)
            # Persist so the run survives tab switches / later reruns.
            st.session_state["process_trace"] = events
            st.session_state["process_answer"] = answer
            st.session_state["process_meta"] = {
                "scenario": scenario_id or sprint, "query": query,
            }
            with trace_box:
                if answer is not None:
                    st.success("✅ Final answer")
                    st.markdown(answer)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Agent run failed (is the Ollama backend up?): {exc}")
    elif st.session_state.get("process_trace"):
        st.divider()
        st.caption("\U0001f4cc Showing your last agent run (kept across tabs and "
                   "reruns). Press **Run agent** to refresh, or **Clear run**.")
        _render_trace(
            st.session_state["process_trace"],
            st.session_state.get("process_answer"),
            st.session_state.get("process_meta", {}),
        )
    else:
        st.info("Press **Run agent** to watch the reason → act → observe "
                "loop execute step by step on the local model.")

# ---- KPIs ----
with tab_kpi:
    _pipeline_banner("Validate")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Velocity (pts)", f"{kpis.velocity:.0f}")
    c2.metric("Completion", f"{kpis.completion_rate:.0%}")
    c3.metric("Tasks done", f"{kpis.completed_tasks}/{kpis.total_tasks}")
    c4.metric("Team size", len({t.assignee for t in tasks if t.assignee}))

    wdf = pd.DataFrame([w.model_dump() for w in kpis.workload])
    fig = px.bar(
        wdf, x="member", y="assigned_points", color="done_tasks",
        title="Workload by team member (open points & completed tasks)",
        labels={"assigned_points": "Assigned points", "member": "Member"},
    )
    st.plotly_chart(fig, use_container_width=True)

    status_counts = pd.Series([t.status.value for t in tasks]).value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    st.plotly_chart(
        px.pie(status_counts, names="status", values="count", title="Task status mix"),
        use_container_width=True,
    )

# ---- Risks ----
with tab_risk:
    _pipeline_banner("Validate")
    st.subheader(f"{len(risks)} risk(s) detected")
    # Show how the detector lines up with the scenario's planted ground truth.
    if scenario_meta.get("ground_truth", {}).get("risky_tasks"):
        truth_ids = {r["task_id"] for r in scenario_meta["ground_truth"]["risky_tasks"]}
        flagged_ids = {tid for r in risks for tid in r.related_task_ids}
        tp = len(flagged_ids & truth_ids)
        precision = tp / len(flagged_ids) if flagged_ids else 1.0
        recall = tp / len(truth_ids) if truth_ids else 1.0
        mc = st.columns(3)
        mc[0].metric("Precision vs ground truth", f"{precision:.0%}")
        mc[1].metric("Recall vs ground truth", f"{recall:.0%}")
        mc[2].metric("Flagged / planted", f"{len(flagged_ids)} / {len(truth_ids)}")
        st.divider()
    if not risks:
        st.success("No risks flagged for this sprint.")
    for i, r in enumerate(risks):
        color = SEVERITY_COLORS.get(r.severity.value, "#666")
        with st.container(border=True):
            st.markdown(
                f"<span style='color:{color};font-weight:700'>[{r.severity.value.upper()}]</span> "
                f"**{r.area}**", unsafe_allow_html=True,
            )
            st.write(r.reason)
            st.caption(f"Recommended: {r.recommended_action}")
            cols = st.columns(2)
            cols[0].checkbox("Accept", key=f"acc_{i}", value=True)
            cols[1].checkbox("Dismiss", key=f"dis_{i}")

# ---- Workload rebalancing (deterministic analysis + LLM proposal + approval) ----
with tab_rebalance:
    _pipeline_banner("Reason")
    st.subheader("Workload rebalancing")
    st.caption(
        "Deterministic analysis finds who is overloaded and who has spare capacity; "
        f"the LLM (`{settings.active_model_name()}`) proposes specific task moves, "
        "each validated against the real tasks (it can't invent a task, owner, or "
        "point value). **Proposals only — nothing is reassigned without your approval.**"
    )

    bal = analyze_workload_balance(tasks)
    bdf = pd.DataFrame([
        {"member": m, "open_points": v["open_points"], "open_tasks": v["open_tasks"]}
        for m, v in bal["load"].items()
    ])
    if not bdf.empty:
        fig_bal = px.bar(
            bdf, x="member", y="open_points", color="open_tasks",
            title="Current open workload (story points per member)",
            labels={"open_points": "Open points", "member": "Member"},
        )
        fig_bal.add_hline(
            y=bal["mean_open_points"], line_dash="dash",
            annotation_text=f"team mean {bal['mean_open_points']}",
        )
        st.plotly_chart(fig_bal, use_container_width=True)
    cbal = st.columns(2)
    cbal[0].metric("Overloaded", ", ".join(bal["overloaded"]) or "none")
    cbal[1].metric("Spare capacity", ", ".join(bal["underloaded"]) or "none")

    if st.button("\U0001f916 Propose rebalancing", type="primary"):
        from agent.tools import AgentContext, set_context, propose_rebalance
        with st.spinner(f"Proposing via {settings.active_model_name()}…"):
            try:
                set_context(AgentContext(tasks=tasks))
                st.session_state["rebalance"] = json.loads(
                    propose_rebalance.invoke({"_": ""})
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Proposal failed (is the LLM backend up?): {exc}")

    if "rebalance" in st.session_state:
        prop = st.session_state["rebalance"]
        st.info(prop.get("summary", ""))
        suggestions = prop.get("suggestions", [])
        if not suggestions:
            st.success("No reassignments needed — workload is already balanced.")
        else:
            st.markdown("#### Proposed moves — accept the ones you want")
            accepted = []
            for i, s in enumerate(suggestions):
                cc = st.columns([0.12, 0.88])
                keep = cc[0].checkbox("Accept", key=f"rb_{i}", value=True)
                cc[1].markdown(
                    f"**{s['task_id']}** · {s['task_title']} — move "
                    f"**{s['points']:.0f} pts** from `{s['from_member']}` → "
                    f"`{s['to_member']}`  \n_{s['reason']}_"
                )
                if keep:
                    accepted.append(s)

            # Deterministic before/after preview of the accepted moves.
            before = open_points_by_member(tasks)
            after = dict(before)
            for s in accepted:
                after[s["from_member"]] = after.get(s["from_member"], 0.0) - s["points"]
                after[s["to_member"]] = after.get(s["to_member"], 0.0) + s["points"]
            rows = []
            for m in sorted(set(before) | set(after)):
                rows.append({"member": m, "state": "before",
                             "open_points": round(before.get(m, 0.0), 1)})
                rows.append({"member": m, "state": "after",
                             "open_points": round(after.get(m, 0.0), 1)})
            fig_ba = px.bar(
                pd.DataFrame(rows), x="member", y="open_points", color="state",
                barmode="group", title="Workload before vs after accepted moves",
            )
            st.plotly_chart(fig_ba, use_container_width=True)

            st.divider()
            st.markdown("### \U0001f512 Approval")
            st.caption("Applying reassignments is a plan for a human to enact — the "
                       "agent never changes assignments autonomously.")
            ok = st.checkbox("I approve this rebalancing plan")
            if st.button("Approve plan", disabled=not ok, type="primary"):
                st.success(f"✅ Plan approved: {len(accepted)} reassignment(s) endorsed. "
                           "Recorded as a proposal — no external system was modified.")

# ---- Meeting summary (HITL edit) ----
with tab_meeting:
    _pipeline_banner("Reason")
    if not meeting_note.strip():
        st.info("No meeting notes for this scenario. Attach one in the sidebar "
                "(curated scenarios) or paste notes (upload mode).")
    else:
        st.text_area("Source notes", meeting_note, height=180, disabled=True)
        if st.button("Summarize with agent"):
            with st.spinner(f"Summarizing via {settings.active_model_name()}…"):
                try:
                    from agent.tools import AgentContext, set_context, summarize_meeting
                    set_context(AgentContext(
                        meeting_text=meeting_note, meeting_source=meeting_source,
                    ))
                    payload = json.loads(summarize_meeting.invoke({"_": ""}))
                    st.session_state["summary"] = payload
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Summarization failed (is the LLM backend up?): {exc}")
        if "summary" in st.session_state:
            s = st.session_state["summary"]
            st.markdown("#### Editable summary")
            st.text_area("Summary", s.get("summary", ""), key="edit_summary", height=120)
            st.markdown("**Action items** (review before approving):")
            st.dataframe(pd.DataFrame(s.get("action_items", [])), use_container_width=True)

# ---- Status report + approval gate ----
with tab_report:
    _pipeline_banner("Review")
    st.subheader("Weekly status report")
    st.caption(f"Active scenario: **{scenario_id or sprint}**. Click *Build report* "
               "to (re)generate it for this scenario before approving.")
    use_llm = st.toggle("Generate narrative with LLM", value=False,
                        help="Off = deterministic template (no LLM needed).")
    if st.button("Build report"):
        with st.spinner("Building report…"):
            try:
                report = build_status_report(
                    tasks, sprint, sprint_start, sprint_end,
                    today=today, use_llm=use_llm,
                )
                st.session_state["report_md"] = status_report_to_markdown(report)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Report build failed: {exc}")

    if "report_md" in st.session_state:
        edited = st.text_area(
            "Review & edit before approval", st.session_state["report_md"], height=320,
        )
        st.divider()
        st.markdown("### \U0001f512 Approval gate")
        st.caption("Export is blocked until you approve. This enforces zero "
                   "autonomous external actions.")
        approved = st.checkbox("I have reviewed and approve this report")
        col_a, col_b = st.columns(2)
        if col_a.button("Approve & export", disabled=not approved, type="primary"):
            from agent.tools import AgentContext, set_context, export_report
            set_context(AgentContext(human_approved=True))
            result = export_report.invoke({"markdown": edited})
            st.success(result)
        if col_b.button("Reject"):
            st.warning("Report rejected. Nothing exported.")
        # Demonstrate the block when not approved:
        if not approved:
            from agent.tools import AgentContext, set_context, export_report
            set_context(AgentContext(human_approved=False))
            st.info(export_report.invoke({"markdown": edited}))

    # ---- Exported reports log -------------------------------------------- #
    # Visible proof of the approval gate: a file lands here ONLY after a human
    # approved it above. Always shown, so the demo can point at concrete output.
    st.divider()
    st.markdown("### \U0001f4c1 Exported reports")
    exports_dir = DATA_DIR / "exports"
    export_files = sorted(exports_dir.glob("*.md"), reverse=True) if exports_dir.exists() else []
    if not export_files:
        st.caption("No reports exported yet. Approve a report above and it will "
                   "appear here.")
    else:
        st.caption(f"{len(export_files)} approved export(s) in `data/exports/`. "
                   "A report reaches this list only after passing the human "
                   "approval gate — never autonomously.")
        for f in export_files:
            md = f.read_text(encoding="utf-8")
            heading = md.splitlines()[0].lstrip("# ").strip() if md.strip() else f.stem
            ts = _dt.datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            with st.expander(f"\U0001f4c4 {heading}  ·  {ts}  ·  {f.name}"):
                st.markdown(md)
                st.download_button(
                    "⬇️ Download .md", md, file_name=f.name,
                    mime="text/markdown", key=f"dl_{f.name}",
                )
