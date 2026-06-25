"""Streamlit human-in-the-loop review dashboard.

Central UI where all agent outputs surface for review. Nothing is exported or
finalized without explicit approval here -- this is the enforcement point for the
"zero autonomous external actions" rule. KPIs are visualized with Plotly.

Run:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import datetime as _dt
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from agent.analytics import compute_kpis, detect_risks
from agent.reporting import build_status_report, status_report_to_markdown
from config import settings
from ingestion import load_tabular, normalize_tasks
from schemas import RiskSeverity, Task

st.set_page_config(page_title="Project Workflow Intelligence", layout="wide")

SEVERITY_COLORS = {
    RiskSeverity.CRITICAL.value: "#b91c1c",
    RiskSeverity.HIGH.value: "#ea580c",
    RiskSeverity.MEDIUM.value: "#ca8a04",
    RiskSeverity.LOW.value: "#16a34a",
}


# --------------------------------------------------------------------------- #
# Data loading helpers
# --------------------------------------------------------------------------- #
def _tasks_from_scenario(scenario: dict) -> list[Task]:
    return [Task.model_validate(t) for t in scenario["tasks"]]


def _load_uploaded(upload) -> list[Task]:
    suffix = upload.name.split(".")[-1].lower()
    tmp = f"/tmp/{upload.name}"
    with open(tmp, "wb") as fh:
        fh.write(upload.getbuffer())
    df = load_tabular(tmp)
    return normalize_tasks(df)


# --------------------------------------------------------------------------- #
# Sidebar: backend banner + data source
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Configuration")
backend_label = "🟢 Claude (demo)" if settings.backend == "claude" else "🔵 Ollama (local)"
st.sidebar.metric("LLM backend", backend_label)
st.sidebar.caption(f"Model: `{settings.active_model_name()}`")
st.sidebar.caption(f"Human approval required: **{settings.require_human_approval}**")

st.sidebar.divider()
st.sidebar.subheader("Data source")
source = st.sidebar.radio("Load tasks from", ["Synthetic scenario", "Upload CSV/Excel"])

tasks: list[Task] = []
sprint = "Sprint"
sprint_start = sprint_end = _dt.date.today()
today = _dt.date.today()
meeting_note = ""

if source == "Synthetic scenario":
    from config import SYNTHETIC_DIR

    files = sorted(SYNTHETIC_DIR.glob("*.json"))
    if not files:
        st.sidebar.warning("No scenarios. Run: `python -m eval.generate -n 24`")
    else:
        pick = st.sidebar.selectbox("Scenario", [f.stem for f in files])
        scenario = json.loads((SYNTHETIC_DIR / f"{pick}.json").read_text())
        tasks = _tasks_from_scenario(scenario)
        sprint = scenario["sprint"]
        sprint_start = _dt.date.fromisoformat(scenario["sprint_start"])
        sprint_end = _dt.date.fromisoformat(scenario["sprint_end"])
        today = _dt.date.fromisoformat(scenario["today"])
        meeting_note = scenario["meeting_note"]
else:
    upload = st.sidebar.file_uploader("Task export", type=["csv", "xlsx", "xls", "tsv"])
    if upload:
        tasks = _load_uploaded(upload)
        sprint = tasks[0].sprint if tasks and tasks[0].sprint else "Sprint"
    meeting_note = st.sidebar.text_area("Paste meeting notes (optional)", height=150)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.title("🧭 AI Project Workflow Intelligence")
st.caption("Human-reviewed outputs. Nothing is exported without your approval.")

if not tasks:
    st.info("Load a synthetic scenario or upload a task export to begin.")
    st.stop()

kpis = compute_kpis(tasks, sprint, sprint_start, sprint_end)
risks = detect_risks(tasks, today=today)

tab_kpi, tab_risk, tab_meeting, tab_report = st.tabs(
    ["📊 KPIs", "⚠️ Risks", "📝 Meeting", "📤 Status Report"]
)

# ---- KPIs ----
with tab_kpi:
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
    st.subheader(f"{len(risks)} risk(s) detected")
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

# ---- Meeting summary (HITL edit) ----
with tab_meeting:
    if not meeting_note.strip():
        st.info("No meeting notes for this scenario.")
    else:
        st.text_area("Source notes", meeting_note, height=180, disabled=True)
        if st.button("Summarize with agent"):
            with st.spinner(f"Summarizing via {settings.active_model_name()}…"):
                try:
                    from agent.tools import AgentContext, set_context, summarize_meeting
                    set_context(AgentContext(
                        meeting_text=meeting_note, meeting_source=f"{sprint}-notes",
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
    st.subheader("Weekly status report")
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
        st.markdown("### 🔒 Approval gate")
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
