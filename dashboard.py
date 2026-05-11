"""
IPIE Live Match Dashboard
Run: streamlit run dashboard.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.kafka_consumer import SimulatedMatchStream
from pipeline.momentum_detector import MomentumDetector
from llm.llm_caller import LLMCaller, _template_narrative
from llm.prompt_builder import MatchSnapshot
from data.ingest.cricbuzz_client import MatchState

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPIE — IPL Predictive Intelligence Engine",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

IPL_TEAMS = ["MI", "CSK", "RCB", "DC", "KKR", "SRH", "PBKS", "RR", "GT", "LSG"]

TEAM_COLORS = {
    "MI":   "#004BA0",
    "CSK":  "#F5A623",
    "RCB":  "#EC1C24",
    "DC":   "#0078BC",
    "KKR":  "#3A225D",
    "SRH":  "#F7A721",
    "PBKS": "#ED1B24",
    "RR":   "#EA1A85",
    "GT":   "#1C1C1C",
    "LSG":  "#A0E0F0",
}

SQUAD: dict[str, list[str]] = {
    "MI":   ["Rohit Sharma", "Ishan Kishan", "Suryakumar Yadav", "Hardik Pandya", "Bumrah"],
    "CSK":  ["Ruturaj Gaikwad", "Devon Conway", "Shivam Dube", "MS Dhoni", "Deepak Chahar"],
    "RCB":  ["Faf du Plessis", "Virat Kohli", "Glenn Maxwell", "Dinesh Karthik", "Siraj"],
    "DC":   ["Prithvi Shaw", "David Warner", "Rishabh Pant", "Axar Patel", "Kuldeep Yadav"],
    "KKR":  ["Shreyas Iyer", "Venkatesh Iyer", "Andre Russell", "Sunil Narine", "Varun Chakravarthy"],
    "SRH":  ["Mayank Agarwal", "Abhishek Sharma", "Aiden Markram", "Heinrich Klaasen", "Bhuvneshwar"],
    "PBKS": ["Shikhar Dhawan", "Jonny Bairstow", "Liam Livingstone", "Sam Curran", "Arshdeep Singh"],
    "RR":   ["Jos Buttler", "Yashasvi Jaiswal", "Sanju Samson", "Shimron Hetmyer", "Trent Boult"],
    "GT":   ["Shubman Gill", "Wriddhiman Saha", "Hardik Pandya", "David Miller", "Mohammed Shami"],
    "LSG":  ["KL Rahul", "Quinton de Kock", "Marcus Stoinis", "Deepak Hooda", "Mohsin Khan"],
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏏 IPIE Dashboard")
    st.markdown("---")

    team1 = st.selectbox("Team 1 (batting first)", IPL_TEAMS, index=0)
    team2 = st.selectbox("Team 2", IPL_TEAMS, index=1)

    if team1 == team2:
        st.error("Teams must be different")
        st.stop()

    seed = st.number_input("Random seed", value=42, min_value=0, max_value=9999, step=1)
    delay = st.slider("Over delay (seconds)", 0.0, 3.0, 0.3, 0.1,
                      help="Pause between overs for live feel")

    st.markdown("---")
    start = st.button("▶ Start Simulation", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("No API keys required. Simulation mode active.")


# ── Main layout ───────────────────────────────────────────────────────────────
st.title(f"🏏 {team1}  vs  {team2}")
st.caption(f"Wankhede Stadium · seed={seed} · Simulation mode")

col_t1, col_vs, col_t2 = st.columns([5, 1, 5])
with col_t1:
    t1_prob = st.metric(label=f"{team1} Win Probability", value="50%", delta=None)
with col_vs:
    st.markdown("<h3 style='text-align:center;margin-top:20px'>VS</h3>", unsafe_allow_html=True)
with col_t2:
    t2_prob = st.metric(label=f"{team2} Win Probability", value="50%", delta=None)

st.markdown("---")

# Chart placeholder
chart_placeholder = st.empty()

st.markdown("---")

# Over stats row
col_inn, col_score, col_wkts, col_rr, col_rrr = st.columns(5)
inn_box    = col_inn.empty()
score_box  = col_score.empty()
wkts_box   = col_wkts.empty()
rr_box     = col_rr.empty()
rrr_box    = col_rrr.empty()

st.markdown("---")

# Narrative + confidence
col_narr, col_conf = st.columns([3, 1])
narr_box = col_narr.empty()
conf_box = col_conf.empty()

st.markdown("---")

# Bottom panels
col_captain, col_risk, col_momentum = st.columns(3)
captain_box  = col_captain.empty()
risk_box     = col_risk.empty()
momentum_box = col_momentum.empty()

st.markdown("---")

# Over history table
st.subheader("Over History")
table_placeholder = st.empty()

# Player impact
st.subheader("Player Impact Scores")
impact_placeholder = st.empty()


def _build_chart(history: list[dict], team1: str, team2: str) -> go.Figure:
    overs  = [f"Inn{h['innings']} Ov{h['over']}" for h in history]
    p1     = [h["win_probability"][team1] * 100 for h in history]
    p2     = [h["win_probability"][team2] * 100 for h in history]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        subplot_titles=("Win Probability %", "Score Progression"),
        vertical_spacing=0.1,
    )

    c1 = TEAM_COLORS.get(team1, "#1f77b4")
    c2 = TEAM_COLORS.get(team2, "#ff7f0e")

    fig.add_trace(go.Scatter(
        x=overs, y=p1, name=team1,
        line=dict(color=c1, width=3),
        fill="tozeroy", fillcolor=f"rgba({int(c1[1:3],16)},{int(c1[3:5],16)},{int(c1[5:7],16)},0.15)",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=overs, y=p2, name=team2,
        line=dict(color=c2, width=3),
        fill="tozeroy", fillcolor=f"rgba({int(c2[1:3],16)},{int(c2[3:5],16)},{int(c2[5:7],16)},0.15)",
    ), row=1, col=1)

    # 50% line
    fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)

    # Momentum alerts
    for h in history:
        if h.get("momentum_alert"):
            fig.add_vline(
                x=f"Inn{h['innings']} Ov{h['over']}",
                line_dash="dot", line_color="yellow", opacity=0.8,
            )

    # Score bars
    scores = [h["score"] for h in history]
    fig.add_trace(go.Bar(
        x=overs, y=scores,
        name="Score",
        marker_color=[c1 if h["innings"] == 1 else c2 for h in history],
        opacity=0.7,
        showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis2=dict(tickangle=-45, tickfont=dict(size=9)),
    )
    fig.update_yaxes(range=[0, 100], row=1, col=1, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(row=2, col=1, gridcolor="rgba(255,255,255,0.1)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")

    return fig


def _build_impact_chart(impact: dict, team1: str, team2: str) -> go.Figure:
    squad1 = SQUAD.get(team1, [])
    squad2 = SQUAD.get(team2, [])
    players = squad1 + squad2
    scores  = [impact.get(p, 0) for p in players]
    colors  = [TEAM_COLORS.get(team1, "#1f77b4")] * len(squad1) + \
              [TEAM_COLORS.get(team2, "#ff7f0e")] * len(squad2)

    fig = go.Figure(go.Bar(
        x=players, y=scores,
        marker_color=colors,
        text=[f"{s:.0f}" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=11),
        margin=dict(l=20, r=20, t=20, b=80),
        yaxis=dict(range=[0, 110], gridcolor="rgba(255,255,255,0.1)"),
        xaxis=dict(tickangle=-30),
    )
    return fig


def _render_state(history: list[dict], team1: str, team2: str) -> None:
    if not history:
        return

    h   = history[-1]
    p1  = h["win_probability"][team1]
    p2  = h["win_probability"][team2]
    prev_p1 = history[-2]["win_probability"][team1] if len(history) > 1 else p1

    col_t1.metric(
        label=f"{team1} Win Probability",
        value=f"{p1:.0%}",
        delta=f"{(p1 - prev_p1):+.1%}",
    )
    col_t2.metric(
        label=f"{team2} Win Probability",
        value=f"{p2:.0%}",
        delta=f"{(p2 - (1 - prev_p1)):+.1%}",
    )

    chart_placeholder.plotly_chart(_build_chart(history, team1, team2), use_container_width=True)

    inn_box.metric("Innings", h["innings"])
    score_box.metric("Score", f"{h['score']}/{h['wickets']}")
    wkts_box.metric("Wickets", h["wickets"])
    rr_box.metric("Run Rate", f"{h.get('run_rate', 0):.2f}")
    rrr_val = h.get("rrr")
    rrr_box.metric("RRR", f"{rrr_val:.2f}" if rrr_val else "—")

    conf = h.get("confidence_tag", "LOW")
    conf_color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}.get(conf, "gray")
    narr_box.info(f"**Over {h['over']}:** {h.get('llm_narrative', '')}")
    conf_box.markdown(
        f"<div style='padding:16px;border-radius:8px;background:{conf_color};"
        f"text-align:center;font-weight:bold;font-size:18px'>{conf}</div>",
        unsafe_allow_html=True,
    )

    captain_box.success(f"**Fantasy Captain**\n\n{h.get('fantasy_captain_pick', 'TBD')}")
    risk_box.warning(f"**Key Risk**\n\n{h.get('key_risk_flag', '—')}")

    alert = h.get("momentum_alert")
    if alert:
        d = alert.get("delta", 0)
        momentum_box.error(f"**MOMENTUM ALERT**\n\nd = {d:+.0%}")
    else:
        total_alerts = sum(1 for x in history if x.get("momentum_alert"))
        momentum_box.info(f"**Momentum Alerts**\n\n{total_alerts} so far")

    # History table
    rows = []
    for x in history:
        rows.append({
            "Inn": x["innings"],
            "Over": x["over"],
            "Score": f"{x['score']}/{x['wickets']}",
            "RR": x.get("run_rate", 0),
            f"{team1} %": f"{x['win_probability'][team1]:.0%}",
            f"{team2} %": f"{x['win_probability'][team2]:.0%}",
            "Conf": x.get("confidence_tag", ""),
            "Alert": "🚨" if x.get("momentum_alert") else "",
        })
    table_placeholder.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Impact chart
    impact = h.get("player_impact_score", {})
    if impact:
        impact_placeholder.plotly_chart(
            _build_impact_chart(impact, team1, team2), use_container_width=True
        )


# ── Simulation runner ─────────────────────────────────────────────────────────
if start:
    rng      = np.random.default_rng(int(seed))
    momentum = MomentumDetector()
    history: list[dict] = []

    # Pre-generate probability curve (Brownian motion, same as simulate.py)
    p = 0.5
    prob_curve: list[float] = []
    for _ in range(40):
        step = rng.normal(0, 0.06)
        p = p + step - 0.02 * (p - 0.5)
        p = float(np.clip(p, 0.05, 0.95))
        prob_curve.append(round(p, 4))

    state_vars   = {"over_index": 0, "prev_prob": 0.5}
    stream       = SimulatedMatchStream(f"DASH-{team1}-{team2}", team1, team2, int(seed))

    progress = st.progress(0, text="Simulating...")

    def on_over(event: dict) -> None:
        prob = prob_curve[min(state_vars["over_index"], len(prob_curve) - 1)]
        state_vars["over_index"] += 1
        delta = prob - state_vars["prev_prob"]
        state_vars["prev_prob"] = prob

        inn  = event["innings"]
        over = event["over"]

        state = MatchState(
            match_id=event["match_id"],
            team1=team1, team2=team2,
            innings=inn, over=over, ball=6,
            score=event["score"], wickets=event["wickets"],
            run_rate=event["run_rate"],
            target=event.get("target"), rrr=event.get("rrr"),
            batting_team=team1 if inn == 1 else team2,
            bowling_team=team2 if inn == 1 else team1,
        )

        squad1 = SQUAD.get(team1, [f"{team1} P{i}" for i in range(1, 6)])
        squad2 = SQUAD.get(team2, [f"{team2} P{i}" for i in range(1, 6)])

        snapshot = MatchSnapshot(
            match_id=event["match_id"],
            team1=team1, team2=team2,
            innings=inn, over=over,
            score=event["score"], wickets=event["wickets"],
            target=event.get("target"), rrr=event.get("rrr"),
            run_rate=event["run_rate"],
            batting_team=state.batting_team,
            bowling_team=state.bowling_team,
            win_probability_team1=prob,
            key_batsmen=squad1[:2],
            key_bowler=squad2[-1],
            venue="Wankhede Stadium",
            pitch_type="batting_paradise",
            dew_flag=over > 14,
            team1_form="W W L W W",
            team2_form="L W W W L",
            player_impact_scores={squad1[0]: round(float(rng.uniform(55, 90)), 1)},
            momentum_delta=delta if over > 1 else None,
        )

        llm_out        = _template_narrative(snapshot)
        momentum_alert = momentum.check(state, delta)

        all_players = squad1 + squad2
        impact_scores = {pl: round(float(rng.uniform(30, 95)), 1) for pl in all_players}

        payload = {
            "match_id": event["match_id"],
            "over": over,
            "innings": inn,
            "score": event["score"],
            "wickets": event["wickets"],
            "run_rate": event["run_rate"],
            "target": event.get("target"),
            "rrr": event.get("rrr"),
            "win_probability": {team1: prob, team2: round(1 - prob, 4)},
            "llm_narrative": llm_out.llm_narrative,
            "confidence_tag": llm_out.confidence_tag,
            "key_risk_flag": llm_out.key_risk_flag,
            "fantasy_captain_pick": llm_out.fantasy_captain_pick,
            "momentum_alert": momentum_alert,
            "player_impact_score": impact_scores,
        }

        history.append(payload)
        _render_state(history, team1, team2)

        total_overs = 40
        done = min(state_vars["over_index"], total_overs)
        progress.progress(done / total_overs, text=f"Inn {inn} · Over {over}/20")

        if delay > 0:
            time.sleep(delay)

    stream.stream(on_over)
    progress.progress(1.0, text="Match complete!")

    # Final result
    final = history[-1]
    p1    = final["win_probability"][team1]
    winner = team1 if p1 >= 0.5 else team2
    alerts = sum(1 for h in history if h.get("momentum_alert"))

    st.success(
        f"**Match Over!** Predicted winner: **{winner}** "
        f"({max(p1, 1-p1):.0%} win probability) · {alerts} momentum alerts"
    )

else:
    # Idle state — show empty chart placeholder
    chart_placeholder.info("Configure teams in the sidebar and click **▶ Start Simulation**.")
    table_placeholder.info("Over history will appear here during simulation.")
