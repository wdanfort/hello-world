"""Streamlit dashboard: 7-tab interactive prediction viewer."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Path helpers (dashboard reads from data/ relative to repo root)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"


def _pred_dir(show: str, season: int) -> Path:
    return DATA_DIR / show / f"s{season}" / "predictions"


def _ep_dir(show: str, season: int) -> Path:
    return DATA_DIR / show / f"s{season}" / "episodes"


def _backfill_dir(show: str, season: int) -> Path:
    return ROOT / "backfill" / show / f"s{season}"


def load_latest(show: str, season: int) -> Optional[dict]:
    p = _pred_dir(show, season) / "latest.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_history(show: str, season: int) -> Optional[dict]:
    p = _pred_dir(show, season) / "history.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_episode_analysis(show: str, season: int, ep: int) -> Optional[dict]:
    p = _ep_dir(show, season) / f"episode_{ep:02d}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_calibration(show: str, season: int) -> Optional[dict]:
    p = _backfill_dir(show, season) / "calibration_summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_sim_results(show: str, season: int) -> Optional[dict]:
    # Sim results saved via CLI to backfill dir
    p = _backfill_dir(show, season) / "simulation_results.json"
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Survivor Prediction Model",
    page_icon="🏝️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar: show/season selector
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🏝️ Survivor Predictor")
    show = st.selectbox("Show", ["survivor", "traitors"], index=0)
    season = st.number_input("Season", min_value=1, max_value=100, value=50)
    st.divider()
    st.caption("Data refreshes automatically via GitHub Actions after each episode.")

latest = load_latest(show, season)
history_data = load_history(show, season)

if latest is None:
    st.warning(f"No prediction data found for {show.title()} Season {season}. Run the pipeline first.")
    st.stop()

generated_at = latest.get("generated_at", "unknown")
episode = latest.get("episode_number", "?")

st.title(f"🏝️ {show.title()} Season {season} — Prediction Model")
st.caption(f"Last updated: {generated_at} | Latest episode: {episode}")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

(
    tab_odds,
    tab_elim,
    tab_trends,
    tab_track,
    tab_analysis,
    tab_sim,
    tab_live,
) = st.tabs([
    "📊 Current Odds",
    "🔥 Elimination",
    "📈 Trends",
    "🎯 Track Record",
    "🔍 Episode Analysis",
    "💰 Bet Simulator",
    "🔴 Live Replay",
])


# ---------------------------------------------------------------------------
# Tab 1 — Current Odds
# ---------------------------------------------------------------------------

with tab_odds:
    contestants_data = latest.get("contestants", {})
    market_odds = latest.get("market_odds", {})

    if not contestants_data:
        st.info("No contestant data found.")
    else:
        rows = []
        for name, data in contestants_data.items():
            row = {
                "Contestant": name,
                "Win% (Model)": data.get("win_prob", 0) * 100,
                "Win% (Market)": (market_odds.get(name, 0) or 0) * 100 if market_odds else None,
                "Elim%": data.get("elim_prob", 0) * 100,
            }
            if market_odds and market_odds.get(name):
                row["Delta"] = row["Win% (Model)"] - row["Win% (Market)"]
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("Win% (Model)", ascending=False)

        # Bar chart
        fig_cols = ["Win% (Model)"]
        if market_odds:
            fig_cols.append("Win% (Market)")

        fig = px.bar(
            df,
            x="Contestant",
            y=fig_cols,
            barmode="group",
            title=f"Winner Probability — Episode {episode}",
            labels={"value": "Probability (%)", "variable": "Source"},
            color_discrete_sequence=["#00b4d8", "#f77f00"],
        )
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

        # Table with mispricings highlighted
        display_df = df.copy()
        display_df["Win% (Model)"] = display_df["Win% (Model)"].map("{:.1f}%".format)
        if "Win% (Market)" in display_df.columns and display_df["Win% (Market)"].notna().any():
            display_df["Win% (Market)"] = display_df["Win% (Market)"].map(
                lambda x: f"{x:.1f}%" if x else "—"
            )
        if "Delta" in display_df.columns:
            display_df["Delta"] = display_df["Delta"].map(
                lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%" if x else "—"
            )
        display_df["Elim%"] = display_df["Elim%"].map("{:.1f}%".format)

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Mispricings
        if market_odds and "Delta" in df.columns:
            mispricings = df[df["Delta"].abs() >= 10].sort_values("Delta", key=abs, ascending=False)
            if not mispricings.empty:
                st.subheader("⚠️ Mispricings (|Δ| > 10%)")
                for _, row in mispricings.iterrows():
                    direction = "📈 underpriced" if row["Delta"] > 0 else "📉 overpriced"
                    st.write(
                        f"**{row['Contestant']}** — model {row['Win% (Model)']}, "
                        f"market {row['Win% (Market)']} → {direction} by {abs(row['Delta']):.1f}%"
                    )


# ---------------------------------------------------------------------------
# Tab 2 — Elimination Forecast
# ---------------------------------------------------------------------------

with tab_elim:
    contestants_data = latest.get("contestants", {})
    df_elim = pd.DataFrame([
        {"Contestant": k, "Elimination%": v.get("elim_prob", 0) * 100}
        for k, v in contestants_data.items()
    ]).sort_values("Elimination%", ascending=False)

    fig_elim = px.bar(
        df_elim,
        x="Contestant",
        y="Elimination%",
        title=f"Next Elimination Probability — Episode {episode}",
        color="Elimination%",
        color_continuous_scale="Reds",
    )
    fig_elim.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_elim, use_container_width=True)

    # Calibration from backfill if available
    calibration = load_calibration(show, season)
    if calibration and calibration.get("calibration"):
        st.subheader("📋 Elimination Accuracy")
        top1 = calibration.get("elimination_accuracy_top1")
        top3 = calibration.get("elimination_accuracy_top3")
        col1, col2, col3 = st.columns(3)
        col1.metric("Top-1 Accuracy", f"{top1 * 100:.1f}%" if top1 else "—")
        col2.metric("Top-3 Accuracy", f"{top3 * 100:.1f}%" if top3 else "—")
        col3.metric("Episodes Evaluated", calibration.get("episodes_processed", "—"))


# ---------------------------------------------------------------------------
# Tab 3 — Trends
# ---------------------------------------------------------------------------

with tab_trends:
    if not history_data:
        st.info("No history data yet. Run multiple episodes to see trends.")
    else:
        episodes_list = history_data.get("episodes", [])
        if not episodes_list:
            st.info("No episode history available yet.")
        else:
            # Build wide dataframe for win probability trends
            records = []
            for ep_data in episodes_list:
                ep_num = ep_data.get("episode_number")
                for name, data in ep_data.get("contestants", {}).items():
                    records.append({
                        "Episode": ep_num,
                        "Contestant": name,
                        "Win Probability (%)": data.get("win_prob", 0) * 100,
                        "Elimination Risk (%)": data.get("elim_prob", 0) * 100,
                    })
            df_hist = pd.DataFrame(records)

            # Top 5 by latest win prob
            latest_ep = df_hist["Episode"].max()
            top5 = (
                df_hist[df_hist["Episode"] == latest_ep]
                .nlargest(5, "Win Probability (%)")["Contestant"]
                .tolist()
            )

            st.subheader("🏆 Win Probability Trends (Top 5)")
            df_top5 = df_hist[df_hist["Contestant"].isin(top5)]
            fig_trend = px.line(
                df_top5,
                x="Episode",
                y="Win Probability (%)",
                color="Contestant",
                markers=True,
                title="Win Probability by Episode",
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            st.subheader("🎬 Edit Visibility Trends")
            # Load raw edit visibility from episode analysis files
            edit_records = []
            for ep_num in df_hist["Episode"].unique():
                ep_analysis = load_episode_analysis(show, season, int(ep_num))
                if ep_analysis:
                    for name, data in ep_analysis.get("contestants", {}).items():
                        ev = data.get("edit_visibility")
                        if ev is not None:
                            edit_records.append({"Episode": ep_num, "Contestant": name, "Edit Visibility": ev})
            if edit_records:
                df_edit = pd.DataFrame(edit_records)
                df_edit_top5 = df_edit[df_edit["Contestant"].isin(top5)]
                fig_edit = px.line(
                    df_edit_top5,
                    x="Episode",
                    y="Edit Visibility",
                    color="Contestant",
                    markers=True,
                    title="Edit Visibility Score by Episode (1-10)",
                )
                st.plotly_chart(fig_edit, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4 — Track Record
# ---------------------------------------------------------------------------

with tab_track:
    calibration = load_calibration(show, season)
    if not calibration:
        st.info("No calibration data yet. Run backfill first.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Top-1 Elim Accuracy", f"{calibration.get('elimination_accuracy_top1', 0) * 100:.1f}%")
        col2.metric("Top-3 Elim Accuracy", f"{calibration.get('elimination_accuracy_top3', 0) * 100:.1f}%")
        col3.metric("Episodes Evaluated", calibration.get("episodes_processed", 0))

        # Calibration scatter: predicted prob vs actual outcome
        cal_entries = calibration.get("calibration", [])
        if cal_entries:
            cal_rows = []
            for entry in cal_entries:
                actual = entry.get("actual_eliminated")
                if actual:
                    assigned = entry.get("assigned_elim_prob", 0)
                    correct = entry.get("correct_elim", False)
                    cal_rows.append({
                        "Episode": entry["episode"],
                        "Contestant": actual,
                        "Predicted Elim %": assigned * 100,
                        "Correct": "Yes" if correct else "No",
                    })
            df_cal = pd.DataFrame(cal_rows)
            if not df_cal.empty:
                st.subheader("📊 Predicted vs Actual Elimination Probability")
                fig_cal = px.scatter(
                    df_cal,
                    x="Episode",
                    y="Predicted Elim %",
                    color="Correct",
                    symbol="Correct",
                    hover_data=["Contestant"],
                    color_discrete_map={"Yes": "green", "No": "red"},
                    title="Assigned Elimination Probability for Actual Boots",
                )
                st.plotly_chart(fig_cal, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 5 — Episode Analysis
# ---------------------------------------------------------------------------

with tab_analysis:
    if not history_data:
        st.info("No episode data yet.")
    else:
        available_eps = sorted(
            {ep.get("episode_number") for ep in history_data.get("episodes", []) if ep.get("episode_number")}
        )
        if not available_eps:
            st.info("No episodes available yet.")
        else:
            selected_ep = st.selectbox("Select Episode", available_eps, index=len(available_eps) - 1)
            ep_analysis = load_episode_analysis(show, season, selected_ep)
            if ep_analysis is None:
                st.warning(f"No analysis file for Episode {selected_ep}.")
            else:
                st.subheader(f"Episode {selected_ep} — LLM Analysis")
                st.write(f"**Summary:** {ep_analysis.get('episode_summary', 'N/A')}")

                key_events = ep_analysis.get("key_events", [])
                if key_events:
                    st.write("**Key Events:**")
                    for ev in key_events:
                        st.write(f"- {ev}")

                st.divider()
                st.subheader("Per-Contestant Scores")

                score_rows = []
                for name, data in ep_analysis.get("contestants", {}).items():
                    row = {"Contestant": name}
                    scoring_keys = [
                        "edit_visibility", "edit_sentiment", "strategic_positioning",
                        "threat_perception", "alliance_strength", "elimination_risk", "winner_signals"
                    ]
                    for key in scoring_keys:
                        row[key.replace("_", " ").title()] = data.get(key, "—")
                    row["Narrative"] = data.get("narrative_summary", "")
                    score_rows.append(row)

                df_scores = pd.DataFrame(score_rows)
                st.dataframe(df_scores, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 6 — Bet Simulator
# ---------------------------------------------------------------------------

with tab_sim:
    sim_results = load_sim_results(show, season)
    if sim_results is None:
        st.info(
            "No simulation results yet. Run:\n\n"
            f"`python -m src.simulation.simulator --season {season}`"
        )
    else:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Starting Bankroll", f"${sim_results['starting_bankroll']:.2f}")
        col2.metric("Ending Bankroll", f"${sim_results['ending_bankroll']:.2f}")
        col3.metric("ROI", f"{sim_results['roi_pct']:.1f}%")
        col4.metric(
            "Win Rate",
            f"{sim_results['win_rate'] * 100:.1f}%" if sim_results.get("win_rate") else "—"
        )
        col5.metric("Max Drawdown", f"{sim_results['max_drawdown_pct']:.1f}%")

        # Bankroll history
        bh = sim_results.get("bankroll_history", [])
        if bh:
            df_bh = pd.DataFrame(bh)
            fig_bh = px.line(
                df_bh,
                x="episode",
                y="bankroll",
                markers=True,
                title="Simulated Bankroll Over Time",
                labels={"episode": "Episode", "bankroll": "Bankroll ($)"},
            )
            fig_bh.add_hline(y=sim_results["starting_bankroll"], line_dash="dash", line_color="gray")
            st.plotly_chart(fig_bh, use_container_width=True)

        # Trades table
        trades = sim_results.get("trades", [])
        if trades:
            st.subheader("All Trades")
            df_trades = pd.DataFrame(trades)
            st.dataframe(df_trades, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 7 — Live Mode Replay
# ---------------------------------------------------------------------------

with tab_live:
    live_dir = DATA_DIR / show / f"s{season}" / "live"
    timeline_files = sorted(live_dir.glob("episode_*_kalshi_timeline.json")) if live_dir.exists() else []

    if not timeline_files:
        st.info("No live episode data recorded yet.")
    else:
        ep_options = [f.stem.replace("_kalshi_timeline", "") for f in timeline_files]
        selected_tl = st.selectbox("Select Live Episode", ep_options)
        tl_path = live_dir / f"{selected_tl}_kalshi_timeline.json"
        timeline_data = json.loads(tl_path.read_text())

        timeline = timeline_data.get("timeline", [])
        if not timeline:
            st.info("Empty timeline.")
        else:
            # Build dataframe
            records = []
            for snap in timeline:
                ts = snap["timestamp"]
                for name, d in snap.get("odds", {}).items():
                    records.append({
                        "Time": ts,
                        "Contestant": name,
                        "Market Probability (%)": d.get("price", 0) * 100,
                    })
            df_tl = pd.DataFrame(records)
            fig_tl = px.line(
                df_tl,
                x="Time",
                y="Market Probability (%)",
                color="Contestant",
                title="Kalshi Odds During Live Episode",
                markers=True,
            )
            st.plotly_chart(fig_tl, use_container_width=True)
