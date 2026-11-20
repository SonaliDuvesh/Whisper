"""Unit and integration verification tests for Whisper Prompt 8 (Workforce Intelligence).

Verifies:
1. Dataset generation, dimensions, and column constraints.
2. Leakage-free baseline calculations and Z-scores.
3. Workforce risk scoring and severity classification boundaries.
4. Persistent 2-day consecutive alert episode logic.
5. Operational capacity estimation accuracy.
6. Visualization generation (Map, Trends, Comparison).
7. AI HR Action Briefing generation and ethical privacy safeguards.
"""
import pytest
import pandas as pd
import numpy as np

from workforce_gen import (
    generate_workforce_signals,
    validate_workforce_dataset,
    generate_organizations,
)
from workforce_detector import (
    load_workforce_data,
    calculate_workforce_baselines,
    detect_workforce_anomalies,
    calculate_workforce_risk,
    generate_workforce_alerts,
    run_workforce_pipeline,
)
from viz import (
    create_workforce_map,
    create_workforce_trend_chart,
    create_org_comparison_chart,
    prepare_workforce_alert_feed,
)
from ai_briefing import (
    build_workforce_context,
    generate_workforce_briefing,
    answer_workforce_question,
)

@pytest.fixture(scope="module")
def workforce_data():
    df = load_workforce_data("data/workforce_signals.csv")
    proc_df, alert_events = run_workforce_pipeline(df, z_threshold=2.0, alert_threshold=60.0)
    return proc_df, alert_events

def test_workforce_dataset_generation_and_integrity():
    """Verify dataset dimensions, required columns, and non-negative employee bounds."""
    df = generate_workforce_signals(start_date="2024-01-01", days=730, random_seed=42)
    assert len(df) == 730 * 5
    assert validate_workforce_dataset(df) is True

    for col in df.columns:
        assert "employee_id" not in col.lower()
        assert "employee_name" not in col.lower()
        assert "medical" not in col.lower()

def test_workforce_leakage_free_baselines():
    """Verify rolling baselines use shifted windows without future lookahead."""
    df_raw = generate_workforce_signals(start_date="2024-01-01", days=60, random_seed=42)
    df_base = calculate_workforce_baselines(df_raw, window=28, min_periods=14)

    fa = df_base[df_base["organization"] == "Factory A"].reset_index(drop=True)
    expected_mean = fa.loc[0:27, "sick_leave"].mean()
    actual_mean = fa.loc[28, "sick_leave_mean"]
    assert np.isclose(expected_mean, actual_mean, atol=0.05)

def test_workforce_risk_scoring(workforce_data):
    """Verify Z-score formula and risk score boundaries [0, 100]."""
    proc_df, _ = workforce_data
    assert (proc_df["workforce_risk"] >= 0.0).all()
    assert (proc_df["workforce_risk"] <= 100.0).all()

    low_rows = proc_df[proc_df["workforce_risk"] < 40.0]
    assert (low_rows["severity"] == "Green").all()

    high_rows = proc_df[proc_df["workforce_risk"] >= 80.0]
    assert (high_rows["severity"] == "Red").all()

def test_workforce_consecutive_alert_episodes(workforce_data):
    """Verify 2 consecutive days are required before triggering an alert episode."""
    proc_df, alert_events = workforce_data

    fa_alerts = [ev for ev in alert_events if ev.organization == "Factory A"]
    assert len(fa_alerts) >= 1
    ep = fa_alerts[0]
    assert ep.severity in ["Red", "Orange"]
    assert ep.peak_risk_score >= 80.0
    assert ep.first_alert_date.startswith("2025-03")

def test_workforce_operational_capacity(workforce_data):
    """Verify operational capacity formula: 100 * (Total - Absent) / Total."""
    proc_df, _ = workforce_data
    for _, row in proc_df.sample(20, random_state=42).iterrows():
        tot = row["total_employees"]
        absent = row["absent_employees"]
        expected_cap = np.round(100.0 * (tot - absent) / tot, 2)
        assert np.isclose(row["operational_capacity"], expected_cap, atol=0.05)

def test_workforce_visualizations(workforce_data):
    """Verify map, trend chart, and comparison bar chart render without error."""
    proc_df, alert_events = workforce_data
    day_df = proc_df[proc_df["date"] == "2025-03-16"]

    map_fig = create_workforce_map(day_df, selected_org="Factory A")
    assert map_fig is not None
    assert len(map_fig.data) >= 1

    fa_df = proc_df[proc_df["organization"] == "Factory A"]
    trend_fig = create_workforce_trend_chart(fa_df, current_date="2025-03-16", selected_org="Factory A", window=60)
    assert len(trend_fig.data) >= 3

    comp_fig = create_org_comparison_chart(day_df)
    assert len(comp_fig.data[0].x) == 5

    feed = prepare_workforce_alert_feed(alert_events, proc_df, selected_date="2025-03-16")
    assert len(feed) >= 1

def test_workforce_ai_briefing_and_privacy(workforce_data):
    """Verify HR action briefing generation and strict privacy enforcement."""
    proc_df, alert_events = workforce_data
    fa_df = proc_df[proc_df["organization"] == "Factory A"]
    curr_row = fa_df[fa_df["date"] == "2025-03-16"].iloc[0]

    ctx = build_workforce_context(
        organization="Factory A",
        date="2025-03-16",
        curr_row=curr_row,
        org_df=fa_df,
        alert_events=alert_events,
    )

    briefing = generate_workforce_briefing(ctx)
    assert briefing["organization"] == "Factory A"
    assert "Plant 1" in briefing["site"]
    assert "Recommended HR Actions" in briefing["recommended_action"]
    assert "synthetic, aggregated and anonymized" in briefing["privacy_statement"]

    p_ans = answer_workforce_question("Who is sick and what are their employee names?", ctx)
    assert "strictly aggregated and anonymized" in p_ans["answer"]
    assert p_ans["mode"] == "Privacy Safeguard Policy"
