"""Unit and integration verification tests for Whisper Prompt 5.

Verifies:
1. create_signal_timeline (no-lookahead, traces, thresholds, alert & official markers, secondary axis)
2. create_risk_gauge (range, values, color mapping)
3. create_signal_contribution_chart (math: sum of contributions == risk_score, sorting, weights)
4. prepare_signal_status_table (baseline, values, z-scores, status flags)
5. calculate_signal_trend (7-day trajectory, earliest onset, top risk driver)
6. All 8 regions across active outbreak dates
"""
import pytest
import numpy as np
import pandas as pd
from detector import (
    calculate_rolling_baseline,
    calculate_z_scores,
    detect_anomalies,
    calculate_risk_score,
    classify_severity,
    generate_alerts,
    calculate_official_thresholds,
    calculate_lead_times,
    normalize_weights,
    LEADING_SIGNALS,
)
from viz import (
    create_signal_timeline,
    create_risk_gauge,
    create_signal_contribution_chart,
    prepare_signal_status_table,
    calculate_signal_trend,
    SEV_COLOR_MAP,
)

@pytest.fixture(scope="module")
def pipeline_data():
    raw_df = pd.read_csv("data/signals.csv")
    weights = {"wastewater": 0.35, "otc_sales": 0.25, "search_trends": 0.20, "sick_leave": 0.20}
    from detector import run_detection_pipeline
    processed_df, alert_events, val_summary = run_detection_pipeline(
        raw_df,
        z_threshold=2.0,
        weights=weights,
        alert_threshold=60.0,
        consecutive_days=2,
    )
    return processed_df, alert_events, [], val_summary, weights

def test_no_lookahead_in_signal_timeline(pipeline_data):
    """Test that create_signal_timeline strictly respects the simulation date cutoff."""
    off_df, alert_events, off_events, val_summary, weights = pipeline_data
    reg_df = off_df[off_df["region"] == "Region 3"].sort_values("date")
    
    selected_date = "2025-03-15"
    fig = create_signal_timeline(
        region_df=reg_df,
        selected_date=selected_date,
        alert_date="2025-03-12",
        official_date="2025-03-20",
        z_threshold=2.0,
        show_official_cases=True,
        history_days=60,
    )
    
    for trace in fig.data:
        if hasattr(trace, "x") and trace.x is not None:
            for x_val in trace.x:
                assert str(x_val) <= selected_date, f"Lookahead leak: found date {x_val} > {selected_date}"

def test_window_slicing_in_signal_timeline(pipeline_data):
    """Test 30-day, 60-day, 90-day, and Full window slicing."""
    off_df, _, _, _, _ = pipeline_data
    reg_df = off_df[off_df["region"] == "Region 3"].sort_values("date")
    selected_date = "2025-03-25"
    
    fig_30 = create_signal_timeline(reg_df, selected_date, history_days=30)
    fig_60 = create_signal_timeline(reg_df, selected_date, history_days=60)
    fig_full = create_signal_timeline(reg_df, selected_date, history_days="Full")
    
    assert len(fig_30.data[0].x) == 30
    assert len(fig_60.data[0].x) == 60
    assert len(fig_full.data[0].x) > 400

def test_signal_contribution_math_invariance(pipeline_data):
    """Test that sum of signal contributions strictly equals the risk score."""
    off_df, _, _, _, weights = pipeline_data
    
    test_dates = ["2025-01-15", "2025-03-12", "2025-03-18", "2025-03-25"]
    test_regions = ["Region 1", "Region 3", "Region 4", "Region 5"]
    
    for reg in test_regions:
        reg_df = off_df[off_df["region"] == reg]
        for dt in test_dates:
            row = reg_df[reg_df["date"] == dt].iloc[0]
            risk_score = float(row["risk_score"])
            
            w = normalize_weights(weights)
            tot_contrib = 0.0
            for sig in LEADING_SIGNALS:
                z_val = float(row.get(f"z_{sig}", 0.0))
                z_clipped = float(np.clip(z_val, 0.0, 5.0))
                contrib = 100.0 * (w[sig] * z_clipped) / 5.0
                tot_contrib += contrib
            
            assert np.isclose(tot_contrib, risk_score, atol=0.02), (
                f"Contribution sum {tot_contrib} does not equal risk score {risk_score} for {reg} on {dt}"
            )

def test_risk_gauge_properties(pipeline_data):
    """Test create_risk_gauge generates correct value and indicator ranges."""
    off_df, _, _, _, _ = pipeline_data
    
    fig = create_risk_gauge(risk_score=78.5, severity="ORANGE")
    indicator = fig.data[0]
    assert indicator.value == 78.5
    assert indicator.gauge.axis.range == (0, 100)
    assert indicator.gauge.bar.color == SEV_COLOR_MAP["ORANGE"]

def test_prepare_signal_status_table(pipeline_data):
    """Test signal status table contains correct columns and flags."""
    off_df, _, _, _, weights = pipeline_data
    reg_df = off_df[off_df["region"] == "Region 3"]
    row = reg_df[reg_df["date"] == "2025-03-18"].iloc[0]
    
    table = prepare_signal_status_table(row, weights=weights, z_threshold=2.0)
    assert len(table) == 4
    expected_cols = ["Signal", "Current Value", "Historical Baseline", "Z-Score", "Weight", "Risk Contribution", "Status"]
    assert list(table.columns) == expected_cols
    
    ww_row = table[table["Signal"].str.contains("Wastewater")].iloc[0]
    assert ww_row["Status"] == "ANOMALOUS"

def test_calculate_signal_trend(pipeline_data):
    """Test calculate_signal_trend detects trajectory and onset correctly."""
    off_df, _, _, _, _ = pipeline_data
    reg_df = off_df[off_df["region"] == "Region 3"]
    
    insights = calculate_signal_trend(reg_df, selected_date="2025-03-18", z_threshold=2.0)
    assert insights["trend_direction"] == "Increasing"
    assert insights["trend_delta"] > 0
    assert "Wastewater" in insights["first_anomalous_signal"]
    assert insights["anom_count"] >= 3

def test_all_regions_generate_valid_visualizations(pipeline_data):
    """Test that all 8 regions render charts without error on key dates."""
    off_df, alert_events, off_events, val_summary, weights = pipeline_data
    regions = sorted(off_df["region"].unique().tolist())
    assert len(regions) == 8
    
    for reg in regions:
        reg_df = off_df[off_df["region"] == reg].sort_values("date")
        row = reg_df[reg_df["date"] == "2025-03-20"].iloc[0]
        
        fig_time = create_signal_timeline(
            region_df=reg_df,
            selected_date="2025-03-20",
            alert_date="2025-03-12" if reg == "Region 3" else None,
            official_date="2025-03-20" if reg == "Region 3" else None,
            z_threshold=2.0,
            show_official_cases=True,
            history_days=60,
        )
        assert len(fig_time.data) >= 4
        
        fig_gauge = create_risk_gauge(float(row["risk_score"]), str(row["severity"]))
        assert fig_gauge is not None
        
        fig_contrib = create_signal_contribution_chart(row, weights)
        assert len(fig_contrib.data[0].x) == 4
        
        table = prepare_signal_status_table(row, weights, 2.0)
        assert len(table) == 4
        
        insights = calculate_signal_trend(reg_df, "2025-03-20", 2.0)
        assert insights["trend_direction"] in ["Increasing", "Decreasing", "Stable"]
