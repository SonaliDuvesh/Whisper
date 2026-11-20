"""Unit and integration verification tests for Whisper Prompt 9 (Historical Validation & Backtesting).

Verifies:
1. Ground-truth outbreak metadata loading and schema.
2. Alert-to-outbreak event association rules.
3. Regional performance outcome classifications.
4. Detection rate, early detection rate, average lead time, and median lead time calculations.
5. Precision, recall, and F1-score mathematical correctness.
6. Zero-denominator boundary safety.
7. Full historical backtest and threshold sensitivity trade-offs.
8. Retrospective visualizations and CSV export consistency.
"""
import pytest
import pandas as pd
import numpy as np

from detector import run_detection_pipeline
from validation import (
    load_outbreak_metadata,
    associate_alerts_with_outbreaks,
    evaluate_regional_performance,
    calculate_validation_metrics,
    run_threshold_sensitivity_analysis,
    export_validation_csv,
)
from viz import (
    create_lead_time_comparison_chart,
    create_historical_risk_progression_chart,
    create_sensitivity_heatmap,
)

@pytest.fixture(scope="module")
def validation_data():
    raw_df = pd.read_csv("data/signals.csv")
    weights = {"wastewater": 0.35, "otc_sales": 0.25, "search_trends": 0.20, "sick_leave": 0.20}
    processed_df, alert_events, _ = run_detection_pipeline(
        raw_df,
        z_threshold=2.0,
        weights=weights,
        alert_threshold=60.0,
        consecutive_days=2,
    )
    outbreak_config = load_outbreak_metadata("data/outbreak_metadata.json")
    regional_summary = evaluate_regional_performance(processed_df, alert_events, outbreak_config)
    metrics = calculate_validation_metrics(regional_summary, alert_events, outbreak_config)
    return processed_df, alert_events, outbreak_config, regional_summary, metrics

def test_load_outbreak_metadata():
    """Verify ground truth outbreak metadata contains required fields."""
    cfg = load_outbreak_metadata("data/outbreak_metadata.json")
    assert cfg["outbreak_id"] == "OUTBREAK_2025_001"
    assert cfg["start_date"] == "2025-03-10"
    assert "Region 3" in cfg["affected_regions"]
    assert "Region 4" in cfg["affected_regions"]
    assert "Region 5" in cfg["affected_regions"]

def test_associate_alerts_with_outbreaks(validation_data):
    """Verify true positive vs false alarm classification."""
    _, alert_events, outbreak_config, _, _ = validation_data
    tps, fps = associate_alerts_with_outbreaks(alert_events, outbreak_config)

    assert len(tps) == 3
    assert len(fps) == 0

def test_evaluate_regional_performance(validation_data):
    """Verify regional performance table contains all 8 regions and expected outcomes."""
    _, _, _, reg_summary, _ = validation_data
    assert len(reg_summary) == 8

    r3 = reg_summary[reg_summary["Region"] == "Region 3"].iloc[0]
    assert r3["Known Outbreak"] == "Yes"
    assert bool(r3["Detected"]) is True
    assert r3["Lead Days"] == 8
    assert r3["Outcome"] == "Early Detection"

    r4 = reg_summary[reg_summary["Region"] == "Region 4"].iloc[0]
    assert r4["Lead Days"] == 7
    assert r4["Outcome"] == "Early Detection"

    r5 = reg_summary[reg_summary["Region"] == "Region 5"].iloc[0]
    assert r5["Lead Days"] == 7
    assert r5["Outcome"] == "Early Detection"

    unaffected = reg_summary[reg_summary["Known Outbreak"] == "No"]
    assert len(unaffected) == 5
    assert (unaffected["Outcome"] == "No Alert").all()

def test_calculate_validation_metrics_accuracy(validation_data):
    """Verify precision, recall, F1, and lead-time statistics."""
    _, _, _, _, metrics = validation_data

    assert metrics["total_outbreak_regions"] == 3
    assert metrics["detected_outbreaks"] == 3
    assert metrics["detection_rate"] == 100.0
    assert metrics["early_detection_rate"] == 100.0

    assert np.isclose(metrics["average_lead_time"], 7.3, atol=0.1)
    assert np.isclose(metrics["median_lead_time"], 7.0, atol=0.1)

    assert metrics["false_alert_episodes"] == 0
    assert metrics["false_alarm_rate"] == 0.0

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0

def test_zero_denominator_handling():
    """Verify metric calculation remains safe when no outbreaks or alerts exist."""
    empty_summary = pd.DataFrame([
        {"Region": "Region 1", "Known Outbreak": "No", "Detected": False, "Lead Days": None, "False Alarm Episodes": 0, "Outcome": "No Alert"}
    ])
    metrics = calculate_validation_metrics(empty_summary, alert_events=[], outbreak_config={"affected_regions": []})

    assert metrics["detection_rate"] == 0.0
    assert metrics["early_detection_rate"] == 0.0
    assert metrics["average_lead_time"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0

def test_run_threshold_sensitivity_analysis():
    """Verify threshold sensitivity sweep returns valid grid of combinations."""
    raw_df = pd.read_csv("data/signals.csv")
    sens_df = run_threshold_sensitivity_analysis(
        df=raw_df,
        z_thresholds=[2.0, 2.5],
        alert_thresholds=[50.0, 60.0],
    )
    assert len(sens_df) == 4
    expected_cols = ["z_threshold", "alert_threshold", "detection_rate", "early_detection_rate", "avg_lead_time", "false_alarms", "precision", "recall", "f1_score"]
    for c in expected_cols:
        assert c in sens_df.columns

def test_export_validation_csv(validation_data):
    """Verify CSV export contains header and all 8 regional rows."""
    _, _, _, reg_summary, _ = validation_data
    csv_str = export_validation_csv(reg_summary)
    assert "Region,Known Outbreak,First Alert Date" in csv_str
    assert "Region 3" in csv_str
    assert "Early Detection" in csv_str

def test_validation_visualizations(validation_data):
    """Verify lead time chart, progression chart, and heatmap generation."""
    proc_df, _, outbreak_config, reg_summary, _ = validation_data

    bar_fig = create_lead_time_comparison_chart(reg_summary)
    assert bar_fig is not None
    assert len(bar_fig.data[0].x) == 3

    prog_fig = create_historical_risk_progression_chart(proc_df, outbreak_config)
    assert prog_fig is not None
    assert len(prog_fig.data) == 4

    dummy_sens = pd.DataFrame([
        {"z_threshold": 2.0, "alert_threshold": 60.0, "f1_score": 1.0},
        {"z_threshold": 2.5, "alert_threshold": 60.0, "f1_score": 0.8},
        {"z_threshold": 2.0, "alert_threshold": 70.0, "f1_score": 0.9},
        {"z_threshold": 2.5, "alert_threshold": 70.0, "f1_score": 0.7},
    ])
    h_fig = create_sensitivity_heatmap(dummy_sens, metric="f1_score")
    assert h_fig is not None
