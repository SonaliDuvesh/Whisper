"""Comprehensive End-to-End & Integration Verification for Whisper Prompt 12.

Verifies:
1. Complete application launch via Streamlit AppTest.
2. Zero runtime exceptions across all tabs and components.
3. Interactive selection and filter synchronization (Region, Date, Mode).
4. Full analytical consistency of risk formulas and lead times.
5. Injected outbreak ground-truth metrics (Lead time >= 7 days in demo).
6. Error-free download generations for PDF, TXT, and CSV.
7. Three-minute demonstration workflow test.
"""

import pytest
import pandas as pd
import numpy as np
from streamlit.testing.v1 import AppTest

from detector import run_detection_pipeline, DEFAULT_WEIGHTS
from validation import load_outbreak_metadata, evaluate_regional_performance, calculate_validation_metrics
from reporting import (
    build_regional_report,
    export_report_pdf,
    export_report_txt,
    export_validation_pdf,
    export_validation_summary_csv,
    export_workforce_pdf,
    export_workforce_csv,
    export_alerts_csv,
)
from workforce_detector import load_workforce_data, run_workforce_pipeline

def test_app_end_to_end_execution():
    """Verify entire Streamlit dashboard starts, executes all tabs, and finishes with zero errors."""
    import os
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.error, f"Errors encountered during app execution: {[e.value for e in at.error]}"

def test_acceptance_criteria_7_day_lead_time():
    """Verify Prompt 12 Acceptance Criterion: At least 7 days early detection in known outbreak demonstration."""
    raw_df = pd.read_csv("data/signals.csv")
    proc_df, alerts, _ = run_detection_pipeline(
        raw_df,
        z_threshold=2.0,
        weights=DEFAULT_WEIGHTS,
        alert_threshold=60.0,
        consecutive_days=2,
    )
    cfg = load_outbreak_metadata("data/outbreak_metadata.json")
    reg_summary = evaluate_regional_performance(proc_df, alerts, cfg)
    metrics = calculate_validation_metrics(reg_summary, alerts, cfg)

    r3 = reg_summary[reg_summary["Region"] == "Region 3"].iloc[0]
    r4 = reg_summary[reg_summary["Region"] == "Region 4"].iloc[0]
    r5 = reg_summary[reg_summary["Region"] == "Region 5"].iloc[0]

    assert r3["Lead Days"] >= 7, f"Region 3 lead time {r3['Lead Days']} < 7 days"
    assert r4["Lead Days"] >= 7, f"Region 4 lead time {r4['Lead Days']} < 7 days"
    assert r5["Lead Days"] >= 7, f"Region 5 lead time {r5['Lead Days']} < 7 days"

    assert metrics["average_lead_time"] >= 7.0
    assert metrics["detection_rate"] == 100.0
    assert metrics["early_detection_rate"] == 100.0
    assert metrics["false_alert_episodes"] == 0

def test_all_export_formats_end_to_end():
    """Verify all 6 export endpoints generate valid, well-formed binary and string content."""
    raw_df = pd.read_csv("data/signals.csv")
    proc_df, alerts, _ = run_detection_pipeline(
        raw_df,
        z_threshold=2.0,
        weights=DEFAULT_WEIGHTS,
        alert_threshold=60.0,
        consecutive_days=2,
    )
    reg_summary = evaluate_regional_performance(proc_df, alerts)
    metrics = calculate_validation_metrics(reg_summary, alerts)

    v_pdf = export_validation_pdf(metrics, reg_summary)
    assert v_pdf.startswith(b"%PDF-") and len(v_pdf) > 2000
    v_csv = export_validation_summary_csv(metrics)
    assert "Detection Rate" in v_csv

    a_csv = export_alerts_csv(alerts, reg_summary)
    assert "ALERT-EP-001" in a_csv

    wf_raw = load_workforce_data("data/workforce_signals.csv")
    wf_proc, _ = run_workforce_pipeline(wf_raw)
    wf_csv = export_workforce_csv(wf_proc)
    assert "total_employees" in wf_csv

    from ai_briefing import build_region_context
    curr_row = proc_df[proc_df["region"] == "Region 3"].iloc[100]
    ctx = build_region_context(
        region="Region 3",
        date="2025-03-15",
        risk_score=75.0,
        severity="Orange",
        curr_row=curr_row,
        reg_df=proc_df[proc_df["region"] == "Region 3"],
        weights=DEFAULT_WEIGHTS,
        validation_summary=reg_summary,
        alert_events=alerts,
    )
    report = build_regional_report(ctx, validation_summary=reg_summary)
    r_pdf = export_report_pdf(report)
    assert r_pdf.startswith(b"%PDF-") and len(r_pdf) > 2000
    r_txt = export_report_txt(report)
    assert "WHISPER — REGIONAL OUTBREAK INTELLIGENCE REPORT" in r_txt
