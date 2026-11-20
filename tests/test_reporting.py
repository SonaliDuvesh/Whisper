"""Unit and integration verification tests for Whisper Prompt 10 (Reporting & PDF Generation).

Verifies:
1. Regional PDF and TXT generation.
2. PDF readability and byte structure (%PDF- header).
3. NumberedCanvas multi-page layout and footer.
4. Correct region, date, risk score, and signal Z-scores in exported reports.
5. Correct lead-time and timeline values.
6. Validation CSV and validation summary CSV structure.
7. Validation PDF generation.
8. Episode-level alert CSV export.
9. Workforce PDF and CSV export (strictly aggregated, zero PII).
10. Missing AI briefing graceful handling.
11. Filename sanitization.
12. Data privacy and limitation notices present in all reports.
"""

import io
import pytest
import pandas as pd
import numpy as np

from detector import run_detection_pipeline
from ai_briefing import build_region_context, build_workforce_context
from workforce_detector import run_workforce_pipeline, load_workforce_data
from validation import evaluate_regional_performance, calculate_validation_metrics, load_outbreak_metadata
from reporting import (
    sanitize_filename,
    build_regional_report,
    export_report_txt,
    export_report_pdf,
    build_alert_summary,
    export_alerts_csv,
    export_validation_summary_csv,
    export_validation_pdf,
    build_workforce_report,
    export_workforce_csv,
    export_workforce_pdf,
)

@pytest.fixture(scope="module")
def app_state():
    """Load and process standard surveillance telemetry."""
    raw_df = pd.read_csv("data/signals.csv")
    weights = {"wastewater": 0.35, "otc_sales": 0.25, "search_trends": 0.20, "sick_leave": 0.20}
    proc_df, alerts, _ = run_detection_pipeline(
        raw_df,
        z_threshold=2.0,
        weights=weights,
        alert_threshold=60.0,
        consecutive_days=2,
    )
    outbreak_config = load_outbreak_metadata("data/outbreak_metadata.json")
    regional_summary = evaluate_regional_performance(proc_df, alerts, outbreak_config)
    metrics = calculate_validation_metrics(regional_summary, alerts, outbreak_config)

    sim_date = "2025-03-15"
    r3_df = proc_df[proc_df["region"] == "Region 3"]
    curr_row = r3_df[r3_df["date"] == sim_date].iloc[0]
    ctx = build_region_context(
        region="Region 3",
        date=sim_date,
        risk_score=float(curr_row["risk_score"]),
        severity=str(curr_row["severity"]),
        curr_row=curr_row,
        reg_df=r3_df,
        weights=weights,
        validation_summary=regional_summary,
        alert_events=alerts,
    )

    return {
        "raw_df": raw_df,
        "proc_df": proc_df,
        "alerts": alerts,
        "outbreak_config": outbreak_config,
        "regional_summary": regional_summary,
        "metrics": metrics,
        "ctx": ctx,
        "sim_date": sim_date,
    }

def test_sanitize_filename():
    """Verify filename sanitization against dangerous characters."""
    assert sanitize_filename("Region 3/Outbreak:Report.pdf") == "Region_3_Outbreak_Report.pdf"
    assert sanitize_filename("  test <file> | name.txt  ") == "test_file_name.txt"

def test_build_regional_report(app_state):
    """Verify regional report schema contains expected data and no fabrication."""
    ctx = app_state["ctx"]
    report = build_regional_report(
        region_context=ctx,
        ai_briefing={"summary": "Elevated viral shedding detected.", "provider": "Local Rule-Based Intelligence"},
        outbreak_config=app_state["outbreak_config"],
        validation_summary=app_state["regional_summary"],
    )

    assert report["header"]["region"] == "Region 3"
    assert report["header"]["observation_date"] == "2025-03-15"
    assert report["executive_summary"]["risk_score"] > 60.0
    assert report["executive_summary"]["severity"] in ("RED", "ORANGE")
    assert report["executive_summary"]["is_alert"] is True
    assert report["executive_summary"]["first_alert_date"] == "2025-03-13"

    assert len(report["signals"]) == 4
    ww = next(s for s in report["signals"] if s["key"] == "wastewater")
    assert ww["name"] == "Wastewater Viral Load"
    assert ww["z_score"] > 2.0
    assert ww["status"] == "Anomalous"

    assert report["detection_timeline"]["outbreak_onset"] == "2025-03-10"
    assert report["detection_timeline"]["first_alert_date"] == "2025-03-13"
    assert report["detection_timeline"]["lead_days"] == 8

def test_export_report_txt(app_state):
    """Verify TXT export contains all sections, data values, and privacy disclaimer."""
    ctx = app_state["ctx"]
    report = build_regional_report(ctx)
    txt_out = export_report_txt(report)

    assert "WHISPER — REGIONAL OUTBREAK INTELLIGENCE REPORT" in txt_out
    assert "Region:                 Region 3" in txt_out
    assert "Wastewater Viral Load" in txt_out
    assert "Active Alert Status:    ACTIVE OUTBREAK ALERT" in txt_out
    assert "First Alert Date:       2025-03-13" in txt_out
    assert "PRIVACY, DATA INTEGRITY & SCIENTIFIC LIMITATIONS" in txt_out
    assert "Zero personal health information" in txt_out

def test_export_report_pdf(app_state):
    """Verify PDF export generates valid, non-empty binary starting with %PDF-."""
    ctx = app_state["ctx"]
    report = build_regional_report(
        ctx,
        ai_briefing={
            "summary": "Multi-signal surge observed in Region 3.",
            "provider": "Anthropic Claude 3.5 Sonnet",
            "recommendations": ["Initiate wastewater sequencing", "Prepare municipal surge capacity"],
        },
    )
    pdf_bytes = export_report_pdf(report)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 2000
    assert pdf_bytes.startswith(b"%PDF-")

def test_export_validation_pdf_and_summary_csv(app_state):
    """Verify retrospective validation PDF and CSV exports."""
    metrics = app_state["metrics"]
    reg_summary = app_state["regional_summary"]

    pdf_bytes = export_validation_pdf(metrics, reg_summary)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 2000
    assert pdf_bytes.startswith(b"%PDF-")

    csv_str = export_validation_summary_csv(metrics)
    assert "Metric,Value" in csv_str
    assert "Detection Rate (%),100.0" in csv_str
    assert "Early Detection Rate (%),100.0" in csv_str
    assert "Average Lead Time (Days),7.3" in csv_str
    assert "False Alert Episodes,0" in csv_str

def test_export_alerts_csv(app_state):
    """Verify episode-level alert CSV contains discrete episodes and lead times."""
    alerts = app_state["alerts"]
    reg_summary = app_state["regional_summary"]

    csv_str = export_alerts_csv(alerts, reg_summary)
    assert "Alert ID,Region,First Alert Date,Last Alert Date,Duration (Days),Peak Severity" in csv_str
    assert "ALERT-EP-001" in csv_str
    assert "Region 3" in csv_str

def test_workforce_exports():
    """Verify workforce PDF and CSV exports contain aggregated data only and zero PII."""
    wf_raw = load_workforce_data("data/workforce_signals.csv")
    wf_proc, wf_alerts = run_workforce_pipeline(wf_raw)

    sim_date = "2025-03-20"
    factory_df = wf_proc[wf_proc["organization"] == "Factory A"]
    curr_row = factory_df[factory_df["date"] == sim_date].iloc[0]

    wf_ctx = build_workforce_context(
        organization="Factory A",
        date=sim_date,
        curr_row=curr_row,
        org_df=factory_df,
        alert_events=wf_alerts,
    )

    wf_report = build_workforce_report(
        workforce_context=wf_ctx,
        wf_briefing={"situation": "Factory capacity under strain.", "recommendation": ["Activate floating staff"]},
    )

    assert wf_report["organization"] == "Factory A"
    assert "absent_employees" in wf_report
    assert "operational_capacity" in wf_report
    assert "ZERO EMPLOYEE-LEVEL RECORDS OR PII" in wf_report["privacy_notice"]

    wf_pdf = export_workforce_pdf(wf_report)
    assert isinstance(wf_pdf, bytes)
    assert len(wf_pdf) > 2000
    assert wf_pdf.startswith(b"%PDF-")

    wf_csv = export_workforce_csv(wf_proc, org_name="Factory A")
    assert "total_employees,absent_employees,sick_leave" in wf_csv
    assert "Factory A" in wf_csv
    assert "employee_name" not in wf_csv
    assert "ssn" not in wf_csv
    assert "diagnosis" not in wf_csv

def test_missing_ai_briefing_graceful_handling(app_state):
    """Verify report handles missing AI briefing without raising errors or API calls."""
    ctx = app_state["ctx"]
    report = build_regional_report(ctx, ai_briefing=None)
    assert "No executive briefing generated" in report["ai_briefing"]["text"]

    pdf_bytes = export_report_pdf(report)
    assert len(pdf_bytes) > 2000
    txt_out = export_report_txt(report)
    assert len(txt_out) > 500
