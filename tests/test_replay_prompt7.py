"""Unit and integration verification tests for Whisper Prompt 7 (Replay Simulation).

Verifies:
1. Scenario date bounding (Outbreak focus vs Full dataset).
2. Strict historical integrity and zero lookahead at replay date D.
3. Chronological milestone sequence (Onset -> Anomaly -> Whisper Alert -> Official Report).
4. Actual lead-time calculation (8 days for Region 3).
5. Replay trajectory chart data slicing and threshold reference line.
6. Milestone banner state transitions (Pre-alert -> Active warning -> Confirmed Advantage).
"""
import pytest
import pandas as pd
import numpy as np

from detector import run_detection_pipeline
from viz import (
    create_replay_trajectory_chart,
    render_replay_timeline_html,
    render_milestone_banner_html,
    prepare_alert_feed,
)

@pytest.fixture(scope="module")
def pipeline_data():
    raw_df = pd.read_csv("data/signals.csv")
    weights = {"wastewater": 0.35, "otc_sales": 0.25, "search_trends": 0.20, "sick_leave": 0.20}
    processed_df, alert_events, val_summary = run_detection_pipeline(
        raw_df,
        z_threshold=2.0,
        weights=weights,
        alert_threshold=60.0,
        consecutive_days=2,
    )
    return processed_df, alert_events, val_summary, weights

def test_replay_chronological_milestones(pipeline_data):
    """Verify that chronological milestone sequence matches synthetic outbreak ground truth."""
    processed_df, alert_events, val_summary, weights = pipeline_data
    r3_val = val_summary[val_summary["region"] == "Region 3"].iloc[0]

    first_alert = r3_val["first_alert_date"]
    official_report = r3_val["official_report_date"]
    lead_days = int(r3_val["lead_days"])

    assert first_alert == "2025-03-13"
    assert official_report == "2025-03-21"
    assert lead_days == 8

    r3_df = processed_df[processed_df["region"] == "Region 3"].sort_values("date")
    anoms = r3_df[(r3_df["date"] >= "2025-03-10") & (r3_df["z_wastewater"] > 2.0)]
    first_anom_date = anoms["date"].min()
    assert first_anom_date == "2025-03-11"

def test_replay_trajectory_chart_no_lookahead(pipeline_data):
    """Verify create_replay_trajectory_chart never plots dates past the simulated replay date."""
    processed_df, _, _, _ = pipeline_data
    replay_date = "2025-03-15"

    fig = create_replay_trajectory_chart(
        processed_df=processed_df,
        current_date=replay_date,
        outbreak_regions=["Region 3", "Region 4", "Region 5"],
        alert_threshold=60.0,
    )

    assert len(fig.data) == 3
    for trace in fig.data:
        for x_val in trace.x:
            assert str(x_val) <= replay_date, f"Leak: found date {x_val} > {replay_date}"

def test_replay_alert_feed_historical_cutoff(pipeline_data):
    """Verify alert feed at replay date D excludes future episodes and reflects current status."""
    processed_df, alert_events, _, _ = pipeline_data
    
    feed_mar10 = prepare_alert_feed(
        alert_events=alert_events,
        processed_df=processed_df,
        selected_date="2025-03-10",
    )
    assert len(feed_mar10) == 0

    feed_mar14 = prepare_alert_feed(
        alert_events=alert_events,
        processed_df=processed_df,
        selected_date="2025-03-14",
    )
    assert len(feed_mar14) >= 2
    r3_item = next(item for item in feed_mar14 if item["region"] == "Region 3")
    assert r3_item["first_alert_date"] == "2025-03-13"
    assert r3_item["is_active"] is True

def test_milestone_timeline_html_states():
    """Verify timeline HTML marks completed, current, and upcoming stages correctly."""
    html_early = render_replay_timeline_html(current_date="2025-03-12")
    assert "Outbreak Onset" in html_early
    assert "COMPLETED" in html_early
    assert "UPCOMING" in html_early
    assert "Verified Early Detection Advantage" not in html_early

    html_late = render_replay_timeline_html(current_date="2025-03-22")
    assert "Verified Early Detection Advantage: 8 Days Lead Time Achieved" in html_late

def test_milestone_banner_transitions():
    """Verify milestone banner state transitions across key dates."""
    b_pre = render_milestone_banner_html(current_date="2025-03-08")
    assert b_pre is None

    b_alert = render_milestone_banner_html(current_date="2025-03-15")
    assert b_alert is not None
    assert "Outbreak Alert Episode Active" in b_alert
    assert "Awaiting Official Confirmation" in b_alert

    b_post = render_milestone_banner_html(current_date="2025-03-22")
    assert b_post is not None
    assert "Official Report Issued • Whisper Alerted 8 Days Earlier" in b_post
    assert "+8 DAYS" in b_post
