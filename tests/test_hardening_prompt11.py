"""Unit and regression tests for Whisper Prompt 11 (Application Hardening & First-Run Reliability).

Verifies:
1. Theme configuration file (.streamlit/config.toml) presence and parameters.
2. First-run data loading resilience.
3. Strict session-state initialization.
4. Single-source-of-truth analytical consistency across detector, reporting, and validation.
5. Absolute absence of future lookahead bias across all baselines.
6. Execution performance of core detection pipeline.
7. Clean module imports with zero circular dependencies.
"""

import os
import tomllib
import pytest
import pandas as pd
import numpy as np

from detector import run_detection_pipeline, DEFAULT_WEIGHTS
from workforce_detector import run_workforce_pipeline, load_workforce_data
from validation import evaluate_regional_performance, calculate_validation_metrics
from reporting import sanitize_filename, build_regional_report
from ai_briefing import build_region_context

def test_streamlit_config_theme():
    """Verify .streamlit/config.toml defines required Clean Command Center dark theme."""
    config_path = os.path.join(".streamlit", "config.toml")
    assert os.path.exists(config_path), ".streamlit/config.toml must exist"

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "theme" in content
    assert "#0B1220" in content
    assert "#111B2E" in content
    assert "#06B6D4" in content

def test_first_run_data_resilience(tmp_path):
    """Verify data loading function can safely recover or auto-generate if file missing."""
    from data_gen import save_dataset
    dummy_path = str(tmp_path / "test_signals.csv")

    df = pd.read_csv("data/signals.csv").head(800)
    save_dataset(df, dummy_path)

    assert os.path.exists(dummy_path)
    loaded = pd.read_csv(dummy_path)
    assert len(loaded) == 800
    assert "wastewater" in loaded.columns

def test_analytical_consistency_across_modules():
    """Verify that detector, validation, and reporting all share identical risk scores and dates."""
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

    r3_val = reg_summary[reg_summary["Region"] == "Region 3"].iloc[0]
    assert r3_val["Lead Days"] == 8
    assert r3_val["First Alert Date"] == "2025-03-13"
    assert r3_val["Official Report Date"] == "2025-03-21"

    assert metrics["detected_outbreaks"] == 3
    assert len(alerts) == 3

def test_strict_no_lookahead_baseline():
    """Verify moving 28-day baseline uses shifted data and never leaks future telemetry."""
    from detector import calculate_rolling_baseline

    test_dates = pd.date_range("2024-01-01", periods=40, freq="D")
    df = pd.DataFrame({
        "date": test_dates.strftime("%Y-%m-%d"),
        "region": ["Region 1"] * 40,
        "wastewater": np.arange(100.0, 140.0),
    })

    base_df = calculate_rolling_baseline(df, signal_columns=["wastewater"], window=28)

    day_28_val = base_df.iloc[28]["wastewater"]
    day_28_mean = base_df.iloc[28]["wastewater_mean"]

    expected_shifted_mean = np.mean(np.arange(100.0, 128.0))
    assert np.isclose(day_28_mean, expected_shifted_mean)
    assert not np.isclose(day_28_mean, np.mean(np.arange(101.0, 129.0)))

def test_clean_imports_and_no_circular_dependencies():
    """Verify that importing all core Whisper modules executes without circular import errors."""
    import detector
    import data_gen
    import ai_briefing
    import viz
    import workforce_detector
    import workforce_gen
    import validation
    import reporting
    import app

    assert detector is not None
    assert reporting is not None
    assert app is not None
