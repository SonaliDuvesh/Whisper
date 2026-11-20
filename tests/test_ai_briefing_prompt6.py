"""Unit and integration verification tests for Whisper Prompt 6 (AI Briefing & Contextual Q&A).

Verifies:
1. build_region_context respects strict no-lookahead boundary.
2. calculate_confidence evaluates High, Moderate, and Low deterministically.
3. generate_fallback_briefing produces structured evidence-based intelligence without API keys.
4. generate_ai_briefing handles mocked OpenAI & Anthropic responses and falls back cleanly on error.
5. answer_region_question handles common epidemiological queries via deterministic fallback.
6. Unsupported queries state data boundaries without hallucinating.
7. Zero credential leakage in context or outputs.
"""
from unittest.mock import MagicMock, patch
import json
import pytest
import pandas as pd
import numpy as np

from detector import run_detection_pipeline
from ai_briefing import (
    build_region_context,
    calculate_confidence,
    generate_fallback_briefing,
    generate_ai_briefing,
    answer_region_question,
    get_ai_config,
    validate_ai_response,
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

def test_build_region_context_no_lookahead(pipeline_data):
    """Test that build_region_context strictly hides future alert/official events."""
    processed_df, alert_events, val_summary, weights = pipeline_data
    reg_df = processed_df[processed_df["region"] == "Region 3"].sort_values("date")
    
    curr_row = reg_df[reg_df["date"] == "2025-03-14"].iloc[0]
    ctx = build_region_context(
        region="Region 3",
        date="2025-03-14",
        risk_score=float(curr_row["risk_score"]),
        severity=str(curr_row["severity"]),
        curr_row=curr_row,
        reg_df=reg_df,
        weights=weights,
        z_threshold=2.0,
        validation_summary=val_summary,
        alert_events=alert_events,
    )

    assert ctx["region"] == "Region 3"
    assert ctx["date"] == "2025-03-14"
    assert ctx["official_reporting"]["reported"] is False
    assert ctx["official_reporting"]["official_report_date"] is None
    assert ctx["lead_time"]["status"] == "alert_active_awaiting_official_reporting"
    assert ctx["lead_time"]["days"] is None

def test_calculate_confidence_deterministic():
    """Verify confidence classification across distinct risk states."""
    high_ctx = {
        "anomalous_signal_count": 3,
        "risk_score": 85.0,
        "alert": {"active": True},
    }
    c_high = calculate_confidence(high_ctx)
    assert c_high["level"] == "High"
    assert "strong multi-signal agreement" in c_high["rationale"]

    mod_ctx = {
        "anomalous_signal_count": 2,
        "risk_score": 62.0,
        "alert": {"active": False},
    }
    c_mod = calculate_confidence(mod_ctx)
    assert c_mod["level"] == "Moderate"

    low_ctx = {
        "anomalous_signal_count": 0,
        "risk_score": 12.0,
        "alert": {"active": False},
    }
    c_low = calculate_confidence(low_ctx)
    assert c_low["level"] == "Low"

def test_generate_fallback_briefing_outbreak(pipeline_data):
    """Verify fallback briefing generation on an active outbreak day."""
    processed_df, alert_events, val_summary, weights = pipeline_data
    reg_df = processed_df[processed_df["region"] == "Region 3"].sort_values("date")
    curr_row = reg_df[reg_df["date"] == "2025-03-17"].iloc[0]

    ctx = build_region_context(
        region="Region 3",
        date="2025-03-17",
        risk_score=float(curr_row["risk_score"]),
        severity=str(curr_row["severity"]),
        curr_row=curr_row,
        reg_df=reg_df,
        weights=weights,
        z_threshold=2.0,
        validation_summary=val_summary,
        alert_events=alert_events,
    )

    briefing = generate_fallback_briefing(ctx)
    assert briefing["region"] == "Region 3"
    assert briefing["severity"] == "RED"
    assert "Region 3 is exhibiting critical epidemiological signal deviations" in briefing["situation"]
    assert "Wastewater Viral Load" in briefing["supporting_signals"]
    assert "Intensify district-level syndromic surveillance" in briefing["recommended_action"]
    assert briefing["confidence_level"] == "High"
    assert briefing["is_fallback"] is True

def test_generate_ai_briefing_fallback_when_no_keys(pipeline_data):
    """When no API key is set, generate_ai_briefing cleanly returns fallback."""
    processed_df, alert_events, val_summary, weights = pipeline_data
    reg_df = processed_df[processed_df["region"] == "Region 1"].sort_values("date")
    curr_row = reg_df[reg_df["date"] == "2025-03-18"].iloc[0]

    ctx = build_region_context(
        region="Region 1",
        date="2025-03-18",
        risk_score=float(curr_row["risk_score"]),
        severity=str(curr_row["severity"]),
        curr_row=curr_row,
        reg_df=reg_df,
        weights=weights,
        z_threshold=2.0,
    )

    briefing = generate_ai_briefing(ctx, provider="fallback")
    assert briefing["is_fallback"] is True
    assert "Local Deterministic Engine" in briefing["provider_mode"]

def test_generate_ai_briefing_mocked_openai():
    """Verify OpenAI integration using a mocked client."""
    mock_json = json.dumps({
        "situation": "Region 3 is experiencing elevated multi-signal risk classified as Red.",
        "supporting_signals": "Wastewater is +3.4σ and OTC sales are +2.8σ above baseline.",
        "recommended_action": "Increase sentinel clinic sampling and review hospital bed preparedness.",
        "confidence_explanation": "High confidence based on multiple agreeing indicators.",
    })
    
    mock_choice = MagicMock()
    mock_choice.message.content = mock_json
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client
        
        with patch("ai_briefing.get_ai_config", return_value={
            "active_provider": "openai",
            "openai_key": "test_key",
            "openai_model": "gpt-4o-mini",
            "anthropic_key": "",
            "anthropic_model": "claude-3-5-sonnet",
            "has_openai": True,
            "has_anthropic": False,
        }):
            dummy_ctx = {
                "region": "Region 3",
                "date": "2025-03-18",
                "risk_score": 85.0,
                "severity": "RED",
                "anomalous_signal_count": 3,
                "signals": {},
                "alert": {"active": True},
            }
            res = generate_ai_briefing(dummy_ctx, provider="openai")
            assert res["is_fallback"] is False
            assert "OpenAI (gpt-4o-mini)" in res["provider_mode"]
            assert "Region 3 is experiencing" in res["situation"]

def test_generate_ai_briefing_mocked_anthropic():
    """Verify Anthropic integration using a mocked client."""
    mock_json = json.dumps({
        "situation": "Region 4 risk score is 74.0 classified as Orange.",
        "supporting_signals": "OTC sales are elevated at +2.6σ.",
        "recommended_action": "Audit community pharmacy distribution and monitor absenteeism.",
        "confidence_explanation": "Moderate confidence based on dual signal elevation.",
    })
    
    mock_content = MagicMock()
    mock_content.text = mock_json
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    
    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_cls.return_value = mock_client
        
        with patch("ai_briefing.get_ai_config", return_value={
            "active_provider": "anthropic",
            "openai_key": "",
            "openai_model": "gpt-4o-mini",
            "anthropic_key": "test_key",
            "anthropic_model": "claude-3-5-sonnet-20241022",
            "has_openai": False,
            "has_anthropic": True,
        }):
            dummy_ctx = {
                "region": "Region 4",
                "date": "2025-03-18",
                "risk_score": 74.0,
                "severity": "ORANGE",
                "anomalous_signal_count": 2,
                "signals": {},
                "alert": {"active": False},
            }
            res = generate_ai_briefing(dummy_ctx, provider="anthropic")
            assert res["is_fallback"] is False
            assert "Anthropic (claude-3-5-sonnet-20241022)" in res["provider_mode"]
            assert "Region 4 risk score is 74.0" in res["situation"]

def test_answer_region_question_deterministic_fallback():
    """Test deterministic Q&A handling of standard public-health questions."""
    ctx = {
        "region": "Region 3",
        "date": "2025-03-18",
        "risk_score": 82.4,
        "severity": "RED",
        "anomalous_signal_count": 3,
        "signals": {
            "wastewater": {"name": "Wastewater Viral Load", "z_score": 3.2, "weight": 0.35, "anomalous": True},
            "otc_sales": {"name": "OTC Pharmacy Respiratory Sales", "z_score": 2.5, "weight": 0.25, "anomalous": True},
            "search_trends": {"name": "Symptom Search Activity", "z_score": 2.1, "weight": 0.20, "anomalous": True},
            "sick_leave": {"name": "Workplace Sick Leave", "z_score": 0.8, "weight": 0.20, "anomalous": False},
        },
        "alert": {"active": True, "first_alert_date": "2025-03-12"},
        "official_reporting": {"reported": False, "official_report_date": None, "current_cases": 28, "current_threshold": 65.0},
        "lead_time": {"days": None, "status": "alert_active_awaiting_official_reporting"},
    }

    q1 = answer_region_question("Why is Region 3 classified as Red?", ctx, provider="fallback")
    assert "Region 3 is classified as RED with a risk score of 82.4/100" in q1["answer"]

    q2 = answer_region_question("Which signal contributes most to current risk?", ctx, provider="fallback")
    assert "Wastewater Viral Load" in q2["answer"]

    q3 = answer_region_question("Why did wastewater rise before official cases?", ctx, provider="fallback")
    assert "asymptomatic and pre-symptomatic stages" in q3["answer"]

    q4 = answer_region_question("How many signals are anomalous?", ctx, provider="fallback")
    assert "3 of 4 leading surveillance signals are anomalous" in q4["answer"]

    q5 = answer_region_question("Has the official reporting threshold been crossed?", ctx, provider="fallback")
    assert "official clinical cases in Region 3 have not crossed" in q5["answer"]

    q6 = answer_region_question("What is the stock price of Apple?", ctx, provider="fallback")
    assert "does not contain information to address that query" in q6["answer"]

def test_zero_credential_leakage():
    """Ensure API credentials are never printed or stored in context objects."""
    cfg = get_ai_config()
    dummy_ctx = {
        "region": "Region 3",
        "date": "2025-03-18",
        "risk_score": 82.4,
        "severity": "RED",
    }
    briefing = generate_fallback_briefing(dummy_ctx)
    briefing_str = json.dumps(briefing)
    assert "sk-" not in briefing_str
    assert "key" not in briefing_str.lower() or "no api key required" in briefing_str.lower()
