"""Whisper Historical Validation, Backtesting & Detection Performance Engine.

Objectively evaluates multi-signal epidemiological surveillance performance across
all 8 monitored regions, benchmarked against ground-truth injected outbreaks and
official clinical reporting thresholds.
"""

from typing import Dict, List, Optional, Tuple, Any
import json
import os
import numpy as np
import pandas as pd

from detector import (
    LEADING_SIGNALS,
    detect_anomalies,
    calculate_risk_score,
    generate_alerts,
    calculate_official_thresholds,
    run_detection_pipeline,
)

def load_outbreak_metadata(path: str = "data/outbreak_metadata.json") -> Dict[str, Any]:
    """Load machine-readable ground truth outbreak metadata."""
    if not os.path.exists(path):
        alt_path = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(alt_path):
            path = alt_path
        else:
            return {
                "outbreak_id": "OUTBREAK_2025_001",
                "start_date": "2025-03-10",
                "end_date": "2025-04-15",
                "evaluation_window_start": "2025-03-05",
                "evaluation_window_end": "2025-04-20",
                "affected_regions": ["Region 3", "Region 4", "Region 5"],
                "epicenter": "Region 3",
                "synthetic": True,
            }

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def associate_alerts_with_outbreaks(
    alert_events: List[Any],
    outbreak_config: Dict[str, Any],
) -> Tuple[List[Any], List[Any]]:
    """Associate alert episodes with known injected outbreaks vs false alarms.

    An alert episode is a True Positive if and only if:
    1. It occurs in an affected outbreak region.
    2. Its first alert date falls within the designated evaluation window.

    Otherwise, it is categorized as a False Alarm episode.

    Returns:
        tuple: (true_positive_alerts, false_alarm_alerts)
    """
    affected_regions = set(outbreak_config.get("affected_regions", []))
    win_start = outbreak_config.get("evaluation_window_start", "2025-03-05")
    win_end = outbreak_config.get("evaluation_window_end", "2025-04-20")

    true_positives = []
    false_alarms = []

    for ev in alert_events:
        reg = getattr(ev, "region", None) or (ev.get("region") if isinstance(ev, dict) else "")
        start_d = getattr(ev, "first_alert_date", None) or (ev.get("first_alert_date") if isinstance(ev, dict) else "")

        if reg in affected_regions and (win_start <= start_d <= win_end):
            true_positives.append(ev)
        else:
            false_alarms.append(ev)

    return true_positives, false_alarms

def evaluate_regional_performance(
    processed_df: pd.DataFrame,
    alert_events: List[Any],
    outbreak_config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Evaluate detection outcomes, lead times, and false alarms for every monitored region."""
    if outbreak_config is None:
        outbreak_config = load_outbreak_metadata()

    affected_regions = set(outbreak_config.get("affected_regions", []))
    win_start = outbreak_config.get("evaluation_window_start", "2025-03-05")
    win_end = outbreak_config.get("evaluation_window_end", "2025-04-20")

    records = []
    all_regions = sorted(processed_df["region"].unique().tolist())

    for reg in all_regions:
        is_outbreak = (reg in affected_regions)
        reg_df = processed_df[processed_df["region"] == reg].sort_values("date")

        reg_alerts = [
            ev for ev in alert_events
            if (getattr(ev, "region", "") == reg or (isinstance(ev, dict) and ev.get("region") == reg))
        ]

        valid_outbreak_alerts = []
        false_alarm_eps = 0

        for ev in reg_alerts:
            st_d = getattr(ev, "first_alert_date", None) or (ev.get("first_alert_date") if isinstance(ev, dict) else "")
            if is_outbreak and (win_start <= st_d <= win_end):
                valid_outbreak_alerts.append(st_d)
            else:
                false_alarm_eps += 1

        first_alert_d = min(valid_outbreak_alerts) if valid_outbreak_alerts else None
        detected = bool(first_alert_d is not None)

        official_report_d = None
        if is_outbreak and first_alert_d:
            post_alert_official = reg_df[(reg_df["date"] >= first_alert_d) & (reg_df["is_official_report"] == True)]
            if not post_alert_official.empty:
                official_report_d = post_alert_official["date"].min()
        elif is_outbreak:
            outbreak_official = reg_df[(reg_df["date"] >= win_start) & (reg_df["is_official_report"] == True)]
            if not outbreak_official.empty:
                official_report_d = outbreak_official["date"].min()

        lead_days = None
        if first_alert_d and official_report_d:
            dt_alert = pd.to_datetime(first_alert_d)
            dt_off = pd.to_datetime(official_report_d)
            lead_days = int((dt_off - dt_alert).days)

        if is_outbreak:
            if detected and official_report_d:
                if lead_days > 0:
                    outcome = "Early Detection"
                elif lead_days == 0:
                    outcome = "Same-Day Detection"
                else:
                    outcome = "Late Detection"
            elif detected and not official_report_d:
                outcome = "Early Detection (Awaiting Official)"
            else:
                outcome = "Missed Detection"
        else:
            if false_alarm_eps > 0:
                outcome = "False Alarm"
            else:
                outcome = "No Alert"

        records.append({
            "Region": reg,
            "Known Outbreak": "Yes" if is_outbreak else "No",
            "First Alert Date": first_alert_d or "N/A",
            "Official Report Date": official_report_d or "N/A",
            "Lead Days": lead_days,
            "Detected": detected,
            "False Alarm Episodes": false_alarm_eps,
            "Outcome": outcome,
        })

    return pd.DataFrame(records)

def calculate_validation_metrics(
    regional_summary: pd.DataFrame,
    alert_events: List[Any],
    outbreak_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Calculate objective performance metrics, lead-time stats, and classification rates."""
    if outbreak_config is None:
        outbreak_config = load_outbreak_metadata()

    outbreak_rows = regional_summary[regional_summary["Known Outbreak"] == "Yes"]
    unaffected_rows = regional_summary[regional_summary["Known Outbreak"] == "No"]

    total_injected = len(outbreak_rows)
    detected_count = int(outbreak_rows["Detected"].sum())
    early_detected_count = len(outbreak_rows[outbreak_rows["Outcome"] == "Early Detection"])

    detection_rate = np.round((detected_count / total_injected * 100.0), 1) if total_injected > 0 else 0.0
    early_detection_rate = np.round((early_detected_count / total_injected * 100.0), 1) if total_injected > 0 else 0.0

    positive_leads = [
        int(r["Lead Days"]) for _, r in outbreak_rows.iterrows()
        if r["Lead Days"] is not None and not pd.isna(r["Lead Days"]) and r["Lead Days"] > 0
    ]
    avg_lead = np.round(float(np.mean(positive_leads)), 1) if positive_leads else 0.0
    med_lead = np.round(float(np.median(positive_leads)), 1) if positive_leads else 0.0

    total_false_alarms = int(regional_summary["False Alarm Episodes"].sum())
    total_alert_eps = len(alert_events)
    false_alarm_rate = np.round((total_false_alarms / total_alert_eps * 100.0), 1) if total_alert_eps > 0 else 0.0

    tp = detected_count
    fp = int(unaffected_rows["False Alarm Episodes"].apply(lambda x: 1 if x > 0 else 0).sum())
    fn = total_injected - tp

    precision = np.round((tp / (tp + fp)), 3) if (tp + fp) > 0 else 0.0
    recall = np.round((tp / (tp + fn)), 3) if (tp + fn) > 0 else 0.0
    f1 = np.round((2.0 * precision * recall / (precision + recall)), 3) if (precision + recall) > 0 else 0.0

    return {
        "total_outbreak_regions": total_injected,
        "detected_outbreaks": detected_count,
        "detection_rate": detection_rate,
        "early_detected_outbreaks": early_detected_count,
        "early_detection_rate": early_detection_rate,
        "average_lead_time": avg_lead,
        "median_lead_time": med_lead,
        "positive_lead_days": positive_leads,
        "total_alert_episodes": total_alert_eps,
        "false_alert_episodes": total_false_alarms,
        "false_alarm_rate": false_alarm_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

def run_threshold_sensitivity_analysis(
    df: pd.DataFrame,
    z_thresholds: Optional[List[float]] = None,
    alert_thresholds: Optional[List[float]] = None,
    outbreak_config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Evaluate performance trade-offs across various sensitivity (Z) and risk thresholds."""
    if z_thresholds is None:
        z_thresholds = [1.5, 2.0, 2.5, 3.0]
    if alert_thresholds is None:
        alert_thresholds = [40.0, 50.0, 60.0, 70.0, 80.0]
    if outbreak_config is None:
        outbreak_config = load_outbreak_metadata()

    from detector import calculate_rolling_baseline, calculate_z_scores
    base_df = calculate_rolling_baseline(df, signal_columns=LEADING_SIGNALS)
    z_df = calculate_z_scores(base_df, signal_columns=LEADING_SIGNALS)

    records = []
    default_weights = {"wastewater": 0.35, "otc_sales": 0.25, "search_trends": 0.20, "sick_leave": 0.20}

    for z_th in z_thresholds:
        anom_df = detect_anomalies(z_df, z_threshold=z_th)
        risk_df = calculate_risk_score(anom_df, weights=default_weights)
        off_df = calculate_official_thresholds(risk_df)

        for a_th in alert_thresholds:
            tagged_df, alert_events = generate_alerts(
                off_df,
                alert_threshold=a_th,
                minimum_signals=2,
                consecutive_days=2,
            )
            reg_summary = evaluate_regional_performance(tagged_df, alert_events, outbreak_config)
            metrics = calculate_validation_metrics(reg_summary, alert_events, outbreak_config)

            records.append({
                "z_threshold": z_th,
                "alert_threshold": int(a_th),
                "detection_rate": metrics["detection_rate"],
                "early_detection_rate": metrics["early_detection_rate"],
                "avg_lead_time": metrics["average_lead_time"],
                "false_alarms": metrics["false_alert_episodes"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            })

    return pd.DataFrame(records)

def export_validation_csv(regional_summary: pd.DataFrame) -> str:
    """Format regional validation summary into a clean CSV string for export."""
    return regional_summary.to_csv(index=False)
