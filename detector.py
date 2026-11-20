"""Whisper Outbreak Detection Engine.

Analytical detection and risk-scoring engine that transforms multi-signal surveillance
data into actionable early-warning alerts, tracks alert episodes, classifies severity,
and calculates outbreak detection lead times relative to official clinical reports.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

LEADING_SIGNALS = ["wastewater", "otc_sales", "search_trend", "sick_leave"]

DEFAULT_WEIGHTS = {
    "wastewater": 0.35,
    "otc_sales": 0.25,
    "search_trend": 0.20,
    "sick_leave": 0.20,
}

WEIGHT_ALIASES = {
    "otc": "otc_sales",
    "search": "search_trend",
    "sick": "sick_leave",
}

SEVERITY_COLORS = {
    "Green": "#2EAD72",
    "Yellow": "#F2C94C",
    "Orange": "#F2994A",
    "Red": "#EB5757",
}

@dataclass
class AlertEvent:
    """Structure representing a distinct outbreak alert episode."""
    region: str
    episode_id: int
    first_alert_date: str
    severity: str
    initial_risk_score: float
    anomalous_signals: List[str]
    explanation: str

def normalize_weights(weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Validate and normalize signal weights so they sum to 1.0.

    Args:
        weights: Dictionary of signal names/aliases to numeric weights.

    Raises:
        ValueError: If weights are negative or all sum to zero.

    Returns:
        dict: Normalized weights dictionary mapping standard signal column names.
    """
    if weights is None:
        return DEFAULT_WEIGHTS.copy()

    standardized: Dict[str, float] = {}
    for k, v in weights.items():
        std_key = WEIGHT_ALIASES.get(k, k)
        if std_key not in LEADING_SIGNALS:
            continue
        if v < 0:
            raise ValueError(f"Signal weight for '{k}' must be non-negative, got {v}")
        standardized[std_key] = float(v)

    for sig in LEADING_SIGNALS:
        if sig not in standardized:
            standardized[sig] = 0.0

    total_sum = sum(standardized.values())
    if total_sum <= 0:
        raise ValueError("Total weights must sum to a positive value. All weights cannot be zero.")

    return {k: v / total_sum for k, v in standardized.items()}

def calculate_rolling_baseline(
    df: pd.DataFrame,
    signal_columns: Optional[List[str]] = None,
    window: int = 28,
    min_periods: int = 14,
) -> pd.DataFrame:
    """Calculate leakage-free rolling baseline mean and standard deviation per region.

    Uses a shifted rolling window (.shift(1)) so the current observation is strictly
    excluded from its own baseline calculation, preventing data leakage.

    Args:
        df: Input DataFrame with 'date', 'region', and signal columns.
        signal_columns: List of columns to compute baselines for.
        window: Rolling window size in days (default: 28).
        min_periods: Minimum observations required (default: 14).

    Returns:
        pd.DataFrame: DataFrame with added *_mean and *_std columns.
    """
    df = df.copy()
    if signal_columns is None:
        signal_columns = LEADING_SIGNALS

    df = df.sort_values(["region", "date"]).reset_index(drop=True)

    for col in signal_columns:
        if col not in df.columns:
            continue

        mean_col = f"{col}_mean"
        std_col = f"{col}_std"

        grouped = df.groupby("region")[col]
        mean_series = grouped.transform(lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).mean())
        std_series = grouped.transform(lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).std())

        exp_mean = grouped.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        exp_std = grouped.transform(lambda s: s.shift(1).expanding(min_periods=2).std())

        mean_series = mean_series.fillna(exp_mean).fillna(df[col])
        std_series = std_series.fillna(exp_std).fillna(1.0)

        std_series = np.where(std_series < 1e-4, 1.0, std_series)

        df[mean_col] = np.round(mean_series, 4)
        df[std_col] = np.round(std_series, 4)

    return df

def calculate_z_scores(
    df: pd.DataFrame,
    signal_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Calculate standardized Z-scores using historical rolling baselines.

    Formula: z = (x - mean) / std

    Args:
        df: DataFrame containing signals and their *_mean and *_std columns.
        signal_columns: Signals to compute Z-scores for.

    Returns:
        pd.DataFrame: DataFrame with added z_* columns.
    """
    df = df.copy()
    if signal_columns is None:
        signal_columns = LEADING_SIGNALS

    for col in signal_columns:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        z_col = f"z_{col}"

        if mean_col not in df.columns or std_col not in df.columns:
            continue

        std_safe = np.where(df[std_col] < 1e-4, 1.0, df[std_col])
        z = (df[col] - df[mean_col]) / std_safe

        z = np.nan_to_num(z, nan=0.0, posinf=5.0, neginf=-5.0)
        df[z_col] = np.round(z, 4)

    return df

def detect_anomalies(
    df: pd.DataFrame,
    z_threshold: float = 2.0,
) -> pd.DataFrame:
    """Flag anomalous leading indicators where the Z-score exceeds the sensitivity threshold.

    Creates standard boolean columns:
    - anomaly_wastewater
    - anomaly_otc (and anomaly_otc_sales)
    - anomaly_search (and anomaly_search_trend)
    - anomaly_sick (and anomaly_sick_leave)
    - anomalous_signals_count

    Args:
        df: DataFrame containing z_* columns.
        z_threshold: Sensitivity threshold (configurable, default 2.0).

    Returns:
        pd.DataFrame: DataFrame with anomaly_* boolean columns and anomalous_signals_count.
    """
    df = df.copy()
    if not (1.0 <= z_threshold <= 5.0):
        raise ValueError(f"z_threshold must be between 1.0 and 5.0, got {z_threshold}")

    sig_map = {
        "wastewater": ["anomaly_wastewater"],
        "otc_sales": ["anomaly_otc", "anomaly_otc_sales"],
        "search_trend": ["anomaly_search", "anomaly_search_trend"],
        "sick_leave": ["anomaly_sick", "anomaly_sick_leave"],
    }

    primary_anomaly_cols = []
    for sig, anom_names in sig_map.items():
        z_col = f"z_{sig}"
        is_anom = (df[z_col] > z_threshold) if z_col in df.columns else pd.Series(False, index=df.index)
        for name in anom_names:
            df[name] = is_anom
        primary_anomaly_cols.append(anom_names[0])

    df["anomalous_signals_count"] = df[primary_anomaly_cols].sum(axis=1)
    return df

def calculate_risk_score(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Calculate a 0–100 composite risk score from leading indicators.

    Formula: Risk = 100 * clip(sum(w_i * clip(z_i, 0, 5)), 0, 5) / 5

    Official cases are strictly excluded from this calculation.

    Args:
        df: DataFrame with z_* columns.
        weights: Optional dictionary of signal weights.

    Returns:
        pd.DataFrame: DataFrame with 'risk_score' and 'severity' columns.
    """
    df = df.copy()
    w = normalize_weights(weights)

    weighted_sum = np.zeros(len(df))
    for sig in LEADING_SIGNALS:
        z_col = f"z_{sig}"
        if z_col in df.columns:
            z_clipped = np.clip(df[z_col].values, 0.0, 5.0)
            weighted_sum += w[sig] * z_clipped

    risk_score = 100.0 * np.clip(weighted_sum, 0.0, 5.0) / 5.0
    df["risk_score"] = np.round(np.clip(risk_score, 0.0, 100.0), 2)
    df["severity"] = df["risk_score"].apply(classify_severity)

    return df

def classify_severity(risk_score: float) -> str:
    """Classify risk score into standardized severity categories.

    Categories:
    - Below 40: Green
    - 40–59.99: Yellow
    - 60–79.99: Orange
    - 80–100: Red

    Args:
        risk_score: Numeric score between 0 and 100.

    Returns:
        str: Severity label ("Green", "Yellow", "Orange", "Red").
    """
    if pd.isna(risk_score):
        return "Green"
    score = float(risk_score)
    if score < 40.0:
        return "Green"
    elif score < 60.0:
        return "Yellow"
    elif score < 80.0:
        return "Orange"
    else:
        return "Red"

def generate_alerts(
    df: pd.DataFrame,
    alert_threshold: float = 60.0,
    minimum_signals: int = 2,
    consecutive_days: int = 2,
) -> Tuple[pd.DataFrame, List[AlertEvent]]:
    """Apply persistent consecutive-day alert rules and identify alert episodes.

    Rules:
    - A day qualifies if anomalous_signals_count >= minimum_signals AND risk_score >= alert_threshold.
    - An alert fires/remains active if qualifying condition holds for consecutive_days.
    - Maintains state and avoids creating duplicate episode IDs for an ongoing outbreak.

    Args:
        df: DataFrame with anomalous_signals_count and risk_score.
        alert_threshold: Minimum risk score required (default 60).
        minimum_signals: Minimum anomalous signals (default 2).
        consecutive_days: Consecutive qualifying days required (default 2).

    Returns:
        tuple: (DataFrame with alert tracking columns, list of AlertEvent objects)
    """
    df = df.copy().sort_values(["region", "date"]).reset_index(drop=True)

    daily_qualifying = (
        (df["anomalous_signals_count"] >= minimum_signals) &
        (df["risk_score"] >= alert_threshold)
    )
    df["daily_qualifying"] = daily_qualifying

    is_alert = np.zeros(len(df), dtype=bool)
    is_alert_start = np.zeros(len(df), dtype=bool)
    alert_episode_id = [None] * len(df)
    alert_events: List[AlertEvent] = []

    global_episode_counter = 0

    for region, grp in df.groupby("region"):
        indices = grp.index.values
        streak = 0
        currently_in_alert = False
        current_episode_id = None

        for idx in indices:
            qualifies = bool(df.loc[idx, "daily_qualifying"])

            if qualifies:
                streak += 1
                if streak >= consecutive_days:
                    is_alert[idx] = True
                    if not currently_in_alert:
                        global_episode_counter += 1
                        current_episode_id = global_episode_counter
                        currently_in_alert = True
                        is_alert_start[idx] = True

                        anom_signals = []
                        for sig in LEADING_SIGNALS:
                            for col_candidate in [f"anomaly_{sig}", f"anomaly_{sig.split('_')[0]}"]:
                                if col_candidate in df.columns and bool(df.loc[idx, col_candidate]):
                                    anom_signals.append(sig)
                                    break

                        event = AlertEvent(
                            region=region,
                            episode_id=current_episode_id,
                            first_alert_date=str(df.loc[idx, "date"]),
                            severity=str(df.loc[idx, "severity"]),
                            initial_risk_score=float(df.loc[idx, "risk_score"]),
                            anomalous_signals=anom_signals,
                            explanation=(
                                f"Multi-signal outbreak alert initiated in {region}. "
                                f"{len(anom_signals)} leading indicators abnormal "
                                f"({', '.join(anom_signals)}) with risk score {df.loc[idx, 'risk_score']}."
                            ),
                        )
                        alert_events.append(event)

                    alert_episode_id[idx] = current_episode_id
                else:
                    is_alert[idx] = False
                    currently_in_alert = False
                    current_episode_id = None
            else:
                streak = 0
                is_alert[idx] = False
                currently_in_alert = False
                current_episode_id = None

    df["is_alert"] = is_alert
    df["is_alert_start"] = is_alert_start
    df["alert_episode_id"] = alert_episode_id

    return df, alert_events

def calculate_official_thresholds(
    df: pd.DataFrame,
    sigma_threshold: float = 2.0,
    window: int = 28,
    min_periods: int = 14,
) -> pd.DataFrame:
    """Calculate leakage-free historical reporting thresholds for official clinical cases.

    Official threshold: mu_cases + sigma_threshold * sigma_cases

    Args:
        df: Input DataFrame containing 'official_cases'.
        sigma_threshold: Multiplier for standard deviation (default 2.0).
        window: Historical rolling baseline window (default 28).
        min_periods: Minimum observations for baseline (default 14).

    Returns:
        pd.DataFrame: DataFrame with official_threshold and is_official_report columns.
    """
    df = df.copy().sort_values(["region", "date"]).reset_index(drop=True)

    grouped = df.groupby("region")["official_cases"]
    cases_mean = grouped.transform(lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).mean())
    cases_std = grouped.transform(lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).std())

    exp_mean = grouped.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    exp_std = grouped.transform(lambda s: s.shift(1).expanding(min_periods=2).std())

    cases_mean = cases_mean.fillna(exp_mean).fillna(df["official_cases"])
    cases_std = cases_std.fillna(exp_std).fillna(1.0)
    cases_std = np.where(cases_std < 1.0, 1.0, cases_std)

    threshold = cases_mean + sigma_threshold * cases_std
    df["official_threshold"] = np.round(threshold, 2)
    df["is_official_report"] = df["official_cases"] > df["official_threshold"]

    return df

def calculate_lead_times(
    df: pd.DataFrame,
    outbreak_config: Optional[Dict] = None,
) -> pd.DataFrame:
    """Calculate outbreak detection lead time per region and evaluate false alarms.

    Lead time formula: LeadDays = OfficialReportDate - FirstAlertDate

    Args:
        df: Processed DataFrame containing 'is_alert_start' and 'is_official_report'.
        outbreak_config: Ground truth configuration specifying affected regions and start date.

    Returns:
        pd.DataFrame: Summary table with columns [region, first_alert_date,
                      official_report_date, lead_days, outbreak_region, detected, false_alarm_count].
    """
    if outbreak_config is None:
        outbreak_config = {
            "start_date": "2025-03-10",
            "affected_regions": ["Region 3", "Region 4", "Region 5"],
        }

    affected_regions = set(outbreak_config.get("affected_regions", []))
    outbreak_start = pd.to_datetime(outbreak_config.get("start_date", "2025-03-10"))

    records = []
    df_work = df.copy()
    df_work["date_dt"] = pd.to_datetime(df_work["date"])

    for region, grp in df_work.groupby("region"):
        is_outbreak_region = region in affected_regions

        alert_starts = grp[grp["is_alert_start"] == True]
        if not alert_starts.empty:
            first_alert_dt = alert_starts["date_dt"].min()
            first_alert_date = first_alert_dt.strftime("%Y-%m-%d")
        else:
            first_alert_dt = None
            first_alert_date = None

        if is_outbreak_region and first_alert_dt is not None:
            post_alert = grp[(grp["date_dt"] >= first_alert_dt) & (grp["is_official_report"] == True)]
            if not post_alert.empty:
                official_report_dt = post_alert["date_dt"].min()
            else:
                pre_alert = grp[(grp["date_dt"] >= outbreak_start) & (grp["is_official_report"] == True)]
                official_report_dt = pre_alert["date_dt"].min() if not pre_alert.empty else None

            official_report_date = official_report_dt.strftime("%Y-%m-%d") if official_report_dt is not None else None
        else:
            official_report_dt = None
            official_report_date = None

        if first_alert_dt is not None and official_report_dt is not None:
            lead_days = float((official_report_dt - first_alert_dt).days)
        else:
            lead_days = None

        if is_outbreak_region:
            detected = bool(first_alert_dt is not None and first_alert_dt >= outbreak_start - pd.Timedelta(days=5))
        else:
            detected = False

        if not is_outbreak_region:
            false_alarm_count = len(alert_starts)
        else:
            pre_outbreak_alerts = alert_starts[alert_starts["date_dt"] < outbreak_start - pd.Timedelta(days=5)]
            false_alarm_count = len(pre_outbreak_alerts)

        records.append({
            "region": region,
            "first_alert_date": first_alert_date,
            "official_report_date": official_report_date,
            "lead_days": lead_days,
            "outbreak_region": is_outbreak_region,
            "detected": detected,
            "false_alarm_count": false_alarm_count,
        })

    return pd.DataFrame(records)

def run_detection_pipeline(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    z_threshold: float = 2.0,
    alert_threshold: float = 60.0,
    minimum_signals: int = 2,
    consecutive_days: int = 2,
    outbreak_config: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, List[AlertEvent], pd.DataFrame]:
    """Execute the complete end-to-end Whisper outbreak detection pipeline.

    Args:
        df: Raw signals surveillance DataFrame.
        weights: Configurable signal weights for risk scoring.
        z_threshold: Anomaly detection Z-score threshold.
        alert_threshold: Minimum risk score for alert qualification.
        minimum_signals: Minimum anomalous leading indicators required.
        consecutive_days: Required consecutive qualifying days.
        outbreak_config: Configuration of known injected outbreak.

    Returns:
        tuple: (processed_df, alert_events, validation_summary)
    """
    df_processed = calculate_rolling_baseline(df, signal_columns=LEADING_SIGNALS)

    df_processed = calculate_z_scores(df_processed, signal_columns=LEADING_SIGNALS)

    df_processed = detect_anomalies(df_processed, z_threshold=z_threshold)

    df_processed = calculate_risk_score(df_processed, weights=weights)

    df_processed, alert_events = generate_alerts(
        df_processed,
        alert_threshold=alert_threshold,
        minimum_signals=minimum_signals,
        consecutive_days=consecutive_days,
    )

    df_processed = calculate_official_thresholds(df_processed)

    validation_summary = calculate_lead_times(df_processed, outbreak_config=outbreak_config)

    return df_processed, alert_events, validation_summary

def main():
    """CLI execution entrypoint for running the detection engine on signals.csv."""
    import os

    csv_path = os.path.join(os.path.dirname(__file__), "data", "signals.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run python data_gen.py first.")
        return

    print("Whisper Outbreak Detection Engine")
    print("==================================")
    print("Loading data/signals.csv...")
    df = pd.read_csv(csv_path)

    print("Running detection pipeline...")
    processed_df, alert_events, validation_summary = run_detection_pipeline(df)

    print("\n--- REGIONAL VALIDATION SUMMARY ---")
    print(validation_summary.to_string(index=False))

    print(f"\nTotal alert episodes generated: {len(alert_events)}")
    for ev in alert_events:
        print(f"  [{ev.first_alert_date}] {ev.region} - Severity: {ev.severity} (Risk: {ev.initial_risk_score}) | {ev.explanation}")

    detected_count = validation_summary[validation_summary["outbreak_region"]]["detected"].sum()
    total_outbreak_regs = validation_summary["outbreak_region"].sum()
    total_false_alarms = validation_summary["false_alarm_count"].sum()

    print(f"\nDetection Performance:")
    print(f"  Outbreak regions detected: {detected_count}/{total_outbreak_regs}")
    print(f"  Total false alarm episodes: {total_false_alarms}")

if __name__ == "__main__":
    main()
