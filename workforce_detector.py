"""Whisper Workforce Health Anomaly Detection & Risk Engine.

Implements leakage-free rolling baseline calculations, Z-score standardization,
organizational risk scoring, severity classification, and consecutive-day alert tracking.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

@dataclass
class WorkforceAlertEvent:
    """Represents a qualified organizational health alert episode."""
    episode_id: str
    organization: str
    site: str
    region: str
    first_alert_date: str
    last_alert_date: str
    initial_risk_score: float
    peak_risk_score: float
    initial_sick_leave: float
    peak_sick_leave: float
    severity: str
    explanation: str
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def load_workforce_data(path: str = "data/workforce_signals.csv") -> pd.DataFrame:
    """Load and validate the synthetic workforce dataset."""
    df = pd.read_csv(path)
    df["date"] = df["date"].astype(str)
    return df.sort_values(["organization", "date"]).reset_index(drop=True)

def calculate_workforce_baselines(
    df: pd.DataFrame,
    window: int = 28,
    min_periods: int = 14,
) -> pd.DataFrame:
    """Calculate leakage-free rolling 28-day baseline mean and standard deviation per organization.

    Formula:
    Mean_t = rolling_mean(sick_leave_{t-window..t-1})
    Std_t  = rolling_std(sick_leave_{t-window..t-1})
    """
    df = df.copy().sort_values(["organization", "date"]).reset_index(drop=True)

    grouped = df.groupby("organization")["sick_leave"]
    roll_mean = grouped.transform(lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).mean())
    roll_std = grouped.transform(lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).std())

    exp_mean = grouped.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    exp_std = grouped.transform(lambda s: s.shift(1).expanding(min_periods=2).std())

    roll_mean = roll_mean.fillna(exp_mean).fillna(df["sick_leave"])
    roll_std = roll_std.fillna(exp_std).fillna(0.3)
    roll_std = np.where(roll_std < 0.2, 0.2, roll_std)

    df["sick_leave_mean"] = np.round(roll_mean, 2)
    df["sick_leave_std"] = np.round(roll_std, 2)
    return df

def detect_workforce_anomalies(
    df: pd.DataFrame,
    z_threshold: float = 2.0,
) -> pd.DataFrame:
    """Standardize sick leave into Z-scores and flag statistically significant deviations."""
    df = df.copy()
    diff = df["sick_leave"] - df["sick_leave_mean"]
    z = diff / df["sick_leave_std"]
    df["z_absence"] = np.round(z, 2)
    df["is_anomaly"] = df["z_absence"] > z_threshold
    return df

def classify_workforce_severity(risk_score: float) -> str:
    """Classify workforce risk into standard four severity tiers."""
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

def calculate_workforce_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate composite workforce risk score (0-100) and assign severity label.

    Formula: Risk = 100 * (clip(z_absence, 0, 5) / 5)
    """
    df = df.copy()
    z_clipped = np.clip(df["z_absence"], 0.0, 5.0)
    risk = 100.0 * (z_clipped / 5.0)
    df["workforce_risk"] = np.round(risk, 1)
    df["severity"] = df["workforce_risk"].apply(classify_workforce_severity)
    return df

def generate_workforce_alerts(
    df: pd.DataFrame,
    alert_threshold: float = 60.0,
    consecutive_days: int = 2,
) -> Tuple[pd.DataFrame, List[WorkforceAlertEvent]]:
    """Apply persistent consecutive-day alert rules and identify unique alert episodes."""
    df = df.copy().sort_values(["organization", "date"]).reset_index(drop=True)
    df["is_alert"] = False
    df["is_alert_start"] = False
    df["episode_id"] = None

    alert_events: List[WorkforceAlertEvent] = []
    episode_counter = 0

    for org, grp in df.groupby("organization"):
        in_alert = False
        consec_count = 0
        current_ep_id = None
        ep_dates = []
        ep_risks = []
        ep_leaves = []
        site_name = grp["site"].iloc[0]
        region_name = grp["region"].iloc[0]

        for idx, row in grp.iterrows():
            is_anom = bool(row["is_anomaly"])
            risk_val = float(row["workforce_risk"])
            qualifies = is_anom and (risk_val >= alert_threshold)

            if qualifies:
                consec_count += 1
            else:
                consec_count = 0

            if consec_count >= consecutive_days:
                if not in_alert:
                    in_alert = True
                    episode_counter += 1
                    current_ep_id = f"WF-ALERT-{episode_counter:03d}"
                    df.at[idx, "is_alert_start"] = True
                    ep_dates = [row["date"]]
                    ep_risks = [risk_val]
                    ep_leaves = [float(row["sick_leave"])]
                else:
                    ep_dates.append(row["date"])
                    ep_risks.append(risk_val)
                    ep_leaves.append(float(row["sick_leave"]))

                df.at[idx, "is_alert"] = True
                df.at[idx, "episode_id"] = current_ep_id

            elif in_alert and not qualifies:
                in_alert = False
                first_d = ep_dates[0]
                last_d = ep_dates[-1]
                init_r = ep_risks[0]
                peak_r = max(ep_risks)
                init_l = ep_leaves[0]
                peak_l = max(ep_leaves)
                sev_label = classify_workforce_severity(peak_r)
                exp_text = f"Persistent sick-leave elevation (+{row['z_absence']:.1f}σ) across {len(ep_dates)} days."

                alert_events.append(
                    WorkforceAlertEvent(
                        episode_id=current_ep_id,
                        organization=org,
                        site=site_name,
                        region=region_name,
                        first_alert_date=first_d,
                        last_alert_date=last_d,
                        initial_risk_score=init_r,
                        peak_risk_score=peak_r,
                        initial_sick_leave=init_l,
                        peak_sick_leave=peak_l,
                        severity=sev_label,
                        explanation=exp_text,
                        is_active=False,
                    )
                )
                current_ep_id = None
                ep_dates = []
                ep_risks = []
                ep_leaves = []

        if in_alert and current_ep_id and ep_dates:
            alert_events.append(
                WorkforceAlertEvent(
                    episode_id=current_ep_id,
                    organization=org,
                    site=site_name,
                    region=region_name,
                    first_alert_date=ep_dates[0],
                    last_alert_date=ep_dates[-1],
                    initial_risk_score=ep_risks[0],
                    peak_risk_score=max(ep_risks),
                    initial_sick_leave=ep_leaves[0],
                    peak_sick_leave=max(ep_leaves),
                    severity=classify_workforce_severity(max(ep_risks)),
                    explanation=f"Active multi-day sick-leave elevation ({len(ep_dates)} days).",
                    is_active=True,
                )
            )

    return df, alert_events

def run_workforce_pipeline(
    df: pd.DataFrame,
    z_threshold: float = 2.0,
    alert_threshold: float = 60.0,
    consecutive_days: int = 2,
) -> Tuple[pd.DataFrame, List[WorkforceAlertEvent]]:
    """Execute end-to-end workforce detection pipeline."""
    df_proc = calculate_workforce_baselines(df)
    df_proc = detect_workforce_anomalies(df_proc, z_threshold=z_threshold)
    df_proc = calculate_workforce_risk(df_proc)
    df_proc, alert_events = generate_workforce_alerts(
        df_proc,
        alert_threshold=alert_threshold,
        consecutive_days=consecutive_days,
    )
    return df_proc, alert_events
