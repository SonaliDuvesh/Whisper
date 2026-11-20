"""Whisper Synthetic Workforce Health Dataset Generator.

Generates aggregated organizational health and sick-leave telemetry across 5 fictional
workplace organizations and industrial sites, aligned with the regional outbreak waves.

Strict Privacy Principles:
- Strictly aggregated and anonymized employee counts.
- Zero individual health records, zero names, phone numbers, or personal identifiers.
- Compliance with public health and organizational ethics safeguards.
"""

from typing import Dict, List, Optional
import os
import numpy as np
import pandas as pd

def generate_organizations() -> Dict[str, Dict]:
    """Define 5 fictional synthetic workplace organizations across distinct regions.

    Returns:
        dict: Metadata for each synthetic organization site.
    """
    return {
        "Factory A": {
            "organization": "Factory A",
            "site": "Plant 1 - Precision Assembly",
            "region": "Region 3",
            "lat": 12.9810,
            "lon": 77.5800,
            "total_employees": 1250,
            "base_absence_rate": 0.030,
            "noise_std": 0.004,
            "outbreak_multiplier": 3.8,
            "outbreak_delay_days": 1,
        },
        "Call Center B": {
            "organization": "Call Center B",
            "site": "Customer Operations Alpha",
            "region": "Region 1",
            "lat": 28.6250,
            "lon": 77.2150,
            "total_employees": 850,
            "base_absence_rate": 0.035,
            "noise_std": 0.005,
            "outbreak_multiplier": 1.0,
            "outbreak_delay_days": 0,
        },
        "Logistics Hub C": {
            "organization": "Logistics Hub C",
            "site": "Central Freight Terminal",
            "region": "Region 4",
            "lat": 22.5800,
            "lon": 88.3500,
            "total_employees": 620,
            "base_absence_rate": 0.032,
            "noise_std": 0.004,
            "outbreak_multiplier": 2.4,
            "outbreak_delay_days": 2,
        },
        "Office Campus D": {
            "organization": "Office Campus D",
            "site": "Innovation Tech Park",
            "region": "Region 2",
            "lat": 19.0850,
            "lon": 72.8850,
            "total_employees": 2100,
            "base_absence_rate": 0.028,
            "noise_std": 0.003,
            "outbreak_multiplier": 1.0,
            "outbreak_delay_days": 0,
        },
        "Manufacturing Unit E": {
            "organization": "Manufacturing Unit E",
            "site": "Heavy Industrial Works",
            "region": "Region 5",
            "lat": 21.1500,
            "lon": 79.0950,
            "total_employees": 1500,
            "base_absence_rate": 0.034,
            "noise_std": 0.004,
            "outbreak_multiplier": 2.1,
            "outbreak_delay_days": 3,
        },
    }

def generate_workforce_signals(
    start_date: str = "2024-01-01",
    days: int = 730,
    random_seed: int = 42,
    outbreak_start_date: str = "2025-03-10",
) -> pd.DataFrame:
    """Generate daily aggregated workforce health observations.

    Args:
        start_date: Initial observation date (YYYY-MM-DD).
        days: Number of consecutive simulation days (730 = 2 years).
        random_seed: Seed for deterministic reproducibility.
        outbreak_start_date: Ground truth start of the March 2025 wave.

    Returns:
        pd.DataFrame: Validated synthetic workforce dataset.
    """
    np.random.seed(random_seed)
    date_range = pd.date_range(start=start_date, periods=days, freq="D")
    orgs = generate_organizations()

    records = []
    outbreak_start_dt = pd.to_datetime(outbreak_start_date)

    for current_dt in date_range:
        d_str = current_dt.strftime("%Y-%m-%d")
        day_of_year = current_dt.day_of_year if hasattr(current_dt, "day_of_year") else current_dt.dayofyear
        day_of_week = current_dt.day_of_week if hasattr(current_dt, "day_of_week") else current_dt.dayofweek

        winter_factor = 1.0 + 0.15 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)

        dow_multipliers = {0: 1.10, 1: 0.98, 2: 0.95, 3: 0.96, 4: 1.08, 5: 0.85, 6: 0.80}
        dow_factor = dow_multipliers.get(day_of_week, 1.0)

        for org_id, org_info in orgs.items():
            base_rate = org_info["base_absence_rate"]
            tot_emp = org_info["total_employees"]
            noise_std = org_info["noise_std"]

            normal_rate = base_rate * winter_factor * dow_factor
            noise = np.random.normal(0, noise_std)
            eff_rate = normal_rate + noise

            multiplier = org_info["outbreak_multiplier"]
            if multiplier > 1.0:
                surge_onset_dt = outbreak_start_dt + pd.Timedelta(days=org_info["outbreak_delay_days"])
                if current_dt >= surge_onset_dt:
                    days_since_surge = (current_dt - surge_onset_dt).days
                    if days_since_surge <= 35:
                        if days_since_surge <= 8:
                            wave_shape = (days_since_surge / 8.0)
                        elif days_since_surge <= 14:
                            wave_shape = 1.0 - 0.05 * (days_since_surge - 8)
                        else:
                            wave_shape = max(0.0, 0.70 * np.exp(-(days_since_surge - 14) / 7.0))

                        added_rate = base_rate * (multiplier - 1.0) * wave_shape
                        eff_rate += added_rate

            eff_rate = float(np.clip(eff_rate, 0.005, 0.40))
            absent_count = int(np.round(tot_emp * eff_rate))
            absent_count = min(absent_count, tot_emp)

            sick_leave_pct = np.round((absent_count / tot_emp) * 100.0, 2)
            available_employees = tot_emp - absent_count
            operational_capacity_pct = np.round((available_employees / tot_emp) * 100.0, 2)
            baseline_absence_pct = np.round(base_rate * 100.0, 2)

            records.append({
                "date": d_str,
                "organization": org_id,
                "site": org_info["site"],
                "region": org_info["region"],
                "lat": org_info["lat"],
                "lon": org_info["lon"],
                "total_employees": tot_emp,
                "absent_employees": absent_count,
                "sick_leave": sick_leave_pct,
                "baseline_absence": baseline_absence_pct,
                "operational_capacity": operational_capacity_pct,
            })

    df = pd.DataFrame(records)
    return df

def validate_workforce_dataset(df: pd.DataFrame) -> bool:
    """Validate data integrity and ethical constraints of the workforce dataset."""
    required_cols = [
        "date", "organization", "site", "region", "lat", "lon",
        "total_employees", "absent_employees", "sick_leave",
        "baseline_absence", "operational_capacity",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in workforce dataset: {col}")

    if df.isnull().any().any():
        raise ValueError("Workforce dataset contains null values")

    if (df["absent_employees"] < 0).any():
        raise ValueError("Negative absent employees found")
    if (df["total_employees"] <= 0).any():
        raise ValueError("Non-positive total employees found")
    if (df["absent_employees"] > df["total_employees"]).any():
        raise ValueError("Absent employees cannot exceed total employees")

    if (df["sick_leave"] < 0).any() or (df["sick_leave"] > 100).any():
        raise ValueError("Sick leave percentage out of bounds [0, 100]")
    if (df["operational_capacity"] < 0).any() or (df["operational_capacity"] > 100).any():
        raise ValueError("Operational capacity out of bounds [0, 100]")

    return True

def save_workforce_dataset(df: pd.DataFrame, output_path: str = "data/workforce_signals.csv") -> None:
    """Save validated dataset to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

def main():
    """Generate and validate data/workforce_signals.csv."""
    print("Generating Whisper synthetic workforce dataset...")
    df = generate_workforce_signals(start_date="2024-01-01", days=730, random_seed=42)
    validate_workforce_dataset(df)
    out_path = os.path.join(os.path.dirname(__file__), "data", "workforce_signals.csv")
    save_workforce_dataset(df, out_path)
    print(f"Successfully generated {len(df)} records across {df['organization'].nunique()} organizations.")
    print(f"Output saved to: {out_path}")

if __name__ == "__main__":
    main()
