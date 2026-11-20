"""Whisper Synthetic Epidemiological Data Generator.

Generates multi-signal surveillance data for 8 fictional regions across
730 days (2 years), incorporating realistic seasonality, leading indicators,
stochastic noise, and an injected multi-region outbreak with demonstrable
detection lead time.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

def generate_region_metadata() -> Dict[str, Dict]:
    """Define 8 fictional geographical demonstration regions distributed across India.

    Returns:
        dict: Region identifiers mapping to name, lat, lon, and baseline characteristics.
    """
    regions = {
        "Region 1": {
            "name": "Northern Plains",
            "lat": 28.6139,
            "lon": 77.2090,
            "base_wastewater": 22.0,
            "base_otc": 55.0,
            "base_search": 35.0,
            "base_sick_leave": 3.2,
            "base_cases": 12,
        },
        "Region 2": {
            "name": "Western Coastal",
            "lat": 19.0760,
            "lon": 72.8777,
            "base_wastewater": 25.0,
            "base_otc": 60.0,
            "base_search": 38.0,
            "base_sick_leave": 3.5,
            "base_cases": 15,
        },
        "Region 3": {
            "name": "Southern Plateau (Epicenter)",
            "lat": 12.9716,
            "lon": 77.5946,
            "base_wastewater": 20.0,
            "base_otc": 50.0,
            "base_search": 32.0,
            "base_sick_leave": 3.0,
            "base_cases": 10,
        },
        "Region 4": {
            "name": "Eastern Delta",
            "lat": 22.5726,
            "lon": 88.3639,
            "base_wastewater": 24.0,
            "base_otc": 58.0,
            "base_search": 36.0,
            "base_sick_leave": 3.4,
            "base_cases": 14,
        },
        "Region 5": {
            "name": "Central Ridge",
            "lat": 23.2599,
            "lon": 77.4126,
            "base_wastewater": 19.0,
            "base_otc": 48.0,
            "base_search": 30.0,
            "base_sick_leave": 2.8,
            "base_cases": 9,
        },
        "Region 6": {
            "name": "Northeastern Valley",
            "lat": 26.1445,
            "lon": 91.7362,
            "base_wastewater": 18.0,
            "base_otc": 45.0,
            "base_search": 28.0,
            "base_sick_leave": 2.6,
            "base_cases": 8,
        },
        "Region 7": {
            "name": "Northwestern Desert",
            "lat": 26.9124,
            "lon": 75.7873,
            "base_wastewater": 21.0,
            "base_otc": 52.0,
            "base_search": 34.0,
            "base_sick_leave": 3.1,
            "base_cases": 11,
        },
        "Region 8": {
            "name": "Southeastern Coast",
            "lat": 13.0827,
            "lon": 80.2707,
            "base_wastewater": 23.0,
            "base_otc": 56.0,
            "base_search": 37.0,
            "base_sick_leave": 3.3,
            "base_cases": 13,
        },
    }
    return regions

def generate_base_signals(
    region_id: str,
    meta: Dict,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate realistic non-outbreak baseline signals with seasonal and weekly patterns.

    Args:
        region_id: Unique region identifier.
        meta: Metadata dictionary containing latitude, longitude, and baseline levels.
        dates: DatetimeIndex of dates.
        rng: Numpy random Generator.

    Returns:
        pd.DataFrame: DataFrame containing daily baseline observations.
    """
    n_days = len(dates)
    day_of_year = dates.day_of_year.values if hasattr(dates, "day_of_year") else dates.dayofyear.values
    day_of_week = dates.day_of_week.values if hasattr(dates, "day_of_week") else dates.dayofweek.values

    winter_phase = 2.0 * np.pi * (day_of_year - 15) / 365.25
    annual_season = 0.25 * np.cos(winter_phase)

    otc_weekly = np.where(day_of_week == 0, 0.12, np.where(day_of_week >= 5, 0.08, -0.04))
    search_weekly = np.where(day_of_week <= 2, 0.06, -0.04)
    sick_weekly = np.where(day_of_week >= 5, -0.40, 0.15)
    cases_weekly = np.where(day_of_week >= 5, -0.25, np.where(day_of_week == 2, 0.20, 0.02))

    noise_ww = rng.normal(0, 1.8, n_days)
    noise_otc = rng.normal(0, 3.2, n_days)
    noise_search = rng.normal(0, 2.5, n_days)
    noise_sick = rng.normal(0, 0.25, n_days)
    noise_cases = rng.normal(0, 2.0, n_days)

    def ar1_filter(noise: np.ndarray, phi: float = 0.6) -> np.ndarray:
        series = np.zeros(len(noise))
        series[0] = noise[0]
        for t in range(1, len(noise)):
            series[t] = phi * series[t - 1] + (1 - phi) * noise[t]
        return series

    smooth_noise_ww = ar1_filter(noise_ww, 0.7)
    smooth_noise_otc = ar1_filter(noise_otc, 0.6)
    smooth_noise_search = ar1_filter(noise_search, 0.6)
    smooth_noise_sick = ar1_filter(noise_sick, 0.7)

    b_ww = meta["base_wastewater"]
    wastewater = b_ww * (1.0 + annual_season) + smooth_noise_ww
    wastewater = np.clip(wastewater, 2.0, None)

    b_otc = meta["base_otc"]
    otc_sales = b_otc * (1.0 + 0.8 * annual_season + otc_weekly) + smooth_noise_otc
    otc_sales = np.clip(otc_sales, 5.0, None)

    b_search = meta["base_search"]
    search_trend = b_search * (1.0 + 0.9 * annual_season + search_weekly) + smooth_noise_search
    search_trend = np.clip(search_trend, 3.0, None)

    b_sick = meta["base_sick_leave"]
    sick_leave = b_sick * (1.0 + 0.6 * annual_season + sick_weekly) + smooth_noise_sick
    sick_leave = np.clip(sick_leave, 0.2, 100.0)

    b_cases = meta["base_cases"]
    base_case_rate = b_cases * (1.0 + annual_season + cases_weekly) + noise_cases
    official_cases = np.clip(np.round(base_case_rate), 0, None).astype(int)

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "region": region_id,
        "lat": meta["lat"],
        "lon": meta["lon"],
        "wastewater": np.round(wastewater, 2),
        "otc_sales": np.round(otc_sales, 2),
        "search_trend": np.round(search_trend, 2),
        "sick_leave": np.round(sick_leave, 2),
        "official_cases": official_cases,
    })
    return df

def inject_outbreak(df: pd.DataFrame, outbreak_config: Dict) -> pd.DataFrame:
    """Inject a realistic synthetic outbreak with realistic lead times across target regions.

    Signal Sequence (Lead times relative to official cases threshold crossing):
    1. Wastewater: begins rising first (~10 days prior)
    2. OTC Medicine Sales: begins rising ~7 days prior
    3. Search Trends: begins rising ~6 days prior
    4. Workplace Sick Leave: begins rising ~4 days prior
    5. Official Cases: rises last (lagging ground truth standard)

    Args:
        df: Base DataFrame of all regions and dates.
        outbreak_config: Configuration dictionary specifying start date,
                         affected regions, and scaling factors.

    Returns:
        pd.DataFrame: DataFrame with the outbreak signals injected.
    """
    df = df.copy()
    dates = pd.to_datetime(df["date"])

    affected_regions = outbreak_config.get("affected_regions", {})
    start_date = pd.to_datetime(outbreak_config.get("start_date", "2025-03-10"))

    def surge(t_series: pd.Series, t_onset: pd.Timestamp, peak_val: float, rise_days: float, fall_days: float) -> np.ndarray:
        dt = (t_series - t_onset).dt.total_seconds() / 86400.0
        val = np.zeros(len(t_series))
        mask_rise = (dt >= 0) & (dt <= rise_days)
        val[mask_rise] = peak_val * 0.5 * (1.0 - np.cos(np.pi * dt[mask_rise] / rise_days))
        mask_fall = (dt > rise_days) & (dt <= rise_days + fall_days)
        dt_fall = dt[mask_fall] - rise_days
        val[mask_fall] = peak_val * 0.5 * (1.0 + np.cos(np.pi * dt_fall / fall_days))
        return val

    for region_id, intensity in affected_regions.items():
        region_mask = (df["region"] == region_id)
        region_dates = dates[region_mask]

        ww_add = surge(region_dates, start_date - pd.Timedelta(days=1), 70.0 * intensity, rise_days=10.0, fall_days=25.0)
        df.loc[region_mask, "wastewater"] = np.round(df.loc[region_mask, "wastewater"] + ww_add, 2)

        otc_add = surge(region_dates, start_date - pd.Timedelta(days=1), 100.0 * intensity, rise_days=10.0, fall_days=25.0)
        df.loc[region_mask, "otc_sales"] = np.round(df.loc[region_mask, "otc_sales"] + otc_add, 2)

        search_add = surge(region_dates, start_date, 90.0 * intensity, rise_days=9.0, fall_days=25.0)
        df.loc[region_mask, "search_trend"] = np.round(df.loc[region_mask, "search_trend"] + search_add, 2)

        sick_add = surge(region_dates, start_date + pd.Timedelta(days=3), 9.0 * intensity, rise_days=9.0, fall_days=22.0)
        new_sick = df.loc[region_mask, "sick_leave"] + sick_add
        df.loc[region_mask, "sick_leave"] = np.round(np.clip(new_sick, 0.0, 100.0), 2)

        cases_add = surge(region_dates, start_date + pd.Timedelta(days=9), 220.0 * intensity, rise_days=11.0, fall_days=28.0)
        new_cases = df.loc[region_mask, "official_cases"] + np.round(cases_add)
        df.loc[region_mask, "official_cases"] = np.clip(new_cases, 0, None).astype(int)

    return df

def validate_dataset(df: pd.DataFrame) -> bool:
    """Validate data integrity, schema, constraints, and chronological structure.

    Args:
        df: The generated synthetic DataFrame.

    Raises:
        ValueError: If any validation rule fails.

    Returns:
        bool: True if validation passes completely.
    """
    required_cols = [
        "date", "region", "lat", "lon", "wastewater",
        "otc_sales", "search_trend", "sick_leave", "official_cases",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Validation failed: Missing columns: {missing_cols}")

    unique_regions = df["region"].unique()
    if len(unique_regions) != 8:
        raise ValueError(f"Validation failed: Expected exactly 8 regions, got {len(unique_regions)}")

    dates_per_region = df.groupby("region")["date"].nunique()
    for reg, count in dates_per_region.items():
        if count != 730:
            raise ValueError(f"Validation failed: Region {reg} has {count} dates, expected 730.")

    dup_count = df.duplicated(subset=["region", "date"]).sum()
    if dup_count > 0:
        raise ValueError(f"Validation failed: Found {dup_count} duplicate (region, date) rows.")

    if df.isna().any().any():
        null_counts = df.isna().sum().to_dict()
        raise ValueError(f"Validation failed: Found NaN values: {null_counts}")

    numeric_cols = ["lat", "lon", "wastewater", "otc_sales", "search_trend", "sick_leave", "official_cases"]
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            raise ValueError(f"Validation failed: Infinite values detected in column '{col}'.")

    if not (df["lat"].between(6.0, 38.0).all() and df["lon"].between(68.0, 98.0).all()):
        raise ValueError("Validation failed: Coordinates out of expected geographical boundaries.")

    for col in ["wastewater", "otc_sales", "search_trend", "official_cases"]:
        if (df[col] < 0).any():
            raise ValueError(f"Validation failed: Negative values detected in column '{col}'.")

    if not df["sick_leave"].between(0.0, 100.0).all():
        raise ValueError("Validation failed: Sick leave values must be strictly between 0 and 100%.")

    if not (df["official_cases"] % 1 == 0).all():
        raise ValueError("Validation failed: Official cases must be integer counts.")

    for reg, grp in df.groupby("region"):
        dates = pd.to_datetime(grp["date"], format="%Y-%m-%d", errors="coerce")
        if dates.isna().any():
            raise ValueError(f"Validation failed: Invalid date format found in region {reg}.")
        if not dates.is_monotonic_increasing:
            raise ValueError(f"Validation failed: Dates are not strictly chronological in region {reg}.")

    return True

def generate_dataset(
    start_date: str = "2024-01-01",
    days: int = 730,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """Generate the complete reproducible synthetic epidemiological dataset.

    Args:
        start_date: Start date string (YYYY-MM-DD).
        days: Total observation days (default 730 = 2 years).
        random_seed: Random seed for reproducibility.

    Returns:
        tuple: (pd.DataFrame, dict of configuration details)
    """
    rng = np.random.default_rng(random_seed)
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    region_metadata = generate_region_metadata()

    frames = []
    for region_id, meta in region_metadata.items():
        reg_df = generate_base_signals(region_id, meta, dates, rng)
        frames.append(reg_df)

    base_df = pd.concat(frames, ignore_index=True)

    outbreak_config = {
        "start_date": "2025-03-10",
        "affected_regions": {
            "Region 3": 1.0,
            "Region 4": 0.72,
            "Region 5": 0.58,
        },
    }

    df = inject_outbreak(base_df, outbreak_config)

    validate_dataset(df)

    config_info = {
        "start_date": dates[0].strftime("%Y-%m-%d"),
        "end_date": dates[-1].strftime("%Y-%m-%d"),
        "days": days,
        "regions_count": len(region_metadata),
        "total_observations": len(df),
        "outbreak_start_date": outbreak_config["start_date"],
        "outbreak_regions": list(outbreak_config["affected_regions"].keys()),
        "random_seed": random_seed,
    }

    return df, config_info

def save_dataset(df: pd.DataFrame, output_path: str = "data/signals.csv") -> None:
    """Save the validated dataset to the designated CSV path.

    Args:
        df: Validated DataFrame.
        output_path: Target file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

def main() -> None:
    """CLI execution entrypoint."""
    print("Whisper Synthetic Data Generator")
    print("----------------------------------")
    print("Generating 8 fictional regions...")
    print("Generating 730 days of observations...")
    print("Injecting synthetic outbreak...")

    df, info = generate_dataset(start_date="2024-01-01", days=730, random_seed=42)

    print("Validating dataset...")
    validate_dataset(df)

    output_path = os.path.join(os.path.dirname(__file__), "data", "signals.csv")
    print(f"Saving {output_path}...")
    save_dataset(df, output_path)

    print()
    print("Dataset generated successfully.")
    print(f"Regions: {info['regions_count']}")
    print(f"Observations: {info['total_observations']}")
    print(f"Outbreak regions: {len(info['outbreak_regions'])} ({', '.join(info['outbreak_regions'])})")
    print("Validation: PASSED")
    print(f"Date range: {info['start_date']} to {info['end_date']}")
    print(f"Outbreak start date: {info['outbreak_start_date']}")
    print(f"Dataset dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Missing values: {df.isna().sum().sum()}")

if __name__ == "__main__":
    main()
