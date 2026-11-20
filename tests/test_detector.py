"""Unit tests for Whisper Outbreak Detection Engine (Prompt 2).

Validates all 17 required algorithmic, mathematical, and boundary rules:
1. Rolling baselines do not include current-day observations (leakage-free).
2. Z-scores are calculated correctly.
3. Zero standard deviation does not produce infinity.
4. Anomaly thresholds work correctly.
5. Risk scores remain strictly within [0, 100].
6. Risk weights are normalized correctly.
7. Zero total weights raise a ValueError.
8. Official cases are strictly excluded from the risk calculation.
9. One anomalous signal does not trigger an alert.
10. One qualifying day does not trigger an alert.
11. Two consecutive qualifying days trigger an alert.
12. Duplicate alert episodes are not created for continuous active alerts.
13. Severity boundaries are correct.
14. Lead-time calculations are correct.
15. Missing official dates are handled properly (None, not 0).
16. Unaffected regions are evaluated for false alarms.
17. Historical calculations do not use future information.
"""

import unittest
import numpy as np
import pandas as pd

from detector import (
    calculate_rolling_baseline,
    calculate_z_scores,
    detect_anomalies,
    calculate_risk_score,
    classify_severity,
    generate_alerts,
    calculate_official_thresholds,
    calculate_lead_times,
    normalize_weights,
    run_detection_pipeline,
)

class TestDetector(unittest.TestCase):
    """Test suite covering analytical correctness of detector.py."""

    def setUp(self):
        """Build small, controlled synthetic datasets for rigorous unit testing."""
        dates = pd.date_range("2025-01-01", periods=30, freq="D")
        self.dates = dates
        self.df_controlled = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "region": "Region A",
            "lat": 20.0,
            "lon": 75.0,
            "wastewater": [10.0] * 29 + [100.0],
            "otc_sales": [20.0] * 30,
            "search_trend": [15.0] * 30,
            "sick_leave": [2.0] * 30,
            "official_cases": [5] * 30,
        })

    def test_01_rolling_baseline_excludes_current_day(self):
        """Requirement 1: Rolling baselines must not include current-day observations."""
        df_base = calculate_rolling_baseline(self.df_controlled, window=7, min_periods=3)
        last_row = df_base.iloc[-1]
        self.assertAlmostEqual(last_row["wastewater_mean"], 10.0, places=2)
        self.assertNotAlmostEqual(last_row["wastewater_mean"], 22.85, places=1)

    def test_02_z_score_calculation(self):
        """Requirement 2: Z-scores are calculated correctly (x - mean) / std."""
        test_df = pd.DataFrame({
            "date": ["2025-01-01"],
            "region": ["Region A"],
            "wastewater": [25.0],
            "wastewater_mean": [15.0],
            "wastewater_std": [5.0],
        })
        res = calculate_z_scores(test_df, signal_columns=["wastewater"])
        expected_z = (25.0 - 15.0) / 5.0
        self.assertAlmostEqual(res["z_wastewater"].iloc[0], expected_z, places=3)

    def test_03_zero_std_no_infinity(self):
        """Requirement 3: Zero standard deviation does not produce infinity or NaN."""
        test_df = pd.DataFrame({
            "date": ["2025-01-01"],
            "region": ["Region A"],
            "wastewater": [10.0],
            "wastewater_mean": [10.0],
            "wastewater_std": [0.0],
        })
        res = calculate_z_scores(test_df, signal_columns=["wastewater"])
        z_val = res["z_wastewater"].iloc[0]
        self.assertFalse(np.isinf(z_val))
        self.assertFalse(np.isnan(z_val))
        self.assertEqual(z_val, 0.0)

    def test_04_anomaly_thresholds(self):
        """Requirement 4: Anomaly flags activate only when Z exceeds threshold."""
        test_df = pd.DataFrame({
            "z_wastewater": [1.9, 2.0, 2.1],
            "z_otc_sales": [0.5, 3.0, 0.2],
            "z_search_trend": [1.0, 1.0, 1.0],
            "z_sick_leave": [0.0, 0.0, 0.0],
        })
        res = detect_anomalies(test_df, z_threshold=2.0)
        self.assertFalse(res["anomaly_wastewater"].iloc[0])
        self.assertEqual(res["anomalous_signals_count"].iloc[0], 0)
        self.assertFalse(res["anomaly_wastewater"].iloc[1])
        self.assertTrue(res["anomaly_otc"].iloc[1])
        self.assertEqual(res["anomalous_signals_count"].iloc[1], 1)
        self.assertTrue(res["anomaly_wastewater"].iloc[2])

    def test_05_risk_score_bounds(self):
        """Requirement 5: Risk scores remain strictly within [0, 100]."""
        test_df = pd.DataFrame({
            "z_wastewater": [-10.0, 0.0, 3.0, 100.0],
            "z_otc_sales": [-5.0, 0.0, 2.0, 50.0],
            "z_search_trend": [-2.0, 0.0, 2.5, 40.0],
            "z_sick_leave": [-1.0, 0.0, 1.5, 30.0],
        })
        res = calculate_risk_score(test_df)
        self.assertTrue((res["risk_score"] >= 0.0).all())
        self.assertTrue((res["risk_score"] <= 100.0).all())
        self.assertEqual(res["risk_score"].iloc[0], 0.0)
        self.assertEqual(res["risk_score"].iloc[3], 100.0)

    def test_06_weight_normalization(self):
        """Requirement 6: Risk weights are correctly normalized to sum to 1.0."""
        raw_weights = {"wastewater": 2.0, "otc": 2.0, "search": 3.0, "sick": 3.0}
        norm = normalize_weights(raw_weights)
        self.assertAlmostEqual(sum(norm.values()), 1.0)
        self.assertAlmostEqual(norm["wastewater"], 0.20)
        self.assertAlmostEqual(norm["search_trend"], 0.30)

    def test_07_zero_total_weights_raises_error(self):
        """Requirement 7: Zero total weights raise a meaningful ValueError."""
        with self.assertRaises(ValueError):
            normalize_weights({"wastewater": 0.0, "otc": 0.0, "search": 0.0, "sick": 0.0})

    def test_08_official_cases_excluded_from_risk(self):
        """Requirement 8: Official cases are strictly excluded from risk scoring."""
        test_df = pd.DataFrame({
            "z_wastewater": [2.0],
            "z_otc_sales": [2.0],
            "z_search_trend": [2.0],
            "z_sick_leave": [2.0],
            "official_cases": [10000],
        })
        res1 = calculate_risk_score(test_df)
        test_df["official_cases"] = [0]
        res2 = calculate_risk_score(test_df)
        self.assertEqual(res1["risk_score"].iloc[0], res2["risk_score"].iloc[0])

    def test_09_one_anomalous_signal_no_alert(self):
        """Requirement 9: One anomalous signal does not trigger an alert."""
        test_df = pd.DataFrame({
            "date": ["2025-01-01", "2025-01-02"],
            "region": ["Region A", "Region A"],
            "anomalous_signals_count": [1, 1],
            "risk_score": [75.0, 75.0],
            "anomaly_wastewater": [True, True],
            "anomaly_otc": [False, False],
            "anomaly_search": [False, False],
            "anomaly_sick": [False, False],
            "severity": ["Orange", "Orange"],
        })
        df_alert, events = generate_alerts(test_df, alert_threshold=60, minimum_signals=2)
        self.assertFalse(df_alert["is_alert"].any())
        self.assertEqual(len(events), 0)

    def test_10_one_qualifying_day_no_alert(self):
        """Requirement 10: A single qualifying day does not trigger an alert."""
        test_df = pd.DataFrame({
            "date": ["2025-01-01", "2025-01-02"],
            "region": ["Region A", "Region A"],
            "anomalous_signals_count": [0, 3],
            "risk_score": [20.0, 80.0],
            "anomaly_wastewater": [False, True],
            "anomaly_otc": [False, True],
            "anomaly_search": [False, True],
            "anomaly_sick": [False, False],
            "severity": ["Green", "Red"],
        })
        df_alert, events = generate_alerts(test_df, alert_threshold=60, minimum_signals=2, consecutive_days=2)
        self.assertFalse(df_alert["is_alert"].any())
        self.assertEqual(len(events), 0)

    def test_11_two_consecutive_days_trigger_alert(self):
        """Requirement 11: Two consecutive qualifying days trigger an alert."""
        test_df = pd.DataFrame({
            "date": ["2025-01-01", "2025-01-02"],
            "region": ["Region A", "Region A"],
            "anomalous_signals_count": [2, 2],
            "risk_score": [65.0, 70.0],
            "anomaly_wastewater": [True, True],
            "anomaly_otc": [True, True],
            "anomaly_search": [False, False],
            "anomaly_sick": [False, False],
            "severity": ["Orange", "Orange"],
        })
        df_alert, events = generate_alerts(test_df, alert_threshold=60, minimum_signals=2, consecutive_days=2)
        self.assertFalse(df_alert["is_alert"].iloc[0])
        self.assertTrue(df_alert["is_alert"].iloc[1])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].first_alert_date, "2025-01-02")

    def test_12_no_duplicate_alert_episodes(self):
        """Requirement 12: Duplicate alert episodes are not created for an ongoing alert."""
        test_df = pd.DataFrame({
            "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
            "region": ["Region A"] * 4,
            "anomalous_signals_count": [2, 2, 2, 2],
            "risk_score": [70.0, 75.0, 80.0, 85.0],
            "anomaly_wastewater": [True] * 4,
            "anomaly_otc": [True] * 4,
            "anomaly_search": [False] * 4,
            "anomaly_sick": [False] * 4,
            "severity": ["Orange", "Orange", "Red", "Red"],
        })
        df_alert, events = generate_alerts(test_df, alert_threshold=60, minimum_signals=2, consecutive_days=2)
        self.assertEqual(len(events), 1)
        self.assertEqual(df_alert["alert_episode_id"].iloc[1], df_alert["alert_episode_id"].iloc[2])
        self.assertEqual(df_alert["alert_episode_id"].iloc[2], df_alert["alert_episode_id"].iloc[3])
        self.assertTrue(df_alert["is_alert_start"].iloc[1])
        self.assertFalse(df_alert["is_alert_start"].iloc[2])

    def test_13_severity_boundaries(self):
        """Requirement 13: Severity classification maps correctly across all boundaries."""
        self.assertEqual(classify_severity(0.0), "Green")
        self.assertEqual(classify_severity(39.99), "Green")
        self.assertEqual(classify_severity(40.0), "Yellow")
        self.assertEqual(classify_severity(59.99), "Yellow")
        self.assertEqual(classify_severity(60.0), "Orange")
        self.assertEqual(classify_severity(79.99), "Orange")
        self.assertEqual(classify_severity(80.0), "Red")
        self.assertEqual(classify_severity(100.0), "Red")

    def test_14_lead_time_calculation(self):
        """Requirement 14: Lead time is correctly calculated as Official - Alert."""
        df_lead = pd.DataFrame({
            "region": ["Region 3", "Region 3"],
            "date": ["2025-03-12", "2025-03-21"],
            "is_alert_start": [True, False],
            "is_official_report": [False, True],
        })
        summary = calculate_lead_times(df_lead, outbreak_config={"affected_regions": ["Region 3"], "start_date": "2025-03-10"})
        self.assertEqual(summary["lead_days"].iloc[0], 9.0)
        self.assertTrue(summary["detected"].iloc[0])

    def test_15_missing_official_dates_handled(self):
        """Requirement 15: Missing official report dates are represented as None/NaN, not 0."""
        df_missing = pd.DataFrame({
            "region": ["Region 1", "Region 1"],
            "date": ["2025-03-12", "2025-03-21"],
            "is_alert_start": [False, False],
            "is_official_report": [False, False],
        })
        summary = calculate_lead_times(df_missing, outbreak_config={"affected_regions": ["Region 3"], "start_date": "2025-03-10"})
        self.assertIsNone(summary["official_report_date"].iloc[0])
        self.assertTrue(pd.isna(summary["lead_days"].iloc[0]) or summary["lead_days"].iloc[0] is None)

    def test_16_unaffected_regions_false_alarms(self):
        """Requirement 16: Unaffected regions are evaluated for false alarms."""
        df_unaffected = pd.DataFrame({
            "region": ["Region 1"],
            "date": ["2025-02-01"],
            "is_alert_start": [True],
            "is_official_report": [False],
        })
        summary = calculate_lead_times(df_unaffected, outbreak_config={"affected_regions": ["Region 3"], "start_date": "2025-03-10"})
        self.assertFalse(summary["outbreak_region"].iloc[0])
        self.assertFalse(summary["detected"].iloc[0])
        self.assertEqual(summary["false_alarm_count"].iloc[0], 1)

    def test_17_historical_leakage_free(self):
        """Requirement 17: Historical calculations strictly do not use future information."""
        df_leak = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=10, freq="D").strftime("%Y-%m-%d"),
            "region": "Region A",
            "lat": 20.0,
            "lon": 75.0,
            "wastewater": [10.0] * 9 + [1000.0],
            "otc_sales": [20.0] * 10,
            "search_trend": [15.0] * 10,
            "sick_leave": [2.0] * 10,
            "official_cases": [5] * 10,
        })
        df_res = calculate_rolling_baseline(df_leak, window=5, min_periods=2)
        self.assertAlmostEqual(df_res["wastewater_mean"].iloc[4], 10.0, places=2)

if __name__ == "__main__":
    unittest.main()
