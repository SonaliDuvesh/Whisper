"""Whisper — Outbreak Early Warning Dashboard.

Clean Command Center interface for multi-signal epidemiological surveillance,
anomaly detection, risk scoring, and lead-time analytics.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import streamlit as st

from detector import (
    DEFAULT_WEIGHTS,
    LEADING_SIGNALS,
    SEVERITY_COLORS,
    classify_severity,
    normalize_weights,
    run_detection_pipeline,
)
from viz import (
    create_risk_map,
    create_severity_legend,
    prepare_alert_feed,
    render_alert_card,
    create_signal_timeline,
    create_risk_gauge,
    create_signal_contribution_chart,
    prepare_signal_status_table,
    calculate_signal_trend,
    create_replay_trajectory_chart,
    render_replay_timeline_html,
    render_milestone_banner_html,
    create_workforce_map,
    create_workforce_trend_chart,
    create_org_comparison_chart,
    prepare_workforce_alert_feed,
    render_workforce_alert_card,
    create_lead_time_comparison_chart,
    create_historical_risk_progression_chart,
    create_sensitivity_heatmap,
)
from ai_briefing import (
    get_ai_config,
    build_region_context,
    calculate_confidence,
    generate_ai_briefing,
    generate_fallback_briefing,
    answer_region_question,
    build_workforce_context,
    generate_workforce_briefing,
    answer_workforce_question,
)
from workforce_detector import (
    load_workforce_data,
    run_workforce_pipeline,
)
from validation import (
    load_outbreak_metadata,
    evaluate_regional_performance,
    calculate_validation_metrics,
    run_threshold_sensitivity_analysis,
    export_validation_csv,
)
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

st.set_page_config(
    page_title="Whisper - Advanced Epidemiological Signal Detection & Early-Warning Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

COMMAND_CENTER_CSS = """
<style>
    /* Dark Command Center Base Palette */
    :root {
        --bg-main: #0B1220;
        --card-bg: #111B2E;
        --border-subtle: #233149;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --accent-cyan: #06B6D4;
        --sev-green: #22C55E;
        --sev-yellow: #EAB308;
        --sev-orange: #F97316;
        --sev-red: #EF4444;
    }

    /* Overall App Background & Typography */
    .stApp {
        background-color: var(--bg-main);
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid var(--border-subtle);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--text-primary);
    }

    /* Header Component */
    .command-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 18px 24px;
        background: #111B2E;
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        margin-bottom: 24px;
    }
    .header-branding {
        display: flex;
        flex-direction: column;
    }
    .brand-title {
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-subtitle {
        font-size: 13px;
        font-weight: 500;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 2px;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.35);
        color: #22C55E;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .status-pulse {
        width: 8px;
        height: 8px;
        background-color: #22C55E;
        border-radius: 50%;
        box-shadow: 0 0 8px #22C55E;
    }

    /* KPI Cards Grid */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-grid-6 {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 12px;
        margin-bottom: 20px;
    }
    @media (max-width: 1200px) {
        .kpi-grid-6 {
            grid-template-columns: repeat(3, 1fr);
        }
    }
    @media (max-width: 768px) {
        .kpi-grid-6 {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    .kpi-card {
        background: #111B2E;
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 16px 20px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        border-color: rgba(6, 182, 212, 0.5);
    }
    .kpi-label, .kpi-title {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.2;
    }
    .kpi-subtext {
        font-size: 12px;
        color: #64748B;
        margin-top: 6px;
    }
    .severity-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 700;
        margin-left: 6px;
    }
    .pill-green { background: rgba(34, 197, 94, 0.2); color: #22C55E; }
    .pill-yellow { background: rgba(234, 179, 8, 0.2); color: #EAB308; }
    .pill-orange { background: rgba(249, 115, 22, 0.2); color: #F97316; }
    .pill-red {
        background: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        animation: pulse-red-pill 2s infinite ease-in-out;
    }
    @keyframes pulse-red-pill {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 0 8px 1px rgba(239, 68, 68, 0.6); }
    }

    /* Upcoming Feature Placeholder Box */
    .placeholder-panel {
        background: #111B2E;
        border: 1px dashed var(--border-subtle);
        border-radius: 8px;
        padding: 32px 24px;
        text-align: center;
        margin-bottom: 20px;
    }
    .placeholder-title {
        font-size: 16px;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 8px;
    }
    .placeholder-desc {
        font-size: 13px;
        color: #94A3B8;
        max-width: 550px;
        margin: 0 auto;
    }
    .placeholder-badge {
        display: inline-block;
        background: rgba(6, 182, 212, 0.15);
        color: #06B6D4;
        font-size: 11px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 4px;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Streamlit Widget Customization */
    div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid var(--border-subtle);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        color: #94A3B8;
        font-size: 14px;
        font-weight: 600;
        padding: 10px 18px;
    }
    .stTabs [aria-selected="true"] {
        color: #06B6D4 !important;
        border-bottom: 2px solid #06B6D4 !important;
    }
</style>
"""
st.markdown(COMMAND_CENTER_CSS, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data(csv_path: str = "data/signals.csv") -> pd.DataFrame:
    """Load and validate the synthetic surveillance dataset.

    Args:
        csv_path: Path to the dataset CSV file.

    Returns:
        pd.DataFrame: Validated surveillance data sorted by region and date.
    """
    if not os.path.exists(csv_path):
        alt_path = os.path.join(os.path.dirname(__file__), csv_path)
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            try:
                from data_gen import generate_dataset, save_dataset
                df_gen, _ = generate_dataset(random_seed=42)
                save_dataset(df_gen, csv_path)
            except Exception as exc:
                raise FileNotFoundError(
                    f"Surveillance dataset '{csv_path}' not found and auto-generation failed: {exc}. "
                    "Please execute `python data_gen.py` first."
                )

    df = pd.read_csv(csv_path)

    required = ["date", "region", "lat", "lon", "wastewater", "otc_sales", "search_trend", "sick_leave", "official_cases"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Malformed dataset: Missing required columns {missing}")

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(["region", "date"]).reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def get_cached_detection_results(
    df: pd.DataFrame,
    weights_tuple: Tuple[Tuple[str, float], ...],
    z_threshold: float,
    alert_threshold: float,
) -> Tuple[pd.DataFrame, list, pd.DataFrame]:
    """Execute and cache detection pipeline with parameter-hashed memoization.

    Args:
        df: Input raw surveillance DataFrame.
        weights_tuple: Hashable tuple of (signal, weight) pairs.
        z_threshold: Sensitivity threshold for Z-score anomalies.
        alert_threshold: Minimum risk score for consecutive alert qualification.

    Returns:
        tuple: (processed_df, alert_events, validation_summary)
    """
    weights_dict = dict(weights_tuple)
    return run_detection_pipeline(
        df,
        weights=weights_dict,
        z_threshold=z_threshold,
        alert_threshold=alert_threshold,
    )

@st.cache_data(show_spinner=False)
def load_workforce_pipeline_data(csv_path: str = "data/workforce_signals.csv") -> Tuple[pd.DataFrame, List[Any]]:
    """Load and process synthetic workforce dataset with rolling baselines and anomaly detection."""
    if not os.path.exists(csv_path):
        alt_path = os.path.join(os.path.dirname(__file__), csv_path)
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            from workforce_gen import generate_workforce_signals, save_workforce_dataset
            df_gen = generate_workforce_signals()
            save_workforce_dataset(df_gen, csv_path)

    raw_wf = load_workforce_data(csv_path)
    proc_wf, wf_alerts = run_workforce_pipeline(raw_wf)
    return proc_wf, wf_alerts

def initialize_session_state(min_date: str, max_date: str, regions: List[str]) -> None:
    """Initialize application session state variables without overwriting user selections."""
    defaults = {
        "mode": "Live",
        "sim_date": max_date,
        "selected_region": "All Regions",
        "z_threshold": 2.0,
        "alert_threshold": 60.0,
        "weight_wastewater": DEFAULT_WEIGHTS["wastewater"],
        "weight_otc": DEFAULT_WEIGHTS["otc_sales"],
        "weight_search": DEFAULT_WEIGHTS["search_trend"],
        "weight_sick": DEFAULT_WEIGHTS["sick_leave"],
        "show_official": True,
        "active_tab": "Overview",
        "cached_briefings": {},
        "qa_history": [],
        "qa_context_tag": "",
        "replay_playing": False,
        "replay_date": "2025-03-01",
        "replay_speed": "1x",
        "replay_scenario": "Demonstration: March 2025 Outbreak Wave",
        "replay_milestones_seen": set(),
        "selected_workforce_org": "All Organizations",
        "workforce_qa_history": [],
        "cached_workforce_briefings": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def render_header(
    is_operational: bool,
    region_count: int,
    selected_date: str,
    total_observations: int,
) -> None:
    """Render the Clean Command Center top header and operational status badge."""
    status_badge = (
        '<div class="status-badge"><span class="status-pulse"></span><span>OPERATIONAL • 8 REGIONS ACTIVE</span></div>'
        if is_operational
        else '<div class="status-badge" style="background: rgba(239, 68, 68, 0.15); border-color: rgba(239, 68, 68, 0.4); color: #EF4444;"><span class="status-pulse" style="background: #EF4444; box-shadow: 0 0 8px #EF4444;"></span><span>SYSTEM OFFLINE • DATA ERROR</span></div>'
    )

    header_html = (
        '<div class="command-header">'
        '<div class="header-branding">'
        '<div class="brand-title"><span>🛡️ Whisper</span></div>'
        '<div class="brand-subtitle">Advanced Epidemiological Signal Detection & Early-Warning Intelligence</div>'
        '</div>'
        '<div style="display: flex; align-items: center; gap: 16px;">'
        '<div style="text-align: right;">'
        '<div style="font-size: 11px; color: #64748B; font-weight: 600; text-transform: uppercase;">Observation Date</div>'
        f'<div style="font-size: 14px; color: #06B6D4; font-weight: 700; font-family: monospace;">{selected_date}</div>'
        '</div>'
        f'{status_badge}'
        '</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

def render_sidebar(
    dates: List[str],
    regions: List[str],
) -> Dict:
    """Render sidebar controls and return the active configuration dictionary."""
    st.sidebar.markdown("### 🎛️ Command Controls")

    mode = st.sidebar.radio(
        "Surveillance Mode",
        options=["Live", "Replay"],
        index=0 if st.session_state.get("mode", "Live") == "Live" else 1,
        help="Live monitors the latest telemetry; Replay enables day-by-day historical outbreak simulation.",
    )
    st.session_state["mode"] = mode

    min_date, max_date = dates[0], dates[-1]
    default_date_idx = len(dates) - 1 if mode == "Live" else dates.index(st.session_state.get("sim_date", max_date))

    selected_date = st.sidebar.select_slider(
        "Simulation Observation Date",
        options=dates,
        value=dates[default_date_idx],
        help="Filter telemetry and calculate active signals up to this calendar day.",
    )
    st.session_state["sim_date"] = selected_date

    region_options = ["All Regions"] + regions
    curr_region = st.session_state.get("selected_region", "All Regions")
    curr_reg_idx = region_options.index(curr_region) if curr_region in region_options else 0

    selected_region = st.sidebar.selectbox(
        "Geographical Region",
        options=region_options,
        index=curr_reg_idx,
        help="Select a specific region to inspect its detailed risk metrics.",
    )
    st.session_state["selected_region"] = selected_region

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Detection Sensitivity")

    z_thresh = st.sidebar.slider(
        "Anomaly Threshold (Z-score)",
        min_value=1.5,
        max_value=3.5,
        value=float(st.session_state.get("z_threshold", 2.0)),
        step=0.1,
        help="Number of standard deviations above 28-day historical baseline required to flag an indicator as anomalous.",
    )
    st.session_state["z_threshold"] = z_thresh

    alert_thresh = st.sidebar.slider(
        "Alert Risk Score Threshold",
        min_value=30.0,
        max_value=90.0,
        value=float(st.session_state.get("alert_threshold", 60.0)),
        step=5.0,
        help="Minimum composite risk score (0–100) required for 2 consecutive days to trigger an active outbreak alert.",
    )
    st.session_state["alert_threshold"] = alert_thresh

    with st.sidebar.expander("⚖️ Signal Fusion Weights", expanded=False):
        st.markdown(
            "<span style='font-size: 12px; color: #94A3B8;'>"
            "Adjust each leading indicator's relative contribution to the composite risk score."
            "</span>",
            unsafe_allow_html=True,
        )
        w_ww = st.slider("Wastewater Viral Load", 0.0, 1.0, float(st.session_state.get("weight_wastewater", 0.35)), 0.05)
        w_otc = st.slider("OTC Medicine Sales", 0.0, 1.0, float(st.session_state.get("weight_otc", 0.25)), 0.05)
        w_search = st.slider("Symptom Search Trends", 0.0, 1.0, float(st.session_state.get("weight_search", 0.20)), 0.05)
        w_sick = st.slider("Workforce Sick Leave", 0.0, 1.0, float(st.session_state.get("weight_sick", 0.20)), 0.05)

        raw_sum = w_ww + w_otc + w_search + w_sick
        st.session_state["weight_wastewater"] = w_ww
        st.session_state["weight_otc"] = w_otc
        st.session_state["weight_search"] = w_search
        st.session_state["weight_sick"] = w_sick

        try:
            norm_weights = normalize_weights({
                "wastewater": w_ww,
                "otc_sales": w_otc,
                "search_trend": w_search,
                "sick_leave": w_sick,
            })
            st.markdown(
                f"<div style='font-size: 11px; color: #22C55E; margin-top: 6px;'>"
                f"✓ Normalized: WW={norm_weights['wastewater']:.2f}, OTC={norm_weights['otc_sales']:.2f}, "
                f"Search={norm_weights['search_trend']:.2f}, Sick={norm_weights['sick_leave']:.2f}"
                f"</div>",
                unsafe_allow_html=True,
            )
        except ValueError as err:
            st.error(str(err))
            norm_weights = DEFAULT_WEIGHTS.copy()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📡 Intelligence Options")

    show_official = st.sidebar.toggle(
        "Show Official Cases Telemetry",
        value=bool(st.session_state.get("show_official", True)),
        help="Overlay official hospital and clinical reporting counts alongside leading indicators.",
    )
    st.session_state["show_official"] = show_official

    with st.sidebar.expander("🔑 AI Settings & API Keys", expanded=False):
        st.markdown(
            "<span style='font-size: 12px; color: #94A3B8;'>"
            "Whisper functions 100% offline out-of-the-box. Optionally connect an OpenAI or Anthropic API key."
            "</span>",
            unsafe_allow_html=True,
        )
        ai_provider_options = [
            "Local Offline Engine (Deterministic)",
            "OpenAI (GPT-4o / GPT-4o-mini)",
            "Anthropic (Claude 3.5 Sonnet)",
        ]
        curr_provider_env = os.getenv("AI_PROVIDER", "").strip().lower()
        if curr_provider_env == "openai" or (not curr_provider_env and os.getenv("OPENAI_API_KEY")):
            default_ai_idx = 1
        elif curr_provider_env == "anthropic" or (not curr_provider_env and os.getenv("ANTHROPIC_API_KEY")):
            default_ai_idx = 2
        else:
            default_ai_idx = 0

        ai_choice = st.selectbox(
            "Active Intelligence Engine",
            options=ai_provider_options,
            index=default_ai_idx,
            key="sidebar_ai_provider_choice",
        )

        if "OpenAI" in ai_choice:
            cur_key = os.getenv("OPENAI_API_KEY", "")
            user_openai_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.get("user_openai_key", cur_key),
                type="password",
                help="Enter your sk-... key. Stored only in this local browser session.",
                key="input_openai_key",
            )
            if user_openai_key:
                os.environ["OPENAI_API_KEY"] = user_openai_key
                os.environ["AI_PROVIDER"] = "openai"
                st.session_state["user_openai_key"] = user_openai_key
                st.markdown("<div style='color: #22C55E; font-size: 11px; margin-top: 4px;'>✓ OpenAI Key Active</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color: #F59E0B; font-size: 11px; margin-top: 4px;'>⚠️ Enter key to enable OpenAI</div>", unsafe_allow_html=True)
        elif "Anthropic" in ai_choice:
            cur_key = os.getenv("ANTHROPIC_API_KEY", "")
            user_anthropic_key = st.text_input(
                "Anthropic API Key",
                value=st.session_state.get("user_anthropic_key", cur_key),
                type="password",
                help="Enter your sk-ant-... key. Stored only in this local browser session.",
                key="input_anthropic_key",
            )
            if user_anthropic_key:
                os.environ["ANTHROPIC_API_KEY"] = user_anthropic_key
                os.environ["AI_PROVIDER"] = "anthropic"
                st.session_state["user_anthropic_key"] = user_anthropic_key
                st.markdown("<div style='color: #22C55E; font-size: 11px; margin-top: 4px;'>✓ Anthropic Key Active</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color: #F59E0B; font-size: 11px; margin-top: 4px;'>⚠️ Enter key to enable Anthropic</div>", unsafe_allow_html=True)
        else:
            os.environ["AI_PROVIDER"] = "fallback"
            st.markdown("<div style='color: #06B6D4; font-size: 11px; margin-top: 4px;'>✓ Offline Local Engine Active (Zero API cost)</div>", unsafe_allow_html=True)

    if st.sidebar.button("🤖 Generate AI Briefing", key="sidebar_btn_gen_ai", use_container_width=True):
        st.session_state["trigger_ai_briefing"] = True
        st.session_state["mode"] = "Live"
        st.rerun()

    return {
        "mode": mode,
        "selected_date": selected_date,
        "selected_region": selected_region,
        "z_threshold": z_thresh,
        "alert_threshold": alert_thresh,
        "weights": norm_weights,
        "show_official": show_official,
    }

def render_kpi_cards(
    day_df: pd.DataFrame,
    validation_summary: pd.DataFrame,
    selected_region: str,
) -> None:
    """Render 4 responsive KPI cards across the top of the command center."""
    if selected_region == "All Regions":
        active_alerts_count = int(day_df["is_alert"].sum())
        active_subtext = f"{active_alerts_count} of {len(day_df)} regions under active alert"
    else:
        reg_row = day_df[day_df["region"] == selected_region]
        is_active = bool(reg_row["is_alert"].iloc[0]) if not reg_row.empty else False
        active_alerts_count = 1 if is_active else 0
        active_subtext = "Active outbreak alert in effect" if is_active else "No active alerts in this region"

    if selected_region == "All Regions":
        top_row = day_df.sort_values("risk_score", ascending=False).iloc[0]
        top_name = top_row["region"]
        top_risk = top_row["risk_score"]
        top_sev = top_row["severity"]
    else:
        reg_row = day_df[day_df["region"] == selected_region]
        if not reg_row.empty:
            top_name = selected_region
            top_risk = reg_row["risk_score"].iloc[0]
            top_sev = reg_row["severity"].iloc[0]
        else:
            top_name, top_risk, top_sev = "N/A", 0.0, "Green"

    pill_class = f"pill-{top_sev.lower()}"

    detected_mask = validation_summary["outbreak_region"] & validation_summary["detected"]
    positive_leads = validation_summary.loc[detected_mask, "lead_days"].dropna()
    positive_leads = positive_leads[positive_leads > 0]

    if not positive_leads.empty:
        avg_lead = float(positive_leads.mean())
        lead_display = f"{avg_lead:.1f} Days"
        lead_subtext = f"Evaluated across {len(positive_leads)} detected outbreak zones"
    else:
        lead_display = "N/A"
        lead_subtext = "No positive lead times recorded yet"

    if selected_region == "All Regions":
        total_possible = len(day_df) * 4
        anom_count = int(day_df["anomalous_signals_count"].sum())
        anom_display = f"{anom_count} / {total_possible}"
        anom_subtext = f"Z > threshold across 32 regional channels"
    else:
        reg_row = day_df[day_df["region"] == selected_region]
        anom_count = int(reg_row["anomalous_signals_count"].iloc[0]) if not reg_row.empty else 0
        anom_display = f"{anom_count} / 4"
        anom_subtext = f"Anomalous leading signals in {selected_region}"

    kpi_html = f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Active Outbreak Alerts</div>
            <div class="kpi-value" style="color: {'#EF4444' if active_alerts_count > 0 else '#F8FAFC'};">
                {active_alerts_count}
            </div>
            <div class="kpi-subtext">{active_subtext}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Highest Risk Region</div>
            <div class="kpi-value" style="font-size: 24px; display: flex; align-items: center; justify-content: space-between;">
                <span>{top_name}</span>
                <span class="severity-pill {pill_class}">{top_sev}</span>
            </div>
            <div class="kpi-subtext">Composite Risk Score: <strong style="color: #F8FAFC;">{top_risk:.1f}/100</strong></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Average Early Detection Lead Time</div>
            <div class="kpi-value" style="color: #06B6D4;">
                {lead_display}
            </div>
            <div class="kpi-subtext">{lead_subtext}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Signals Currently Anomalous</div>
            <div class="kpi-value">
                {anom_display}
            </div>
            <div class="kpi-subtext">{anom_subtext}</div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

def render_regional_intelligence(
    processed_df: pd.DataFrame,
    validation_summary: pd.DataFrame,
    selected_region: str,
    selected_date: str,
    weights: Dict[str, float],
    z_threshold: float,
    show_official_cases: bool,
    regions: List[str],
) -> None:
    """Render comprehensive regional analytics, timeseries, gauge, and signal decomposition."""
    st.markdown(
        """
        <div style="margin-top: 36px; margin-bottom: 18px; border-top: 1px solid #233149; padding-top: 24px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div>
                    <span style="font-size: 20px; font-weight: 700; color: #F8FAFC;">📡 Regional Intelligence</span>
                    <div style="font-size: 13px; color: #94A3B8;">Signal-level analysis, multi-channel decomposition, and outbreak detection timeline</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if selected_region == "All Regions":
        st.markdown(
            """
            <div class="placeholder-panel" style="padding: 24px 20px; text-align: left; background: #111B2E; border: 1px solid #233149; margin-bottom: 24px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
                    <div>
                        <div style="font-size: 15px; font-weight: 700; color: #F8FAFC;">🔍 Select an Individual Region for Deep Telemetry</div>
                        <div style="font-size: 13px; color: #94A3B8; margin-top: 4px;">
                            Inspect detailed timeseries deviations, 28-day baselines, risk contribution breakdown, and detection lead times.
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<span style='font-size: 12px; color: #64748B; font-weight: 600;'>QUICK FOCUS:</span>", unsafe_allow_html=True)
        cols = st.columns(8)
        for i, reg in enumerate(regions):
            if cols[i].button(reg, key=f"quick_pick_{reg}", use_container_width=True):
                st.session_state["selected_region"] = reg
                st.rerun()
        return

    reg_df = processed_df[processed_df["region"] == selected_region].sort_values("date").reset_index(drop=True)
    current_slice = reg_df[reg_df["date"] == selected_date]

    if current_slice.empty:
        st.warning(f"No observation data found for {selected_region} on {selected_date}.")
        return

    curr_row = current_slice.iloc[0]
    risk_score = float(curr_row["risk_score"])
    severity = str(curr_row["severity"])
    pill_class = f"pill-{severity.lower()}"
    anom_count = int(curr_row["anomalous_signals_count"])
    is_alert = bool(curr_row["is_alert"])

    reg_val = validation_summary[validation_summary["region"] == selected_region]
    alert_date = reg_val["first_alert_date"].iloc[0] if not reg_val.empty else None
    if pd.isna(alert_date):
        alert_date = None
    official_date = reg_val["official_report_date"].iloc[0] if not reg_val.empty else None
    if pd.isna(official_date):
        official_date = None
    lead_days = reg_val["lead_days"].iloc[0] if not reg_val.empty else None

    if lead_days is not None and not pd.isna(lead_days) and lead_days > 0:
        lead_display = f"{lead_days:.0f} Days Earlier"
    else:
        lead_display = "N/A"

    summary_html = f"""
    <div style="background: #111B2E; border: 1px solid #233149; border-radius: 8px; padding: 18px 22px; margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
            <div>
                <span style="font-size: 20px; font-weight: 700; color: #F8FAFC;">{selected_region}</span>
                <span class="severity-pill {pill_class}" style="font-size: 13px; padding: 3px 10px;">{severity}</span>
                <span style="margin-left: 10px; font-size: 12px; color: #94A3B8;">Surveillance Status:</span>
                <strong style="color: {'#EF4444' if is_alert else '#22C55E'}; font-size: 13px; margin-left: 4px;">
                    {'ACTIVE OUTBREAK ALERT' if is_alert else 'NORMAL BASELINE'}
                </strong>
            </div>
            <div style="display: flex; gap: 24px; font-size: 13px;">
                <div>
                    <span style="color: #64748B;">Whisper Alert:</span>
                    <strong style="color: #06B6D4; font-family: monospace; margin-left: 4px;">{alert_date or 'N/A'}</strong>
                </div>
                <div>
                    <span style="color: #64748B;">Official Report:</span>
                    <strong style="color: #22C55E; font-family: monospace; margin-left: 4px;">{official_date or 'N/A'}</strong>
                </div>
                <div>
                    <span style="color: #64748B;">Early Detection Lead:</span>
                    <strong style="color: #F8FAFC; margin-left: 4px;">{lead_display}</strong>
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)

    col_gauge, col_contrib = st.columns([1, 2], gap="large")

    with col_gauge:
        gauge_fig = create_risk_gauge(risk_score, severity, height=250)
        st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})

    with col_contrib:
        st.markdown(
            """
            <div style="font-size: 14px; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">
                Signal Contribution to Regional Risk (0–100)
            </div>
            <div style="font-size: 12px; color: #94A3B8; margin-bottom: 8px;">
                Breakdown of weighted leading indicator contributions summing to overall risk score.
            </div>
            """,
            unsafe_allow_html=True,
        )
        contrib_fig = create_signal_contribution_chart(curr_row, weights, height=220)
        st.plotly_chart(contrib_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    tc1, tc2 = st.columns([3, 1])
    with tc1:
        st.markdown(
            f"""
            <div style="font-size: 16px; font-weight: 700; color: #F8FAFC;">
                📈 Signal Behaviour & Outbreak Timeline ({selected_region})
            </div>
            <div style="font-size: 12px; color: #94A3B8;">
                Historical Z-scores up to {selected_date}. Anomaly reference line (+{z_threshold:.1f}σ) and alert milestones.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with tc2:
        window_choice = st.selectbox(
            "Historical Window",
            options=["30 Days", "60 Days", "90 Days", "Full Timeline"],
            index=1,
            label_visibility="collapsed",
        )
        history_days = int(window_choice.split()[0]) if "Days" in window_choice else "Full"

    timeline_fig = create_signal_timeline(
        region_df=reg_df,
        selected_date=selected_date,
        alert_date=alert_date,
        official_date=official_date,
        z_threshold=z_threshold,
        show_official_cases=show_official_cases,
        history_days=history_days,
        height=420,
    )
    st.plotly_chart(timeline_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    st_col1, st_col2 = st.columns([3, 2], gap="large")

    with st_col1:
        st.markdown(
            """
            <div style="font-size: 15px; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;">
                📋 Channel Telemetry & Anomaly Status
            </div>
            """,
            unsafe_allow_html=True,
        )
        status_table = prepare_signal_status_table(curr_row, weights, z_threshold)
        st.dataframe(status_table, use_container_width=True, hide_index=True)

    with st_col2:
        st.markdown(
            """
            <div style="font-size: 15px; font-weight: 700; color: #F8FAFC; margin-bottom: 6px;">
                💡 Historical Signal Insights
            </div>
            """,
            unsafe_allow_html=True,
        )
        insights = calculate_signal_trend(reg_df, selected_date, z_threshold)
        delta_sign = "+" if insights["trend_delta"] >= 0 else ""
        delta_color = "#EF4444" if insights["trend_direction"] == "Increasing" else ("#22C55E" if insights["trend_direction"] == "Decreasing" else "#94A3B8")

        insights_html = f"""
        <div style="background: #111B2E; border: 1px solid #233149; border-radius: 6px; padding: 14px 18px; font-size: 13px;">
            <div style="margin-bottom: 10px;">
                <span style="color: #64748B;">7-Day Risk Trajectory:</span><br>
                <strong style="color: {delta_color}; font-size: 14px;">{insights['trend_direction']} ({delta_sign}{insights['trend_delta']:.1f} pts)</strong>
            </div>
            <div style="margin-bottom: 10px;">
                <span style="color: #64748B;">Primary Risk Driver:</span><br>
                <strong style="color: #F8FAFC;">{insights['top_driver']}</strong>
            </div>
            <div style="margin-bottom: 10px;">
                <span style="color: #64748B;">First Leading Indicator Onset:</span><br>
                <strong style="color: #06B6D4;">{insights['first_anomalous_signal']}</strong>
            </div>
            <div>
                <span style="color: #64748B;">Anomalous Leading Channels:</span><br>
                <strong style="color: #F8FAFC;">{insights['anom_count']} of 4 indicators above threshold</strong>
            </div>
        </div>
        """
        st.markdown(insights_html, unsafe_allow_html=True)

def render_ai_briefing_section(
    processed_df: pd.DataFrame,
    validation_summary: pd.DataFrame,
    alert_events: List[Any],
    selected_region: str,
    selected_date: str,
    weights: Dict[str, float],
    z_threshold: float,
) -> None:
    """Render evidence-based AI briefing, confidence methodology, and interactive Q&A assistant."""
    st.markdown(
        """
        <div style="margin-top: 36px; margin-bottom: 18px; border-top: 1px solid #233149; padding-top: 24px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div>
                    <span style="font-size: 20px; font-weight: 700; color: #F8FAFC;">🤖 AI Public Health Briefing & Contextual Intelligence</span>
                    <div style="font-size: 13px; color: #94A3B8;">Multi-signal epidemiological synthesis and interactive Q&A assistant</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if selected_region == "All Regions":
        st.info("💡 Select an individual region from the map or quick-pick buttons above to generate an AI briefing.")
        return

    reg_df = processed_df[processed_df["region"] == selected_region].sort_values("date").reset_index(drop=True)
    current_slice = reg_df[reg_df["date"] == selected_date]
    if current_slice.empty:
        st.warning(f"No observation data found for {selected_region} on {selected_date}.")
        return

    curr_row = current_slice.iloc[0]
    risk_score = float(curr_row["risk_score"])
    severity = str(curr_row["severity"])

    context = build_region_context(
        region=selected_region,
        date=selected_date,
        risk_score=risk_score,
        severity=severity,
        curr_row=curr_row,
        reg_df=reg_df,
        weights=weights,
        z_threshold=z_threshold,
        validation_summary=validation_summary,
        alert_events=alert_events,
    )

    weights_tuple = tuple(sorted((k, round(v, 2)) for k, v in weights.items()))
    cache_key = f"{selected_region}_{selected_date}_{round(z_threshold, 1)}_{weights_tuple}"

    ai_cfg = get_ai_config()
    provider_label = {
        "openai": f"OpenAI ({ai_cfg['openai_model']})",
        "anthropic": f"Anthropic ({ai_cfg['anthropic_model']})",
        "fallback": "Local Deterministic Engine (Offline)",
    }.get(ai_cfg["active_provider"], "Local Engine")

    trigger_sidebar = st.session_state.pop("trigger_ai_briefing", False)
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        generate_clicked = st.button("🤖 Generate AI Briefing", key="btn_gen_ai_briefing", type="primary", use_container_width=True) or trigger_sidebar
    with col_info:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; height: 100%; gap: 10px; font-size: 12px; color: #94A3B8;">
                <span>Active Provider:</span>
                <span style="background: #111B2E; border: 1px solid #233149; padding: 4px 10px; border-radius: 4px; color: #06B6D4; font-weight: 600;">
                    {provider_label}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if generate_clicked:
        with st.spinner("Synthesizing multi-signal telemetry and generating public health briefing..."):
            briefing_result = generate_ai_briefing(context)
            if "cached_briefings" not in st.session_state:
                st.session_state["cached_briefings"] = {}
            st.session_state["cached_briefings"][cache_key] = briefing_result

    cached_briefing = st.session_state.get("cached_briefings", {}).get(cache_key)
    if cached_briefing:
        conf_color = "#22C55E" if cached_briefing["confidence_level"] == "High" else ("#EAB308" if cached_briefing["confidence_level"] == "Moderate" else "#94A3B8")
        briefing_html = f"""
        <div style="background: #111B2E; border: 1px solid #233149; border-radius: 8px; padding: 20px 24px; margin-top: 14px; margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-wrap: wrap; gap: 8px;">
                <div style="font-size: 15px; font-weight: 700; color: #F8FAFC;">
                    📑 Epidemiological Intelligence Briefing — {selected_region}
                </div>
                <div style="display: flex; gap: 8px; align-items: center;">
                    <span style="background: rgba(6, 182, 212, 0.12); color: #06B6D4; border: 1px solid rgba(6, 182, 212, 0.3); font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 4px;">
                        {cached_briefing['provider_mode']}
                    </span>
                    <span style="background: rgba(34, 197, 94, 0.12); color: {conf_color}; border: 1px solid {conf_color}; font-size: 11px; font-weight: 600; padding: 3px 8px; border-radius: 4px;">
                        Confidence: {cached_briefing['confidence_level']}
                    </span>
                </div>
            </div>
            <div style="margin-bottom: 14px; border-left: 3px solid #06B6D4; padding-left: 14px;">
                <div style="font-size: 12px; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px;">1. Situation Summary</div>
                <div style="font-size: 14px; color: #F8FAFC; line-height: 1.6; margin-top: 4px;">{cached_briefing['situation']}</div>
            </div>
            <div style="margin-bottom: 14px; border-left: 3px solid #38BDF8; padding-left: 14px;">
                <div style="font-size: 12px; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px;">2. Supporting Signals & Evidence</div>
                <div style="font-size: 14px; color: #F8FAFC; line-height: 1.6; margin-top: 4px;">{cached_briefing['supporting_signals']}</div>
            </div>
            <div style="margin-bottom: 14px; border-left: 3px solid #EAB308; padding-left: 14px;">
                <div style="font-size: 12px; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px;">3. Recommended Precautionary Actions</div>
                <div style="font-size: 14px; color: #F8FAFC; line-height: 1.6; margin-top: 4px;">{cached_briefing['recommended_action']}</div>
            </div>
            <div style="margin-top: 16px; padding-top: 12px; border-top: 1px solid #1E293B; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; font-size: 12px; color: #64748B;">
                <div><strong>Confidence Basis:</strong> {cached_briefing['confidence_rationale']}</div>
                <div><em>⚠️ Synthetic epidemiological surveillance demonstration. Not intended for individual clinical diagnosis.</em></div>
            </div>
        </div>
        """
        st.markdown(briefing_html, unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
            <div class="placeholder-panel" style="padding: 18px 20px; text-align: left; background: #111B2E; border: 1px solid #233149; margin-top: 12px; margin-bottom: 24px;">
                <div style="font-size: 13px; color: #94A3B8;">
                    Click <strong style="color: #06B6D4;">Generate AI Briefing</strong> above to synthesize multi-signal telemetry for {selected_region} on {selected_date}.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 14px; margin-bottom: 8px;">
            <span style="font-size: 14px; font-weight: 700; color: #F8FAFC;">📥 Regional Outbreak Intelligence Exports</span>
            <span style="font-size: 11px; color: #94A3B8;">Publication-ready PDF & UTF-8 TXT</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_dl1, col_dl2 = st.columns(2)
    reg_report = build_regional_report(
        region_context=context,
        ai_briefing=cached_briefing,
        validation_summary=validation_summary,
    )
    safe_region = sanitize_filename(selected_region)
    safe_date = sanitize_filename(selected_date)

    with col_dl1:
        pdf_bytes = export_report_pdf(reg_report)
        st.download_button(
            label="📑 Download Intelligence Report (PDF)",
            data=pdf_bytes,
            file_name=f"Whisper_{safe_region}_{safe_date}.pdf",
            mime="application/pdf",
            key=f"dl_pdf_{selected_region}_{selected_date}",
            use_container_width=True,
        )
    with col_dl2:
        txt_str = export_report_txt(reg_report)
        st.download_button(
            label="📄 Download Intelligence Report (TXT)",
            data=txt_str,
            file_name=f"Whisper_{safe_region}_{safe_date}.txt",
            mime="text/plain",
            key=f"dl_txt_{selected_region}_{selected_date}",
            use_container_width=True,
        )

    st.markdown(
        """
        <div style="font-size: 16px; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">
            💬 Ask Whisper AI
        </div>
        <div style="font-size: 12px; color: #94A3B8; margin-bottom: 12px;">
            Ask questions about this region's outbreak signals, risk drivers, or detection lead times.
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_tag = f"{selected_region}@{selected_date}"
    if st.session_state.get("qa_context_tag") != current_tag:
        st.session_state["qa_context_tag"] = current_tag
        st.session_state["qa_history"] = []

    st.markdown("<span style='font-size: 11px; color: #64748B; font-weight: 600;'>QUICK QUESTIONS:</span>", unsafe_allow_html=True)
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    quick_q = None
    if q_col1.button(f"Why is {selected_region} {severity}?", key="btn_q_why", use_container_width=True):
        quick_q = f"Why is {selected_region} classified as {severity}?"
    if q_col2.button("Which signal contributes most?", key="btn_q_top", use_container_width=True):
        quick_q = "Which signal contributes most to the current risk score?"
    if q_col3.button("Why did wastewater rise first?", key="btn_q_ww", use_container_width=True):
        quick_q = "Why did wastewater rise before official clinical cases?"
    if q_col4.button("Has official threshold crossed?", key="btn_q_off", use_container_width=True):
        quick_q = "Has the official clinical reporting threshold been crossed?"

    qa_history = st.session_state.get("qa_history", [])
    if qa_history:
        for turn in qa_history:
            if turn["role"] == "user":
                st.markdown(
                    f"""
                    <div style="background: #1E293B; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; margin-left: 40px; color: #F8FAFC; font-size: 13px;">
                        <strong style="color: #38BDF8;">You:</strong> {turn['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                mode_tag = turn.get("mode", "Whisper AI")
                st.markdown(
                    f"""
                    <div style="background: #111B2E; border: 1px solid #233149; border-left: 3px solid #06B6D4; border-radius: 6px; padding: 12px 16px; margin-bottom: 12px; margin-right: 40px; color: #F8FAFC; font-size: 13px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <strong style="color: #06B6D4;">🛡️ Whisper AI:</strong>
                            <span style="font-size: 11px; color: #64748B;">{mode_tag}</span>
                        </div>
                        <div style="line-height: 1.5; color: #E2E8F0;">{turn['content']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with st.form("qa_form", clear_on_submit=True):
        col_in, col_ask = st.columns([5, 1])
        with col_in:
            user_question = st.text_input(
                "Ask question",
                value=quick_q or "",
                placeholder=f"e.g., Why did wastewater rise before official cases in {selected_region}?",
                label_visibility="collapsed",
            )
        with col_ask:
            submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

    c_left, c_right = st.columns([4, 1])
    with c_right:
        if st.button("Clear Conversation", key="btn_clear_qa", use_container_width=True):
            st.session_state["qa_history"] = []
            st.rerun()

    query_to_run = quick_q if quick_q else (user_question if submitted and user_question.strip() else None)
    if query_to_run:
        with st.spinner("Analyzing question with regional surveillance context..."):
            resp = answer_region_question(
                question=query_to_run,
                context=context,
                conversation_history=st.session_state.get("qa_history", []),
            )
            st.session_state["qa_history"].append({"role": "user", "content": query_to_run})
            st.session_state["qa_history"].append({
                "role": "assistant",
                "content": resp["answer"],
                "mode": resp.get("mode", "Whisper Intelligence"),
            })
            st.rerun()

def render_replay_simulator(
    processed_df: pd.DataFrame,
    validation_summary: pd.DataFrame,
    alert_events: List[Any],
    all_dates: List[str],
    regions: List[str],
    weights: Dict[str, float],
    z_threshold: float,
    alert_threshold: float = 60.0,
) -> None:
    """Render interactive outbreak replay simulator, timeline milestones, and telemetry synchronization."""
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div>
                    <span style="font-size: 22px; font-weight: 700; color: #F8FAFC;">⏪ Outbreak Replay Simulator</span>
                    <div style="font-size: 13px; color: #94A3B8;">Watch how Whisper identifies emerging health risks before official reporting.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_scen, col_spd = st.columns([3, 1])
    with col_scen:
        scenario = st.selectbox(
            "Replay Scenario",
            options=[
                "Demonstration: March 2025 Outbreak Wave (Region 3, 4, 5)",
                "Full Historical Dataset (2024–2025)",
            ],
            index=0,
            key="replay_scenario_select",
            help="Select the historical replay scenario window.",
        )
    with col_spd:
        speed = st.selectbox(
            "Playback Speed",
            options=["0.5x", "1x", "2x", "5x", "10x"],
            index=1,
            key="replay_speed_select",
        )

    if "March 2025" in scenario:
        scenario_dates = [d for d in all_dates if "2025-03-01" <= d <= "2025-04-05"]
    else:
        scenario_dates = all_dates

    current_replay_date = st.session_state.get("replay_date", scenario_dates[0])
    if current_replay_date not in scenario_dates:
        current_replay_date = scenario_dates[0]
        st.session_state["replay_date"] = current_replay_date

    speed_intervals = {
        "0.5x": 1.2,
        "1x": 0.7,
        "2x": 0.35,
        "5x": 0.15,
        "10x": 0.05,
    }
    sleep_delay = speed_intervals.get(speed, 0.7)

    b1, b2, b3, b4, b5, b6 = st.columns([2, 1.5, 1.5, 1.5, 2.5, 2])
    is_playing = st.session_state.get("replay_playing", False)
    curr_idx = scenario_dates.index(current_replay_date)

    with b1:
        if not is_playing:
            if st.button("▶ Play Replay", key="btn_play_replay", type="primary", use_container_width=True):
                st.session_state["replay_playing"] = True
                st.rerun()
        else:
            if st.button("⏸ Pause Replay", key="btn_pause_replay", type="primary", use_container_width=True):
                st.session_state["replay_playing"] = False
                st.rerun()

    with b2:
        if st.button("↺ Restart", key="btn_restart_replay", use_container_width=True):
            st.session_state["replay_playing"] = False
            st.session_state["replay_date"] = scenario_dates[0]
            st.session_state["replay_milestones_seen"] = set()
            st.rerun()

    with b3:
        prev_disabled = (curr_idx == 0)
        if st.button("◀ Prev Day", key="btn_prev_replay", disabled=prev_disabled, use_container_width=True):
            st.session_state["replay_playing"] = False
            st.session_state["replay_date"] = scenario_dates[curr_idx - 1]
            st.rerun()

    with b4:
        next_disabled = (curr_idx == len(scenario_dates) - 1)
        if st.button("Next Day ▶", key="btn_next_replay", disabled=next_disabled, use_container_width=True):
            st.session_state["replay_playing"] = False
            st.session_state["replay_date"] = scenario_dates[curr_idx + 1]
            st.rerun()

    with b5:
        if st.button("⚡ Jump to Outbreak (Mar 10)", key="btn_jump_outbreak", use_container_width=True):
            st.session_state["replay_playing"] = False
            if "2025-03-10" in scenario_dates:
                st.session_state["replay_date"] = "2025-03-10"
            st.rerun()

    with b6:
        st.markdown(
            f"""
            <div style="background: #111B2E; border: 1px solid #233149; border-radius: 6px; padding: 7px 10px; text-align: center;">
                <span style="font-size: 11px; color: #64748B; font-weight: 600;">SIMULATION DATE</span><br>
                <span style="font-size: 13px; font-weight: 700; color: #06B6D4; font-family: monospace;">{current_replay_date}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    slider_date = st.select_slider(
        "Navigate Replay Timeline",
        options=scenario_dates,
        value=current_replay_date,
        key="replay_slider_date",
        label_visibility="collapsed",
    )
    if slider_date != current_replay_date:
        st.session_state["replay_date"] = slider_date
        st.session_state["replay_playing"] = False
        st.rerun()

    r3_val = validation_summary[validation_summary["region"] == "Region 3"]
    first_alert_date = r3_val["first_alert_date"].iloc[0] if not r3_val.empty else "2025-03-13"
    official_report_date = r3_val["official_report_date"].iloc[0] if not r3_val.empty else "2025-03-21"
    lead_days = int(r3_val["lead_days"].iloc[0]) if not r3_val.empty and not pd.isna(r3_val["lead_days"].iloc[0]) else 8

    timeline_html = render_replay_timeline_html(
        current_date=current_replay_date,
        onset_date="2025-03-10",
        anomaly_date="2025-03-11",
        alert_date=first_alert_date,
        official_date=official_report_date,
        lead_days=lead_days,
    )
    st.markdown(timeline_html, unsafe_allow_html=True)

    milestone_banner = render_milestone_banner_html(
        current_date=current_replay_date,
        region="Region 3 (Epicenter)",
        alert_date=first_alert_date,
        official_date=official_report_date,
        lead_days=lead_days,
        initial_risk=78.4,
    )
    if milestone_banner:
        st.markdown(milestone_banner, unsafe_allow_html=True)

    day_df = processed_df[processed_df["date"] == current_replay_date]
    active_alerts_cnt = int(day_df["is_alert"].sum()) if "is_alert" in day_df.columns else 0
    anom_sig_sum = int(day_df["anomalous_signals_count"].sum()) if "anomalous_signals_count" in day_df.columns else 0

    max_risk_row = day_df.loc[day_df["risk_score"].idxmax()] if not day_df.empty else None
    highest_reg_str = f"{max_risk_row['region']} ({max_risk_row['risk_score']:.1f})" if max_risk_row is not None else "None"
    highest_sev = str(max_risk_row["severity"]).upper() if max_risk_row is not None else "GREEN"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Current Replay Date</div>
                <div class="kpi-value" style="color: #06B6D4; font-size: 22px;">{current_replay_date}</div>
                <div class="kpi-subtext">Day {curr_idx + 1} of {len(scenario_dates)} in scenario</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Active Alert Episodes</div>
                <div class="kpi-value" style="color: {'#EF4444' if active_alerts_cnt > 0 else '#22C55E'};">{active_alerts_cnt}</div>
                <div class="kpi-subtext">Qualifying 2-day consecutive warnings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        sev_color = "#EF4444" if highest_sev == "RED" else ("#F97316" if highest_sev == "ORANGE" else ("#EAB308" if highest_sev == "YELLOW" else "#22C55E"))
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Peak Regional Risk</div>
                <div class="kpi-value" style="font-size: 20px; color: {sev_color};">{highest_reg_str}</div>
                <div class="kpi-subtext">Severity: {highest_sev}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Anomalous Signals</div>
                <div class="kpi-value">{anom_sig_sum}</div>
                <div class="kpi-subtext">Sensors across all 8 regions > +2.0σ</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    map_col, feed_col = st.columns([3, 2], gap="large")

    with map_col:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
                <div>
                    <span style="font-size: 16px; font-weight: 700; color: #F8FAFC;">🗺️ Replay Geospatial Risk Map</span>
                    <div style="font-size: 12px; color: #94A3B8;">Dynamic marker scales and severity colors on {current_replay_date}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        map_fig = create_risk_map(day_df, selected_region="All Regions")
        st.plotly_chart(map_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(create_severity_legend(), unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
        traj_fig = create_replay_trajectory_chart(
            processed_df=processed_df,
            current_date=current_replay_date,
            outbreak_regions=["Region 3", "Region 4", "Region 5"],
            alert_threshold=alert_threshold,
            height=260,
        )
        st.plotly_chart(traj_fig, use_container_width=True, config={"displayModeBar": False})

    with feed_col:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
                <div>
                    <span style="font-size: 16px; font-weight: 700; color: #F8FAFC;">🚨 Replay Alert Feed</span>
                    <div style="font-size: 12px; color: #94A3B8;">Episodes initiated on or before {current_replay_date}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        replay_feed_filter = st.selectbox(
            "Filter Replay Alerts",
            options=["All Alerts", "Active Alerts", "Resolved Alerts", "Red Alerts", "Orange Alerts"],
            index=0,
            key="replay_feed_filter_select",
            label_visibility="collapsed",
        )
        feed_items = prepare_alert_feed(
            alert_events=alert_events,
            processed_df=processed_df,
            selected_date=current_replay_date,
            status_filter=replay_feed_filter,
        )
        feed_container = st.container(height=560)
        with feed_container:
            if not feed_items:
                st.markdown(
                    """
                    <div style="text-align: center; padding: 48px 16px; color: #64748B;">
                        <div style="font-size: 28px; margin-bottom: 8px;">🛡️</div>
                        <div style="font-size: 13px; font-weight: 500;">No qualifying outbreak alert episodes as of this simulation date.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                for item in feed_items:
                    st.markdown(render_alert_card(item), unsafe_allow_html=True)

    if is_playing:
        if curr_idx < len(scenario_dates) - 1:
            time.sleep(sleep_delay)
            st.session_state["replay_date"] = scenario_dates[curr_idx + 1]
            st.rerun()
        else:
            st.session_state["replay_playing"] = False
            st.rerun()

def render_workforce_intelligence(
    selected_date: str,
) -> None:
    """Render the complete Workforce Health Intelligence module in Tab 3."""
    st.markdown(
        """
        <div style="margin-bottom: 18px;">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <div>
                    <span style="font-size: 22px; font-weight: 700; color: #F8FAFC;">🏢 Workforce Health Intelligence</span>
                    <div style="font-size: 13px; color: #94A3B8;">Aggregated workplace health monitoring, operational capacity risk scoring, and site comparison.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background: rgba(6, 182, 212, 0.08); border: 1px solid rgba(6, 182, 212, 0.25); border-radius: 6px; padding: 12px 18px; margin-bottom: 22px; font-size: 12px; color: #94A3B8;">
            🛡️ <strong>Ethical Privacy Guarantee:</strong> All workforce indicators are synthetic, aggregated, and anonymized at the organizational/site level.
            No individual employee health records, personal identifiers, names, or sickness histories are collected, tracked, or processed.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        wf_df, wf_alerts = load_workforce_pipeline_data()
    except Exception as exc:
        st.error(f"Error loading workforce dataset: {exc}")
        return

    all_orgs = sorted(wf_df["organization"].unique().tolist())

    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    with col_f1:
        org_options = ["All Organizations"] + all_orgs
        sel_org = st.selectbox(
            "Select Organization / Site Focus",
            options=org_options,
            index=0,
            key="workforce_org_select",
        )
    with col_f2:
        st.markdown(
            f"""
            <div style="background: #111B2E; border: 1px solid #233149; border-radius: 6px; padding: 7px 12px; margin-top: 24px; text-align: center;">
                <span style="font-size: 11px; color: #64748B; font-weight: 600;">SIMULATION DATE</span><br>
                <span style="font-size: 13px; font-weight: 700; color: #06B6D4; font-family: monospace;">{selected_date}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_f3:
        window_choice = st.selectbox(
            "Historical Trend Window",
            options=["30 Days", "60 Days", "90 Days", "Full Timeline"],
            index=1,
            key="workforce_window_select",
        )
        hist_window = int(window_choice.split()[0]) if "Days" in window_choice else "Full"

    day_df = wf_df[wf_df["date"] == selected_date]
    if day_df.empty:
        st.warning(f"No workforce telemetry available for {selected_date}.")
        return

    if sel_org != "All Organizations":
        focus_df = day_df[day_df["organization"] == sel_org]
    else:
        focus_df = day_df

    tot_workforce = int(focus_df["total_employees"].sum())
    tot_absent = int(focus_df["absent_employees"].sum())
    avail_workforce = tot_workforce - tot_absent
    avg_absence_rate = (tot_absent / tot_workforce * 100.0) if tot_workforce > 0 else 0.0
    avg_capacity = (avail_workforce / tot_workforce * 100.0) if tot_workforce > 0 else 100.0
    max_risk = float(focus_df["workforce_risk"].max()) if not focus_df.empty else 0.0
    highest_sev = str(focus_df.loc[focus_df["workforce_risk"].idxmax()]["severity"]).upper() if not focus_df.empty else "GREEN"
    active_alerts_cnt = int(focus_df["is_alert"].sum()) if "is_alert" in focus_df.columns else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Monitored Workforce</div>
                <div class="kpi-value">{tot_workforce:,}</div>
                <div class="kpi-subtext">{'Across 5 sites' if sel_org == 'All Organizations' else focus_df['site'].iloc[0]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Current Absenteeism</div>
                <div class="kpi-value" style="color: {'#EF4444' if avg_absence_rate > 5.0 else '#F8FAFC'};">{avg_absence_rate:.1f}%</div>
                <div class="kpi-subtext">{tot_absent:,} staff absent today</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        cap_color = "#22C55E" if avg_capacity >= 95.0 else ("#EAB308" if avg_capacity >= 90.0 else "#EF4444")
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Operational Capacity</div>
                <div class="kpi-value" style="color: {cap_color};">{avg_capacity:.1f}%</div>
                <div class="kpi-subtext">{avail_workforce:,} personnel available</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        sev_color = "#EF4444" if highest_sev == "RED" else ("#F97316" if highest_sev == "ORANGE" else ("#EAB308" if highest_sev == "YELLOW" else "#22C55E"))
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Workforce Risk</div>
                <div class="kpi-value" style="color: {sev_color};">{max_risk:.1f} <span style="font-size: 13px;">/ 100</span></div>
                <div class="kpi-subtext">Severity: {highest_sev}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Active Workforce Alerts</div>
                <div class="kpi-value" style="color: {'#EF4444' if active_alerts_cnt > 0 else '#22C55E'};">{active_alerts_cnt}</div>
                <div class="kpi-subtext">Sites with active alerts</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    col_vis, col_side = st.columns([3, 2], gap="large")

    with col_vis:
        st.markdown(
            f"""
            <div style="font-size: 16px; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">
                🗺️ Workplace Sites Risk Map
            </div>
            <div style="font-size: 12px; color: #94A3B8; margin-bottom: 8px;">
                Geographical distribution of synthetic employment facilities color-coded by current workforce risk.
            </div>
            """,
            unsafe_allow_html=True,
        )
        map_fig = create_workforce_map(day_df, selected_org=sel_org, height=360)
        st.plotly_chart(map_fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        target_org = sel_org if sel_org != "All Organizations" else "Factory A"
        target_org_df = wf_df[wf_df["organization"] == target_org].sort_values("date").reset_index(drop=True)
        trend_fig = create_workforce_trend_chart(
            org_df=target_org_df,
            current_date=selected_date,
            selected_org=target_org,
            window=hist_window,
            z_threshold=2.0,
            height=300,
        )
        st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size: 15px; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">
                📊 Organization Risk & Operational Capacity Ranking
            </div>
            """,
            unsafe_allow_html=True,
        )
        comp_fig = create_org_comparison_chart(day_df, height=220)
        st.plotly_chart(comp_fig, use_container_width=True, config={"displayModeBar": False})

    with col_side:
        st.markdown(
            """
            <div style="font-size: 16px; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">
                🚨 Workforce Alert Feed
            </div>
            <div style="font-size: 12px; color: #94A3B8; margin-bottom: 8px;">
                Sites with persistent multi-day sick leave anomalies exceeding +2.0σ.
            </div>
            """,
            unsafe_allow_html=True,
        )
        feed_filter = st.selectbox(
            "Filter Workforce Alerts",
            options=["All Alerts", "Active Alerts", "Resolved Alerts", "Red Alerts", "Orange Alerts"],
            index=0,
            key="workforce_feed_filter_select",
            label_visibility="collapsed",
        )
        wf_feed_items = prepare_workforce_alert_feed(
            alert_events=wf_alerts,
            processed_df=wf_df,
            selected_date=selected_date,
            status_filter=feed_filter,
        )
        if sel_org != "All Organizations":
            wf_feed_items = [it for it in wf_feed_items if it["organization"] == sel_org]

        feed_box = st.container(height=360)
        with feed_box:
            if not wf_feed_items:
                st.markdown(
                    """
                    <div style="text-align: center; padding: 40px 16px; color: #64748B;">
                        <div style="font-size: 26px; margin-bottom: 6px;">🛡️</div>
                        <div style="font-size: 13px; font-weight: 500;">No qualifying workforce alerts as of this simulation date.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                for item in wf_feed_items:
                    st.markdown(render_workforce_alert_card(item), unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size: 16px; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">
                🤖 AI-Assisted HR Action Briefing
            </div>
            """,
            unsafe_allow_html=True,
        )

        focus_org_name = sel_org if sel_org != "All Organizations" else "Factory A"
        f_row = day_df[day_df["organization"] == focus_org_name].iloc[0]
        f_hist_df = wf_df[wf_df["organization"] == focus_org_name]

        wf_ctx = build_workforce_context(
            organization=focus_org_name,
            date=selected_date,
            curr_row=f_row,
            org_df=f_hist_df,
            alert_events=wf_alerts,
        )

        wf_cache_key = f"wf_{focus_org_name}_{selected_date}"
        if st.button("Generate HR Briefing", key=f"btn_gen_wf_briefing_{focus_org_name}", type="primary", use_container_width=True):
            with st.spinner("Generating evidence-based HR recommendations..."):
                wf_briefing_res = generate_workforce_briefing(wf_ctx)
                if "cached_workforce_briefings" not in st.session_state:
                    st.session_state["cached_workforce_briefings"] = {}
                st.session_state["cached_workforce_briefings"][wf_cache_key] = wf_briefing_res

        cached_wf = st.session_state.get("cached_workforce_briefings", {}).get(wf_cache_key)
        if not cached_wf:
            cached_wf = generate_workforce_briefing(wf_ctx)
            if "cached_workforce_briefings" not in st.session_state:
                st.session_state["cached_workforce_briefings"] = {}
            st.session_state["cached_workforce_briefings"][wf_cache_key] = cached_wf

        conf_c = "#22C55E" if cached_wf["confidence_level"] == "High" else ("#EAB308" if cached_wf["confidence_level"] == "Moderate" else "#94A3B8")
        st.markdown(
            f"""
            <div style="background: #111B2E; border: 1px solid #233149; border-radius: 8px; padding: 16px 18px; margin-top: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <strong style="color: #F8FAFC; font-size: 14px;">{cached_wf['organization']} ({cached_wf['site']})</strong>
                    <span style="background: rgba(34, 197, 94, 0.12); color: {conf_c}; border: 1px solid {conf_c}; font-size: 11px; padding: 2px 7px; border-radius: 4px;">
                        Confidence: {cached_wf['confidence_level']}
                    </span>
                </div>
                <div style="font-size: 13px; color: #CBD5E1; line-height: 1.5; margin-bottom: 8px;">
                    {cached_wf['situation']}
                </div>
                <div style="font-size: 12px; color: #94A3B8; margin-bottom: 10px; border-left: 2px solid #06B6D4; padding-left: 10px;">
                    {cached_wf['supporting_signals']}
                </div>
                <div style="background: rgba(6, 182, 212, 0.08); border-left: 3px solid #06B6D4; border-radius: 4px; padding: 10px 12px; font-size: 12px; color: #F8FAFC; line-height: 1.4; margin-bottom: 10px;">
                    {cached_wf['recommended_action']}
                </div>
                <div style="font-size: 11px; color: #64748B; border-top: 1px solid #1E293B; padding-top: 6px;">
                    <em>{cached_wf['privacy_statement']}</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size: 15px; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">
                💬 Ask About Workforce Trends
            </div>
            """,
            unsafe_allow_html=True,
        )

        q_c1, q_c2 = st.columns(2)
        wf_quick_q = None
        if q_c1.button(f"Why is {focus_org_name} at risk?", key="btn_wq1", use_container_width=True):
            wf_quick_q = f"Why is {focus_org_name} showing elevated workforce risk?"
        if q_c2.button("What is operational capacity?", key="btn_wq2", use_container_width=True):
            wf_quick_q = f"What is the current operational capacity for {focus_org_name}?"

        wf_qa_hist = st.session_state.get("workforce_qa_history", [])
        if wf_qa_hist:
            for t in wf_qa_hist[-3:]:
                if t["role"] == "user":
                    st.markdown(
                        f"""
                        <div style="background: #1E293B; border-radius: 6px; padding: 8px 12px; margin-bottom: 6px; color: #F8FAFC; font-size: 12px;">
                            <strong style="color: #38BDF8;">You:</strong> {t['content']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="background: #111B2E; border: 1px solid #233149; border-left: 3px solid #06B6D4; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; color: #F8FAFC; font-size: 12px;">
                            <div style="line-height: 1.4; color: #E2E8F0;">{t['content']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with st.form("wf_qa_form", clear_on_submit=True):
            col_win, col_wask = st.columns([4, 1])
            with col_win:
                wf_user_q = st.text_input(
                    "Ask workforce question",
                    value=wf_quick_q or "",
                    placeholder=f"e.g. What actions should HR consider for {focus_org_name}?",
                    label_visibility="collapsed",
                )
            with col_wask:
                wf_sub = st.form_submit_button("Ask", type="primary", use_container_width=True)

        wf_query = wf_quick_q if wf_quick_q else (wf_user_q if wf_sub and wf_user_q.strip() else None)
        if wf_query:
            with st.spinner("Analyzing workforce context..."):
                wf_ans = answer_workforce_question(wf_query, wf_ctx)
                st.session_state["workforce_qa_history"].append({"role": "user", "content": wf_query})
                st.session_state["workforce_qa_history"].append({"role": "assistant", "content": wf_ans["answer"]})
                st.rerun()

        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 20px; margin-bottom: 8px;">
                <span style="font-size: 14px; font-weight: 700; color: #F8FAFC;">📥 Workforce Health Intelligence Exports</span>
                <span style="font-size: 11px; color: #94A3B8;">Aggregated Enterprise PDF & Anonymized CSV</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_wfdl1, col_wfdl2 = st.columns(2)
        wf_report = build_workforce_report(
            workforce_context=wf_ctx,
            wf_briefing=st.session_state.get("workforce_briefing_cache", {}).get(f"{focus_org_name}_{selected_date}"),
        )
        safe_org = sanitize_filename(focus_org_name)
        safe_wf_date = sanitize_filename(selected_date)

        with col_wfdl1:
            wf_pdf_bytes = export_workforce_pdf(wf_report)
            st.download_button(
                label=f"📑 Download {focus_org_name} Briefing (PDF)",
                data=wf_pdf_bytes,
                file_name=f"Whisper_Workforce_{safe_org}_{safe_wf_date}.pdf",
                mime="application/pdf",
                key=f"dl_wf_pdf_{safe_org}_{safe_wf_date}",
                use_container_width=True,
            )
        with col_wfdl2:
            wf_csv_str = export_workforce_csv(wf_df, org_name=focus_org_name)
            st.download_button(
                label="📊 Download Absence Telemetry (CSV)",
                data=wf_csv_str,
                file_name=f"Whisper_Workforce_{safe_org}_{safe_wf_date}.csv",
                mime="text/csv",
                key=f"dl_wf_csv_{safe_org}_{safe_wf_date}",
                use_container_width=True,
            )

@st.cache_data(show_spinner=False)
def get_cached_sensitivity_analysis(csv_path: str = "data/signals.csv") -> pd.DataFrame:
    """Precompute and cache parameter sensitivity analysis grid across Z and alert thresholds."""
    raw_df = load_data(csv_path)
    return run_threshold_sensitivity_analysis(
        df=raw_df,
        z_thresholds=[1.5, 2.0, 2.5, 3.0],
        alert_thresholds=[40.0, 50.0, 60.0, 70.0, 80.0],
    )

def render_validation_dashboard(
    processed_df: pd.DataFrame,
    alert_events: List[Any],
    raw_df: pd.DataFrame,
) -> None:
    """Render Tab 4: Historical Validation, Backtesting & Detection Performance."""
    st.markdown("## 📈 Detection Validation & Backtesting")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 14px; margin-top: -8px; margin-bottom: 20px;'>"
        "Retrospective evaluation of Whisper's synthetic outbreak detection performance across all 8 monitored regions."
        "</p>",
        unsafe_allow_html=True,
    )

    outbreak_config = load_outbreak_metadata()
    regional_summary = evaluate_regional_performance(processed_df, alert_events, outbreak_config)
    metrics = calculate_validation_metrics(regional_summary, alert_events, outbreak_config)

    kpi_html = f"""
    <div class="kpi-grid-6">
        <div class="kpi-card">
            <div class="kpi-label">Detection Rate</div>
            <div class="kpi-value" style="color: #10B981;">{metrics['detection_rate']:.1f}%</div>
            <div class="kpi-subtext">{metrics['detected_outbreaks']} of {metrics['total_outbreak_regions']} Outbreak Zones</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Early Detection Rate</div>
            <div class="kpi-value" style="color: #06B6D4;">{metrics['early_detection_rate']:.1f}%</div>
            <div class="kpi-subtext">Before Official Clinical Case Report</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Average Lead Time</div>
            <div class="kpi-value" style="color: #38BDF8;">+{metrics['average_lead_time']:.1f}d</div>
            <div class="kpi-subtext">Median: +{metrics['median_lead_time']:.1f} Days Earlier</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">False Alert Episodes</div>
            <div class="kpi-value" style="color: {'#EF4444' if metrics['false_alert_episodes'] > 0 else '#F8FAFC'};">{metrics['false_alert_episodes']}</div>
            <div class="kpi-subtext">{metrics['false_alarm_rate']:.1f}% Episode Proportion</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Precision (PPV)</div>
            <div class="kpi-value" style="color: #F8FAFC;">{metrics['precision']:.3f}</div>
            <div class="kpi-subtext">TP / (TP + FP)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">F1 Score</div>
            <div class="kpi-value" style="color: #F8FAFC;">{metrics['f1_score']:.3f}</div>
            <div class="kpi-subtext">Harmonic Mean (P & R)</div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    st.markdown("### 📋 Regional Detection Performance")
    col_filter, col_dl = st.columns([2, 1])
    with col_filter:
        outcome_options = ["All Regions", "Known Outbreaks Only", "Early Detection", "No Alert"]
        selected_filter = st.selectbox(
            "Filter Performance Table",
            options=outcome_options,
            label_visibility="collapsed",
        )
    with col_dl:
        csv_data = export_validation_csv(regional_summary)
        st.download_button(
            label="📥 Regional Table (CSV)",
            data=csv_data,
            file_name="Whisper_Regional_Validation.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_val_reg_csv",
        )

    col_val1, col_val2 = st.columns(2)
    with col_val1:
        st.download_button(
            label="📊 Download Validation Summary Metrics (CSV)",
            data=export_validation_summary_csv(metrics),
            file_name="Whisper_Validation_Summary_Metrics.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_val_metrics_csv",
        )
    with col_val2:
        val_pdf_bytes = export_validation_pdf(metrics, regional_summary)
        st.download_button(
            label="📑 Download Validation Benchmark Report (PDF)",
            data=val_pdf_bytes,
            file_name="Whisper_Detection_Validation_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="dl_val_pdf",
        )

    display_summary = regional_summary.copy()
    if selected_filter == "Known Outbreaks Only":
        display_summary = display_summary[display_summary["Known Outbreak"] == "Yes"]
    elif selected_filter == "Early Detection":
        display_summary = display_summary[display_summary["Outcome"].str.contains("Early Detection")]
    elif selected_filter == "No Alert":
        display_summary = display_summary[display_summary["Outcome"] == "No Alert"]

    st.dataframe(
        display_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Lead Days": st.column_config.NumberColumn(
                "Lead Days",
                help="Days earlier than official clinical reporting threshold crossing (mu + 2sigma)",
                format="%d days",
            ),
            "Detected": st.column_config.CheckboxColumn("Detected", help="Whether an outbreak was successfully identified"),
        },
    )

    st.markdown("### 📊 Detection Lead Time & Parameter Sensitivity")
    col_chart_left, col_chart_right = st.columns([1, 1])

    with col_chart_left:
        lead_fig = create_lead_time_comparison_chart(regional_summary, height=270)
        st.plotly_chart(lead_fig, use_container_width=True)

    with col_chart_right:
        sens_metric = st.selectbox(
            "Sensitivity Metric Matrix",
            options=["f1_score", "detection_rate", "false_alarms", "avg_lead_time"],
            format_func=lambda x: {
                "f1_score": "F1 Score Matrix",
                "detection_rate": "Detection Rate (%)",
                "false_alarms": "False Alarm Episodes",
                "avg_lead_time": "Average Lead Time (Days)",
            }.get(x, x),
        )
        try:
            sens_df = get_cached_sensitivity_analysis("data/signals.csv")
            heat_fig = create_sensitivity_heatmap(sens_df, metric=sens_metric, height=230)
            st.plotly_chart(heat_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Sensitivity Analysis calculation error: {e}")

    st.markdown("### ⏱️ Historical Risk Progression (Retrospective 2-Year Evaluation)")
    st.markdown(
        "<p style='color: #94A3B8; font-size: 13px; margin-top: -6px; margin-bottom: 12px;'>"
        "Retrospective longitudinal tracking across all 730 days. Compares outbreak epicenter (Region 3) and secondary zones "
        "(Regions 4 & 5) against a stable baseline control (Region 1). Vertical indicators benchmark wave onset (Mar 10), "
        "Whisper consecutive-day alert qualification (Mar 13), and official clinical case reporting (Mar 21)."
        "</p>",
        unsafe_allow_html=True,
    )
    prog_fig = create_historical_risk_progression_chart(processed_df, outbreak_config, height=320)
    st.plotly_chart(prog_fig, use_container_width=True)

    with st.expander("ℹ️ How Whisper Is Evaluated (Validation Methodology & Rigor)", expanded=False):
        st.markdown(
            """
            #### 1. Injected Outbreak Ground Truth
            A synthetic epidemiological outbreak wave was injected starting **March 10, 2025** across three contiguous regions:
            **Region 3** (epicenter), **Region 4** (secondary wave), and **Region 5** (secondary wave). The remaining five regions
            (Regions 1, 2, 6, 7, and 8) serve as baseline unaffected control zones.

            #### 2. Leakage-Free Rolling Baselines
            Detection models compute moving 28-day baseline means and standard deviations strictly on shifted historical data
            (`df['signal'].shift(1).rolling(28)`). Current and future observations are never leaked into historical baseline calculations.

            #### 3. Multi-Signal Alert Qualification Rules
            An active alert episode is triggered only when:
            - **Composite Risk Score** $\\ge 60/100$
            - **At least 2 leading signals** exhibit simultaneous Z-score anomalies ($Z \\ge 2.0$)
            - Anomalous state persists for **at least 2 consecutive days** to suppress transient noise and one-off spikes.

            #### 4. Independent Official Clinical Reporting Threshold
            Official public health reporting dates are calculated independently based on clinical case counts exceeding
            baseline historical averages by $\\mu + 2\\sigma$.

            #### 5. Early-Warning Lead Time Formula
            Lead time is calculated as:
            $$\\text{Lead Time} = \\text{Official Confirmation Date} - \\text{Whisper First Alert Date}$$
            Positive lead times represent actionable days gained for public health interventions before clinical case surveillance detects the wave.

            #### 6. Discrete False Alarm Episode Accounting
            False alarms are measured as distinct **alert episodes** (unbroken runs of consecutive alert days) occurring in unaffected
            control regions or outside the outbreak evaluation window, preventing single persistent events from being inflated into dozens of daily errors.

            ---
            > [!NOTE]
            > **Scientific Limitation & Benchmark Disclaimer:** Because the prototype uses synthetic surveillance data with known injected outbreaks,
            > these metrics demonstrate algorithm behaviour under controlled conditions. They do not establish real-world epidemiological accuracy.
            """
        )

def render_upcoming_placeholders(selected_region: str, selected_date: str) -> None:
    """Render structured placeholder for future prompt integration."""
    pass

def render_data_sources_expander() -> None:
    """Render expandable reference panel detailing data sources and intended simulated lead times."""
    with st.expander("📚 Data Sources & Simulated Signal Lead Times", expanded=False):
        st.markdown(
            """
            | Leading Signal Channel | Surveillance Modality & Role | Synthetic Simulated Lead Time |
            | :--- | :--- | :--- |
            | **Wastewater Viral Shedding** | Molecular pathogen concentration in municipal effluent | **~10 Days Earlier** |
            | **OTC Pharmacy Sales** | Fever, cough & cold relief point-of-sale volume | **~7 Days Earlier** |
            | **Symptom Search Trends** | Digital symptom query index (e.g. fever, chills, cough) | **~6 Days Earlier** |
            | **Workplace Absenteeism** | Aggregated institutional sick leave & absence rates | **~4 Days Earlier** |
            | **Official Clinical Cases** | Confirmed lab diagnostics & hospital reports (Lagging Ground Truth) | **Baseline Anchor (0d)** |

            *Note: These lead times describe the synthetic epidemiological scenario design for controlled algorithmic testing, not guaranteed real-world predictive performance.*
            """
        )

def render_privacy_trust_section() -> None:
    """Render dedicated Privacy, Data Governance & Trust section in accordance with public health ethics."""
    st.markdown(
        """
        <div style="margin-top: 48px; border-top: 1px solid #233149; padding-top: 24px; margin-bottom: 24px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                <span style="font-size: 18px; font-weight: 700; color: #F8FAFC;">🔒 Privacy, Data Governance & Trust</span>
                <span style="font-size: 11px; background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); padding: 3px 10px; border-radius: 4px; font-weight: 600;">100% SYNTHETIC & DE-IDENTIFIED</span>
            </div>
            <div style="background: #111B2E; border: 1px solid #233149; border-left: 4px solid #06B6D4; border-radius: 6px; padding: 14px 18px; margin-bottom: 16px;">
                <div style="font-size: 13px; font-weight: 600; color: #F8FAFC; margin-bottom: 4px;">
                    Public Health Data Protection Commitment:
                </div>
                <div style="font-size: 12px; color: #94A3B8; line-height: 1.5;">
                    <strong>All data used in this prototype is synthetic, aggregated and anonymized. No individual-level health data is collected, stored, or processed.</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ Read Privacy Policy, AI Processing Disclosures & Prototype Limitations", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                #### 🛡️ Data Privacy & Anonymization
                - **Zero PII/PHI:** No patient names, medical records, national IDs, or home addresses are ever collected or stored.
                - **Workforce Safeguards:** Workplace absence data is strictly aggregated at the enterprise and site level. No individual employee health records or absence reasons are processed.
                - **No Diagnostic Inferences:** Whisper does not predict individual health outcomes or diagnose patients.

                #### 🧬 Synthetic Data Generation
                - The 2-year surveillance dataset (5,840 daily regional records, 3,650 workforce records) was generated using reproducible mathematical models with known injected outbreak parameters to test algorithmic detection capabilities under controlled benchmark conditions.
                """
            )
        with col2:
            st.markdown(
                """
                #### 🤖 AI Processing & Third-Party Disclosure
                - When external AI providers (OpenAI / Anthropic) are enabled via API keys, only **aggregated regional indicators** (e.g. risk scores, signal deviations) are submitted as context.
                - External LLM providers process API requests according to their respective API terms and data privacy policies.
                - When no API keys are present, Whisper operates entirely offline using local rule-based intelligence.

                #### ⚖️ Intended Use & Clinical Advisory
                - Whisper is an experimental early-warning decision-support prototype.
                - It is **not** a certified medical device, diagnostic tool, or replacement for official epidemiological surveillance platforms.
                - All scenarios and observations are fictional and must not be used for real-world clinical decision-making.
                """
            )

def render_glossary_expander() -> None:
    """Render comprehensive surveillance terminology & mathematical concepts glossary."""
    with st.expander("📖 Surveillance Terminology & Mathematical Concepts Glossary", expanded=False):
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(
                """
                - **Z-Score (Standardized Deviation):** Number of standard deviations a leading signal deviates above its moving 28-day baseline ($Z = \\frac{x - \\mu}{\\sigma}$). Values $\\ge 2.0\\sigma$ represent anomalous statistical surges ($p < 0.023$).
                - **Composite Risk Score (0–100):** Multi-signal fused index aggregating weighted Z-scores ($R = \\sum w_i \\cdot Z_i \\cdot 20$), bounded to a 0–100 scale representing overall community risk.
                - **Alert Threshold:** Minimum composite risk score (default $\\ge 60/100$) sustained across $\\ge 2$ leading signals for $\\ge 2$ consecutive days required to trigger an active outbreak alert.
                """
            )
        with col_g2:
            st.markdown(
                """
                - **Early-Warning Lead Time:** Actionable window of days gained between Whisper's first qualifying alert and official clinical case threshold crossing ($\\mu + 2\\sigma$).
                - **False Alarm Episode:** An alert episode (unbroken sequence of alert days) occurring in unaffected control regions or outside the designated evaluation window.
                - **Operational Capacity (%):** Institutional workforce availability rate ($100\\% - \\text{Absenteeism}\\%$), measuring real-time staffing resilience during emerging health waves.
                """
            )

def main():
    """Main Streamlit execution function."""
    try:
        raw_df = load_data("data/signals.csv")
    except Exception as exc:
        st.error(f"Data Loading Error: {exc}")
        st.stop()

    dates = sorted(raw_df["date"].unique().tolist())
    regions = sorted(raw_df["region"].unique().tolist())

    initialize_session_state(dates[0], dates[-1], regions)

    config = render_sidebar(dates, regions)

    weights_tuple = tuple(sorted(config["weights"].items()))
    processed_df, alert_events, validation_summary = get_cached_detection_results(
        raw_df,
        weights_tuple=weights_tuple,
        z_threshold=config["z_threshold"],
        alert_threshold=config["alert_threshold"],
    )

    render_header(
        is_operational=True,
        region_count=len(regions),
        selected_date=config["selected_date"],
        total_observations=len(processed_df),
    )

    tabs = st.tabs([
        "📊 Surveillance Overview",
        "⏪ Outbreak Replay Simulation",
        "🏢 Workforce Intelligence",
        "📈 Historical Validation & Backtesting",
    ])

    with tabs[0]:
        day_df = processed_df[processed_df["date"] == config["selected_date"]]

        if day_df.empty:
            st.warning(f"No telemetry observations available for {config['selected_date']}.")
        else:
            render_kpi_cards(day_df, validation_summary, config["selected_region"])

            map_col, feed_col = st.columns([3, 2], gap="large")

            with map_col:
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
                        <div>
                            <span style="font-size: 16px; font-weight: 700; color: #F8FAFC;">🗺️ Regional Outbreak Risk Map</span>
                            <div style="font-size: 12px; color: #94A3B8;">Geographical distribution of detected health anomalies</div>
                        </div>
                        <div style="font-size: 12px; color: #06B6D4; font-family: monospace; font-weight: 600;">
                            {config['selected_date']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                map_fig = create_risk_map(day_df, selected_region=config["selected_region"])
                st.plotly_chart(map_fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown(create_severity_legend(), unsafe_allow_html=True)

                sub_col1, _ = st.columns([2, 1])
                with sub_col1:
                    current_map_reg = config["selected_region"]
                    options = ["All Regions"] + regions
                    idx = options.index(current_map_reg) if current_map_reg in options else 0
                    new_reg = st.selectbox(
                        "Inspect Regional Telemetry (Map Focus)",
                        options=options,
                        index=idx,
                        key="sync_region_select",
                        help="Select a region to focus telemetry and highlight on the geospatial map."
                    )
                    if new_reg != st.session_state["selected_region"]:
                        st.session_state["selected_region"] = new_reg
                        st.rerun()

            with feed_col:
                st.markdown(
                    """
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
                        <div>
                            <span style="font-size: 16px; font-weight: 700; color: #F8FAFC;">🚨 Real-Time Alert Feed</span>
                            <div style="font-size: 12px; color: #94A3B8;">Qualifying multi-signal outbreak episodes</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_ffilt, col_fdl = st.columns([2, 1])
                with col_ffilt:
                    feed_filter = st.selectbox(
                        "Filter Alert Feed",
                        options=["All Alerts", "Active Alerts", "Resolved Alerts", "Red Alerts", "Orange Alerts", "Yellow Alerts"],
                        index=0,
                        label_visibility="collapsed",
                    )
                with col_fdl:
                    st.download_button(
                        label="📥 Alerts (CSV)",
                        data=export_alerts_csv(alert_events, validation_summary),
                        file_name="Whisper_Alert_History.csv",
                        mime="text/csv",
                        key="dl_alert_history_csv",
                        use_container_width=True,
                    )

                feed_items = prepare_alert_feed(
                    alert_events=alert_events,
                    processed_df=processed_df,
                    selected_date=config["selected_date"],
                    status_filter=feed_filter,
                )

                if config["selected_region"] != "All Regions":
                    feed_items = [item for item in feed_items if item["region"] == config["selected_region"]]

                feed_container = st.container(height=520)
                with feed_container:
                    if not feed_items:
                        st.markdown(
                            """
                            <div style="text-align: center; padding: 48px 16px; color: #64748B;">
                                <div style="font-size: 28px; margin-bottom: 8px;">🛡️</div>
                                <div style="font-size: 13px; font-weight: 500;">No qualifying outbreak alerts for the selected period.</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        for item in feed_items:
                            st.markdown(render_alert_card(item), unsafe_allow_html=True)

            render_regional_intelligence(
                processed_df=processed_df,
                validation_summary=validation_summary,
                selected_region=config["selected_region"],
                selected_date=config["selected_date"],
                weights=config["weights"],
                z_threshold=config["z_threshold"],
                show_official_cases=config["show_official"],
                regions=regions,
            )

            render_ai_briefing_section(
                processed_df=processed_df,
                validation_summary=validation_summary,
                alert_events=alert_events,
                selected_region=config["selected_region"],
                selected_date=config["selected_date"],
                weights=config["weights"],
                z_threshold=config["z_threshold"],
            )

    with tabs[1]:
        render_replay_simulator(
            processed_df=processed_df,
            validation_summary=validation_summary,
            alert_events=alert_events,
            all_dates=dates,
            regions=regions,
            weights=config["weights"],
            z_threshold=config["z_threshold"],
            alert_threshold=config["alert_threshold"],
        )

    with tabs[2]:
        render_workforce_intelligence(selected_date=config["selected_date"])

    with tabs[3]:
        render_validation_dashboard(
            processed_df=processed_df,
            alert_events=alert_events,
            raw_df=raw_df,
        )

    render_glossary_expander()
    render_data_sources_expander()
    render_privacy_trust_section()

if __name__ == "__main__":
    main()
