"""Whisper Visualization Module.

Plotly geospatial risk mapping, severity legends, signal decomposition formatting,
and dynamic alert feed generation for the Clean Command Center dashboard.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from detector import LEADING_SIGNALS, SEVERITY_COLORS

SEV_COLOR_MAP = {
    "Green": "#22C55E",
    "Yellow": "#EAB308",
    "Orange": "#F97316",
    "Red": "#EF4444",
    "GREEN": "#22C55E",
    "YELLOW": "#EAB308",
    "ORANGE": "#F97316",
    "RED": "#EF4444",
}

def format_signal_z(val: float) -> str:
    """Format numeric Z-score with sign and sigma symbol."""
    if pd.isna(val):
        return "N/A"
    v = float(val)
    prefix = "+" if v >= 0 else ""
    return f"{prefix}{v:.1f}σ"

def format_signal_explanation(row: pd.Series) -> str:
    """Generate a clean, human-readable summary of abnormal signals for a region-day."""
    anom_items = []
    sig_labels = {
        "wastewater": "Wastewater",
        "otc_sales": "OTC Sales",
        "search_trend": "Search Trends",
        "sick_leave": "Sick Leave",
    }
    for sig, label in sig_labels.items():
        anom_col = f"anomaly_{sig}"
        z_col = f"z_{sig}"
        if anom_col in row and bool(row[anom_col]):
            z_val = row.get(z_col, 0.0)
            anom_items.append(f"{label} ({format_signal_z(z_val)})")

    if not anom_items:
        return "All leading surveillance signals within normal historical baselines."
    return ", ".join(anom_items)

def create_risk_map(
    regional_data: pd.DataFrame,
    selected_region: Optional[str] = None,
    height: int = 520,
) -> go.Figure:
    """Create the interactive Plotly Scattermap displaying regional outbreak risks.

    Markers dynamically scale by risk score and color by severity classification.
    Uses open MapLibre / Carto Darkmatter tiles without requiring an external paid API key.

    Args:
        regional_data: DataFrame filtered to the selected observation date (8 regions).
        selected_region: Name of the currently selected region (for visual highlighting).
        height: Map canvas height in pixels.

    Returns:
        go.Figure: Interactive Plotly map figure.
    """
    df = regional_data.copy()

    sizes = np.clip(16.0 + (df["risk_score"].values / 100.0) * 26.0, 16.0, 42.0)
    colors = [SEV_COLOR_MAP.get(sev, "#22C55E") for sev in df["severity"]]

    custom_data = []
    for _, row in df.iterrows():
        reg = row["region"]
        risk = float(row.get("risk_score", 0.0))
        sev = str(row.get("severity", "Green"))

        ww_z = format_signal_z(row.get("z_wastewater", 0.0))
        otc_z = format_signal_z(row.get("z_otc_sales", 0.0))
        search_z = format_signal_z(row.get("z_search_trend", 0.0))
        sick_z = format_signal_z(row.get("z_sick_leave", 0.0))

        is_alert = bool(row.get("is_alert", False))
        alert_status = "ACTIVE OUTBREAK ALERT" if is_alert else "Normal Baseline"
        obs_date = str(row.get("date", ""))

        custom_data.append([
            reg,
            risk,
            sev,
            ww_z,
            otc_z,
            search_z,
            sick_z,
            alert_status,
            obs_date,
        ])

    hovertemplate = (
        "<b>%{customdata[0]}</b><br><br>"
        "Risk Score: <b>%{customdata[1]:.1f} / 100</b><br>"
        "Severity Level: <b>%{customdata[2]}</b><br><br>"
        "<b>Leading Signals:</b><br>"
        "• Wastewater: %{customdata[3]}<br>"
        "• OTC Sales: %{customdata[4]}<br>"
        "• Search Trends: %{customdata[5]}<br>"
        "• Sick Leave: %{customdata[6]}<br><br>"
        "Surveillance Status: <b>%{customdata[7]}</b><br>"
        "Date: %{customdata[8]}"
        "<extra></extra>"
    )

    if hasattr(go, "Scattermap"):
        scatter_cls = go.Scattermap
        map_layout_name = "map"
    else:
        scatter_cls = go.Scattermapbox
        map_layout_name = "mapbox"

    fig = go.Figure()

    fig.add_trace(
        scatter_cls(
            lat=df["lat"],
            lon=df["lon"],
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.88,
            ),
            text=df["region"],
            textposition="top right",
            textfont=dict(color="#F8FAFC", size=11, family="sans-serif"),
            customdata=custom_data,
            hovertemplate=hovertemplate,
            name="Regions",
        )
    )

    red_df = df[df["severity"] == "Red"]
    if not red_df.empty:
        red_sizes = np.clip(16.0 + (red_df["risk_score"].values / 100.0) * 26.0, 16.0, 42.0) + 16.0
        fig.add_trace(
            scatter_cls(
                lat=red_df["lat"],
                lon=red_df["lon"],
                mode="markers",
                marker=dict(
                    size=red_sizes,
                    color="#EF4444",
                    opacity=0.30,
                ),
                hoverinfo="skip",
                showlegend=False,
                name="Severe Risk Glow",
            )
        )

    if selected_region and selected_region != "All Regions":
        sel_row = df[df["region"] == selected_region]
        if not sel_row.empty:
            sel_size = np.clip(16.0 + (sel_row["risk_score"].iloc[0] / 100.0) * 26.0, 16.0, 42.0) + 14
            fig.add_trace(
                scatter_cls(
                    lat=sel_row["lat"],
                    lon=sel_row["lon"],
                    mode="markers",
                    marker=dict(
                        size=[sel_size],
                        color="#06B6D4",
                        opacity=0.40,
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                    name="Selected Highlight",
                )
            )

    map_dict = dict(
        style="carto-darkmatter",
        center=dict(lat=21.5, lon=80.0),
        zoom=3.6,
    )

    layout_kwargs = {
        map_layout_name: map_dict,
        "margin": dict(l=0, r=0, t=0, b=0),
        "height": height,
        "showlegend": False,
        "paper_bgcolor": "#111B2E",
        "plot_bgcolor": "#111B2E",
    }
    fig.update_layout(**layout_kwargs)

    return fig

def create_severity_legend() -> str:
    """Generate HTML snippet explaining severity classifications and marker scales."""
    legend_html = """
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-top: 10px; font-size: 12px; color: #94A3B8;">
        <div style="display: flex; align-items: center; gap: 16px;">
            <span style="display: flex; align-items: center; gap: 6px;">
                <span style="width: 10px; height: 10px; border-radius: 50%; background: #22C55E;"></span> Green (&lt;40)
            </span>
            <span style="display: flex; align-items: center; gap: 6px;">
                <span style="width: 10px; height: 10px; border-radius: 50%; background: #EAB308;"></span> Yellow (40–59)
            </span>
            <span style="display: flex; align-items: center; gap: 6px;">
                <span style="width: 10px; height: 10px; border-radius: 50%; background: #F97316;"></span> Orange (60–79)
            </span>
            <span style="display: flex; align-items: center; gap: 6px;">
                <span style="width: 10px; height: 10px; border-radius: 50%; background: #EF4444;"></span> Red (80–100)
            </span>
        </div>
        <div style="color: #64748B;">
            Marker size represents composite risk score (0–100)
        </div>
    </div>
    """
    return legend_html

def prepare_alert_feed(
    alert_events: List,
    processed_df: pd.DataFrame,
    selected_date: str,
    status_filter: str = "All Alerts",
) -> List[Dict]:
    """Prepare and filter alert events relative to the simulation observation date.

    Guarantees historical integrity:
    - Alerts initiated AFTER selected_date are excluded (no future information leakage).
    - Episode status (Active vs Resolved) is evaluated strictly on selected_date.
    - Sorted newest initiation date first.

    Args:
        alert_events: Raw list of AlertEvent objects from detection engine.
        processed_df: Processed DataFrame containing daily alert telemetry.
        selected_date: Active simulation date string (YYYY-MM-DD).
        status_filter: Filter option string.

    Returns:
        list of dicts: Processed and filtered alert records ready for rendering.
    """
    sel_dt = pd.to_datetime(selected_date)
    feed_items = []

    current_day_df = processed_df[processed_df["date"] == selected_date]
    active_regions_map = {}
    for _, row in current_day_df.iterrows():
        active_regions_map[row["region"]] = {
            "is_alert": bool(row["is_alert"]),
            "current_risk": float(row["risk_score"]),
            "current_severity": str(row["severity"]),
            "episode_id": row["alert_episode_id"],
        }

    for event in alert_events:
        event_dt = pd.to_datetime(event.first_alert_date)

        if event_dt > sel_dt:
            continue

        reg_info = active_regions_map.get(event.region, {})
        is_currently_active = (
            reg_info.get("is_alert", False) and
            reg_info.get("episode_id") == event.episode_id
        )

        item = {
            "region": event.region,
            "episode_id": event.episode_id,
            "first_alert_date": event.first_alert_date,
            "severity": event.severity,
            "initial_risk_score": event.initial_risk_score,
            "current_risk_score": reg_info.get("current_risk", event.initial_risk_score) if is_currently_active else None,
            "current_severity": reg_info.get("current_severity", event.severity) if is_currently_active else None,
            "anomalous_signals": event.anomalous_signals,
            "is_active": is_currently_active,
            "status_label": "ACTIVE ALERT" if is_currently_active else "RESOLVED",
            "explanation": event.explanation,
        }

        if status_filter == "Active Alerts" and not is_currently_active:
            continue
        elif status_filter == "Resolved Alerts" and is_currently_active:
            continue
        elif status_filter == "Red Alerts" and item["severity"] != "Red":
            continue
        elif status_filter == "Orange Alerts" and item["severity"] != "Orange":
            continue
        elif status_filter == "Yellow Alerts" and item["severity"] != "Yellow":
            continue

        feed_items.append(item)

    feed_items.sort(key=lambda x: x["first_alert_date"], reverse=True)
    return feed_items

def render_alert_card(alert: Dict) -> str:
    """Render a styled HTML card for a single alert episode entry."""
    sev = alert.get("current_severity") if alert["is_active"] else alert["severity"]
    pill_class = f"pill-{sev.lower()}"
    status_color = "#EF4444" if alert["is_active"] else "#64748B"
    status_bg = "rgba(239, 68, 68, 0.15)" if alert["is_active"] else "rgba(100, 116, 139, 0.15)"

    signals_formatted = ", ".join(alert["anomalous_signals"]) if alert["anomalous_signals"] else "Multi-signal anomaly"

    card_html = f"""
    <div style="background: #111B2E; border: 1px solid #233149; border-radius: 6px; padding: 14px 16px; margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
            <div>
                <span style="font-size: 14px; font-weight: 700; color: #F8FAFC;">{alert['region']}</span>
                <span class="severity-pill {pill_class}">{sev}</span>
            </div>
            <span style="background: {status_bg}; color: {status_color}; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; letter-spacing: 0.5px;">
                {alert['status_label']}
            </span>
        </div>
        <div style="font-size: 12px; color: #94A3B8; margin-bottom: 8px;">
            Initiated: <strong style="color: #F8FAFC;">{alert['first_alert_date']}</strong> • Initial Risk: <strong>{alert['initial_risk_score']:.1f}/100</strong>
        </div>
        <div style="font-size: 12px; color: #CBD5E1; background: rgba(15, 23, 42, 0.6); padding: 8px 10px; border-radius: 4px; border-left: 3px solid #06B6D4;">
            {alert['explanation']}
        </div>
    </div>
    """
    return card_html

SIGNAL_PALETTE = {
    "wastewater": "#06B6D4",
    "otc_sales": "#A855F7",
    "search_trend": "#3B82F6",
    "sick_leave": "#F59E0B",
}

SIGNAL_DISPLAY_NAMES = {
    "wastewater": "Wastewater Viral Load",
    "otc_sales": "OTC Medicine Sales",
    "search_trend": "Symptom Search Trends",
    "sick_leave": "Workplace Sick Leave",
}

def create_signal_timeline(
    region_df: pd.DataFrame,
    selected_date: str,
    alert_date: Optional[str] = None,
    official_date: Optional[str] = None,
    z_threshold: float = 2.0,
    show_official_cases: bool = True,
    history_days: Union[int, str] = 60,
    height: int = 420,
) -> go.Figure:
    """Create the primary regional timeseries chart with Z-scores, thresholds, and lead times.

    Strict No-Lookahead Safeguard:
    Never displays data after selected_date.

    Args:
        region_df: Historical telemetry for the selected region.
        selected_date: Current simulation observation date.
        alert_date: Date of first Whisper alert episode start (if known).
        official_date: Date official cases crossed reporting threshold (if known).
        z_threshold: Anomaly sensitivity threshold (default 2.0).
        show_official_cases: Whether to overlay official cases on secondary Y-axis.
        history_days: Window size (30, 60, 90, or 'Full').
        height: Chart height in pixels.

    Returns:
        go.Figure: Interactive dual-axis Plotly figure.
    """
    from plotly.subplots import make_subplots

    df_hist = region_df[region_df["date"] <= selected_date].copy()
    if df_hist.empty:
        df_hist = region_df.head(1).copy()

    if isinstance(history_days, int) and len(df_hist) > history_days:
        df_view = df_hist.tail(history_days).copy()
    else:
        df_view = df_hist.copy()

    has_secondary = show_official_cases and "official_cases" in df_view.columns
    fig = make_subplots(specs=[[{"secondary_y": has_secondary}]])

    for sig in LEADING_SIGNALS:
        z_col = f"z_{sig}"
        if z_col in df_view.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_view["date"],
                    y=df_view[z_col],
                    mode="lines",
                    name=SIGNAL_DISPLAY_NAMES.get(sig, sig),
                    line=dict(color=SIGNAL_PALETTE.get(sig, "#38BDF8"), width=2.2),
                    hovertemplate=f"<b>{SIGNAL_DISPLAY_NAMES.get(sig, sig)}</b>: %{{y:+.2f}}σ<extra></extra>",
                ),
                secondary_y=False,
            )

    fig.add_hline(
        y=0.0,
        line=dict(color="#475569", width=1, dash="dot"),
        secondary_y=False,
    )

    fig.add_hline(
        y=z_threshold,
        line=dict(color="#EF4444", width=1.5, dash="dash"),
        annotation_text=f"Anomaly Threshold (+{z_threshold:.1f}σ)",
        annotation_position="top left",
        annotation_font=dict(color="#EF4444", size=10),
        secondary_y=False,
    )

    min_view_date = df_view["date"].iloc[0]
    max_view_date = df_view["date"].iloc[-1]

    if alert_date and (min_view_date <= alert_date <= selected_date):
        fig.add_vline(
            x=alert_date,
            line=dict(color="#06B6D4", width=2, dash="dashdot"),
            annotation_text=f"Whisper Alert ({alert_date})",
            annotation_position="top left",
            annotation_font=dict(color="#06B6D4", size=11, family="monospace"),
        )

    if official_date and (min_view_date <= official_date <= selected_date):
        fig.add_vline(
            x=official_date,
            line=dict(color="#EF4444", width=2, dash="dashdot"),
            annotation_text=f"Official Report ({official_date})",
            annotation_position="top right",
            annotation_font=dict(color="#EF4444", size=11, family="monospace"),
        )

        if alert_date and (min_view_date <= alert_date <= selected_date):
            dt_alert = pd.to_datetime(alert_date)
            dt_off = pd.to_datetime(official_date)
            lead_days = (dt_off - dt_alert).days
            if lead_days > 0:
                mid_date = (dt_alert + pd.Timedelta(days=lead_days / 2.0)).strftime("%Y-%m-%d")
                fig.add_annotation(
                    x=mid_date,
                    y=z_threshold + 1.2,
                    text=f"★ Detected {lead_days} Days Earlier",
                    showarrow=False,
                    font=dict(color="#06B6D4", size=12, family="sans-serif"),
                    bgcolor="rgba(6, 182, 212, 0.15)",
                    bordercolor="#06B6D4",
                    borderwidth=1,
                    borderpad=4,
                )

    fig.add_vline(
        x=selected_date,
        line=dict(color="#F8FAFC", width=1.5, dash="dot"),
        annotation_text="Today",
        annotation_position="bottom right",
        annotation_font=dict(color="#F8FAFC", size=10),
    )

    if has_secondary:
        fig.add_trace(
            go.Scatter(
                x=df_view["date"],
                y=df_view["official_cases"],
                mode="lines",
                name="Official Cases (Reported)",
                line=dict(color="#F8FAFC", width=2.0, dash="dash"),
                hovertemplate="<b>Official Cases</b>: %{y} cases<extra></extra>",
            ),
            secondary_y=True,
        )
        if "official_threshold" in df_view.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_view["date"],
                    y=df_view["official_threshold"],
                    mode="lines",
                    name="Official Reporting Threshold",
                    line=dict(color="rgba(248, 250, 252, 0.45)", width=1.2, dash="dot"),
                    hovertemplate="<b>Official Threshold</b>: %{y:.1f}<extra></extra>",
                ),
                secondary_y=True,
            )

    fig.update_layout(
        paper_bgcolor="#111B2E",
        plot_bgcolor="#111B2E",
        height=height,
        margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color="#94A3B8", size=11),
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#1E293B",
        linecolor="#233149",
        tickfont=dict(color="#94A3B8", size=11),
    )
    fig.update_yaxes(
        title="Anomaly Z-Score (σ)",
        title_font=dict(color="#94A3B8", size=12),
        showgrid=True,
        gridcolor="#1E293B",
        linecolor="#233149",
        tickfont=dict(color="#94A3B8", size=11),
        secondary_y=False,
    )
    if has_secondary:
        fig.update_yaxes(
            title="Official Clinical Cases",
            title_font=dict(color="#22C55E", size=12),
            showgrid=False,
            tickfont=dict(color="#22C55E", size=11),
            secondary_y=True,
        )

    return fig

def create_risk_gauge(risk_score: float, severity: str, height: int = 240) -> go.Figure:
    """Create a Plotly gauge showing current regional risk score and severity steps."""
    bar_color = SEV_COLOR_MAP.get(severity, "#22C55E")
    val = float(np.clip(risk_score, 0.0, 100.0))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=val,
            title={"text": "Current Regional Risk", "font": {"size": 14, "color": "#94A3B8"}},
            number={"font": {"size": 30, "color": "#F8FAFC"}, "suffix": " / 100"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748B", "tickfont": {"size": 10, "color": "#64748B"}},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "#1E293B",
                "borderwidth": 1,
                "bordercolor": "#233149",
                "steps": [
                    {"range": [0, 40], "color": "rgba(34, 197, 94, 0.15)"},
                    {"range": [40, 60], "color": "rgba(234, 179, 8, 0.15)"},
                    {"range": [60, 80], "color": "rgba(249, 115, 22, 0.15)"},
                    {"range": [80, 100], "color": "rgba(239, 68, 68, 0.15)"},
                ],
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="#111B2E",
        plot_bgcolor="#111B2E",
        margin=dict(l=25, r=25, t=35, b=20),
        height=height,
    )
    return fig

def create_signal_contribution_chart(
    region_row: pd.Series,
    weights: Dict[str, float],
    height: int = 240,
) -> go.Figure:
    """Create a horizontal bar chart showing each leading signal's exact contribution to risk.

    Formula: Contribution_i = 100 * (w_i * clip(z_i, 0, 5)) / 5
    Sum of contributions equals overall risk score.

    Args:
        region_row: Series containing selected region's observation on simulation date.
        weights: Dictionary of signal weights.
        height: Chart height in pixels.

    Returns:
        go.Figure: Horizontal bar chart sorted by contribution descending.
    """
    from detector import normalize_weights

    w = normalize_weights(weights)
    items = []

    for sig in LEADING_SIGNALS:
        z_col = f"z_{sig}"
        z_val = float(region_row.get(z_col, 0.0))
        z_clipped = float(np.clip(z_val, 0.0, 5.0))
        weight_norm = w.get(sig, 0.25)
        contrib = 100.0 * (weight_norm * z_clipped) / 5.0

        items.append({
            "signal": sig,
            "label": SIGNAL_DISPLAY_NAMES.get(sig, sig),
            "contribution": np.round(contrib, 2),
            "z_score": z_val,
            "weight": weight_norm,
            "color": SIGNAL_PALETTE.get(sig, "#06B6D4"),
        })

    items.sort(key=lambda x: x["contribution"], reverse=True)

    labels = [it["label"] for it in items]
    contribs = [it["contribution"] for it in items]
    colors = [it["color"] for it in items]
    hover_texts = [
        f"<b>{it['label']}</b><br>Contribution: {it['contribution']:.1f} pts<br>Z-score: {it['z_score']:+.2f}σ<br>Weight: {it['weight']*100:.0f}%"
        for it in items
    ]

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=contribs,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#233149", width=1)),
            text=[f"+{c:.1f} pts" for c in contribs],
            textposition="auto",
            textfont=dict(color="#F8FAFC", size=11),
            hoverinfo="text",
            hovertext=hover_texts,
        )
    )

    fig.update_layout(
        paper_bgcolor="#111B2E",
        plot_bgcolor="#111B2E",
        height=height,
        margin=dict(l=10, r=20, t=20, b=25),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#F8FAFC", size=11)),
        xaxis=dict(
            title="Risk Contribution (Points)",
            title_font=dict(color="#94A3B8", size=11),
            showgrid=True,
            gridcolor="#1E293B",
            tickfont=dict(color="#94A3B8", size=10),
        ),
    )
    return fig

def prepare_signal_status_table(
    region_row: pd.Series,
    weights: Dict[str, float],
    z_threshold: float,
) -> pd.DataFrame:
    """Format a detailed table of current signal values, baselines, Z-scores, and status.

    Args:
        region_row: Series containing selected region's observation on simulation date.
        weights: Dictionary of signal weights.
        z_threshold: Sensitivity threshold.

    Returns:
        pd.DataFrame: Formatted signal status table.
    """
    from detector import normalize_weights
    w = normalize_weights(weights)

    records = []
    for sig in LEADING_SIGNALS:
        label = SIGNAL_DISPLAY_NAMES.get(sig, sig)
        raw_val = float(region_row.get(sig, 0.0))
        mean_val = float(region_row.get(f"{sig}_mean", 0.0))
        std_val = float(region_row.get(f"{sig}_std", 1.0))
        z_val = float(region_row.get(f"z_{sig}", 0.0))
        weight_norm = w.get(sig, 0.25)
        contrib = 100.0 * (weight_norm * np.clip(z_val, 0.0, 5.0)) / 5.0

        is_anom = z_val > z_threshold
        status = "ANOMALOUS" if is_anom else "NORMAL"

        records.append({
            "Signal": label,
            "Current Value": f"{raw_val:.2f}",
            "Historical Baseline": f"{mean_val:.2f} ± {std_val:.2f}",
            "Z-Score": f"{z_val:+.2f}σ",
            "Weight": f"{weight_norm * 100:.0f}%",
            "Risk Contribution": f"{contrib:.1f} pts",
            "Status": status,
        })

    return pd.DataFrame(records)

def calculate_signal_trend(
    region_df: pd.DataFrame,
    selected_date: str,
    z_threshold: float = 2.0,
) -> Dict:
    """Evaluate 7-day trajectory, leading indicator onset, and top risk driver.

    Args:
        region_df: Telemetry for the selected region.
        selected_date: Simulation observation date.
        z_threshold: Threshold for anomaly onset.

    Returns:
        dict: Summary metrics for historical signal insights.
    """
    df_past = region_df[region_df["date"] <= selected_date].sort_values("date").copy()
    if df_past.empty:
        return {
            "trend_direction": "Stable",
            "trend_delta": 0.0,
            "first_anomalous_signal": "None",
            "top_driver": "None",
            "anom_count": 0,
        }

    current_row = df_past.iloc[-1]
    current_risk = float(current_row.get("risk_score", 0.0))

    if len(df_past) >= 8:
        past_7_risk = float(df_past.iloc[-8].get("risk_score", current_risk))
        delta = current_risk - past_7_risk
        if delta > 3.0:
            direction = "Increasing"
        elif delta < -3.0:
            direction = "Decreasing"
        else:
            direction = "Stable"
    else:
        direction = "Stable"
        delta = 0.0

    first_sig = "None"
    first_sig_date = None
    for sig in LEADING_SIGNALS:
        z_col = f"z_{sig}"
        if z_col in df_past.columns:
            anoms = df_past[df_past[z_col] > z_threshold]
            if not anoms.empty:
                min_d = anoms["date"].min()
                if first_sig_date is None or min_d < first_sig_date:
                    first_sig_date = min_d
                    first_sig = f"{SIGNAL_DISPLAY_NAMES.get(sig, sig)} ({min_d})"

    top_driver = "None"
    max_z = -999.0
    for sig in LEADING_SIGNALS:
        z_col = f"z_{sig}"
        z_val = float(current_row.get(z_col, 0.0))
        if z_val > max_z:
            max_z = z_val
            top_driver = f"{SIGNAL_DISPLAY_NAMES.get(sig, sig)} ({z_val:+.2f}σ)"

    anom_count = int(current_row.get("anomalous_signals_count", 0))

    return {
        "trend_direction": direction,
        "trend_delta": np.round(delta, 1),
        "first_anomalous_signal": first_sig,
        "top_driver": top_driver,
        "anom_count": anom_count,
    }

def create_replay_trajectory_chart(
    processed_df: pd.DataFrame,
    current_date: str,
    outbreak_regions: Optional[List[str]] = None,
    alert_threshold: float = 60.0,
    height: int = 280,
) -> go.Figure:
    """Generate risk score evolution chart for affected outbreak regions strictly up to current_date.

    Guarantees zero-lookahead: no future points after current_date are rendered.

    Args:
        processed_df: Full processed surveillance DataFrame.
        current_date: Current simulation playback date.
        outbreak_regions: List of regions to compare (defaults to Region 3, 4, 5).
        alert_threshold: Threshold line to display.
        height: Chart height in pixels.

    Returns:
        go.Figure: Plotly figure showing dynamic risk trajectories.
    """
    if outbreak_regions is None:
        outbreak_regions = ["Region 3", "Region 4", "Region 5"]

    palette = {
        "Region 3": "#EF4444",
        "Region 4": "#F97316",
        "Region 5": "#EAB308",
    }

    df_past = processed_df[processed_df["date"] <= current_date]
    fig = go.Figure()

    for reg in outbreak_regions:
        reg_df = df_past[df_past["region"] == reg].sort_values("date")
        if not reg_df.empty:
            color = palette.get(reg, "#38BDF8")
            is_epicenter = " (Epicenter)" if reg == "Region 3" else ""
            fig.add_trace(
                go.Scatter(
                    x=reg_df["date"],
                    y=reg_df["risk_score"],
                    mode="lines",
                    name=f"{reg}{is_epicenter}",
                    line=dict(color=color, width=2.2 if reg == "Region 3" else 1.8),
                    hovertemplate=f"<b>{reg}</b>: %{{y:.1f}}/100<extra></extra>",
                )
            )

    fig.add_hline(
        y=alert_threshold,
        line=dict(color="#F97316", width=1.5, dash="dash"),
        annotation_text=f"Alert Threshold ({alert_threshold:.0f})",
        annotation_position="top left",
        annotation_font=dict(color="#F97316", size=10),
    )

    fig.update_layout(
        paper_bgcolor="#111B2E",
        plot_bgcolor="#111B2E",
        height=height,
        margin=dict(l=35, r=25, t=25, b=25),
        title=dict(
            text="Epidemic Wave Risk Trajectories (Outbreak Regions)",
            font=dict(size=13, color="#F8FAFC"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#94A3B8", size=10),
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#1E293B",
        linecolor="#233149",
        tickfont=dict(color="#94A3B8", size=10),
    )
    fig.update_yaxes(
        title="Risk Score (0–100)",
        title_font=dict(color="#94A3B8", size=10),
        range=[0, 100],
        showgrid=True,
        gridcolor="#1E293B",
        linecolor="#233149",
        tickfont=dict(color="#94A3B8", size=10),
    )
    return fig

def render_replay_timeline_html(
    current_date: str,
    onset_date: str = "2025-03-10",
    anomaly_date: str = "2025-03-11",
    alert_date: str = "2025-03-13",
    official_date: str = "2025-03-21",
    lead_days: int = 8,
) -> str:
    """Generate dynamic horizontal HTML milestone cards reflecting playback progression."""
    milestones = [
        {"id": "onset", "date": onset_date, "label": "Outbreak Onset", "desc": "Synthetic wave initiates"},
        {"id": "anom", "date": anomaly_date, "label": "First Anomaly", "desc": "Wastewater crosses +2.0σ"},
        {"id": "alert", "date": alert_date, "label": "Whisper Alert", "desc": "2-day persistent warning"},
        {"id": "official", "date": official_date, "label": "Official Report", "desc": "Clinical cases exceed baseline"},
    ]

    steps_html = []
    for m in milestones:
        m_date = m["date"]
        if current_date > m_date:
            badge_class = "step-completed"
            icon = "✓"
            status_text = "COMPLETED"
            border_color = "#22C55E"
            text_color = "#F8FAFC"
        elif current_date == m_date:
            badge_class = "step-active"
            icon = "●"
            status_text = "CURRENT DAY"
            border_color = "#06B6D4"
            text_color = "#06B6D4"
        else:
            badge_class = "step-pending"
            icon = "○"
            status_text = "UPCOMING"
            border_color = "#334155"
            text_color = "#64748B"

        step_card = f"""
        <div style="flex: 1; min-width: 170px; background: #111B2E; border: 1px solid {border_color}; border-radius: 6px; padding: 12px 14px; position: relative;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 11px; font-weight: 700; color: {border_color};">{icon} {status_text}</span>
                <span style="font-family: monospace; font-size: 11px; color: #94A3B8;">{m_date}</span>
            </div>
            <div style="font-size: 13px; font-weight: 700; color: {text_color};">{m['label']}</div>
            <div style="font-size: 11px; color: #64748B; margin-top: 2px;">{m['desc']}</div>
        </div>
        """
        steps_html.append(step_card)

    lead_badge = ""
    if current_date >= official_date:
        lead_badge = f"""
        <div style="display: flex; align-items: center; justify-content: center; background: rgba(6, 182, 212, 0.15); border: 1px solid #06B6D4; border-radius: 6px; padding: 10px 16px; margin-top: 10px; font-size: 13px; font-weight: 700; color: #06B6D4;">
            ★ Verified Early Detection Advantage: {lead_days} Days Lead Time Achieved
        </div>
        """

    return f"""
    <div style="margin-bottom: 20px;">
        <div style="display: flex; gap: 12px; flex-wrap: wrap; justify-content: space-between;">
            {''.join(steps_html)}
        </div>
        {lead_badge}
    </div>
    """

def render_milestone_banner_html(
    current_date: str,
    region: str = "Region 3",
    alert_date: str = "2025-03-13",
    official_date: str = "2025-03-21",
    lead_days: int = 8,
    initial_risk: float = 78.4,
) -> Optional[str]:
    """Generate high-visibility milestone notification banners during replay progression."""
    if current_date >= official_date:
        return f"""
        <div style="background: linear-gradient(90deg, rgba(6, 182, 212, 0.18) 0%, rgba(34, 197, 94, 0.18) 100%); border: 1px solid #06B6D4; border-radius: 8px; padding: 16px 20px; margin-bottom: 18px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <span style="font-size: 16px; font-weight: 700; color: #F8FAFC;">
                        🏥 Official Report Issued • Whisper Alerted {lead_days} Days Earlier
                    </span>
                    <div style="font-size: 12px; color: #94A3B8; margin-top: 3px;">
                        {region} official clinical cases crossed threshold on {official_date}.
                        Whisper issued its first qualifying alert on {alert_date} (Risk: {initial_risk:.1f}/100).
                    </div>
                </div>
                <div style="font-size: 20px; font-weight: 800; color: #06B6D4; font-family: monospace;">
                    +{lead_days} DAYS
                </div>
            </div>
        </div>
        """
    elif current_date >= alert_date:
        return f"""
        <div style="background: rgba(249, 115, 22, 0.12); border: 1px solid #F97316; border-radius: 8px; padding: 14px 18px; margin-bottom: 18px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="font-size: 14px; font-weight: 700; color: #F97316;">
                        🚨 Outbreak Alert Episode Active • {region}
                    </span>
                    <div style="font-size: 12px; color: #94A3B8; margin-top: 2px;">
                        First alert triggered on {alert_date}. Clinical cases have not yet crossed official reporting threshold.
                    </div>
                </div>
                <span style="font-size: 12px; font-weight: 600; color: #F97316; background: rgba(249, 115, 22, 0.2); padding: 4px 10px; border-radius: 4px;">
                    Awaiting Official Confirmation
                </span>
            </div>
        </div>
        """
    return None

def create_workforce_map(
    day_df: pd.DataFrame,
    selected_org: str = "All Organizations",
    height: int = 400,
) -> go.Figure:
    """Create interactive geospatial map of synthetic workplace organizations and sites.

    Markers scaled by workforce_risk and color-coded by severity.
    """
    df_plot = day_df.copy()
    scatter_cls = getattr(go, "Scattermap", getattr(go, "Scattermapbox", None))

    colors = [SEV_COLOR_MAP.get(str(sev).capitalize(), "#22C55E") for sev in df_plot["severity"]]
    sizes = [float(np.clip(16 + (r / 100.0) * 22, 16, 38)) for r in df_plot["workforce_risk"]]

    customdata = np.stack(
        (
            df_plot["organization"],
            df_plot["site"],
            df_plot["region"],
            df_plot["total_employees"],
            df_plot["absent_employees"],
            df_plot["sick_leave"],
            df_plot["operational_capacity"],
            df_plot["workforce_risk"],
            df_plot["severity"],
        ),
        axis=-1,
    )

    hovertemplate = (
        "<b>%{customdata[0]}</b><br>"
        "Site: %{customdata[1]} (%{customdata[2]})<br>"
        "Total Workforce: %{customdata[3]:,} employees<br>"
        "Absent: %{customdata[4]:,} (%{customdata[5]:.1f}%)<br>"
        "Operational Capacity: %{customdata[6]:.1f}%<br>"
        "Workforce Risk: <b>%{customdata[7]:.1f} / 100</b> (%{customdata[8]})"
        "<extra></extra>"
    )

    fig = go.Figure()

    fig.add_trace(
        scatter_cls(
            lat=df_plot["lat"],
            lon=df_plot["lon"],
            mode="markers",
            marker=dict(size=sizes, color=colors, opacity=0.90),
            customdata=customdata,
            hovertemplate=hovertemplate,
            name="Sites",
        )
    )

    if selected_org != "All Organizations":
        foc_df = df_plot[df_plot["organization"] == selected_org]
        if not foc_df.empty:
            foc_sizes = [float(np.clip(26 + (r / 100.0) * 24, 26, 50)) for r in foc_df["workforce_risk"]]
            fig.add_trace(
                scatter_cls(
                    lat=foc_df["lat"],
                    lon=foc_df["lon"],
                    mode="markers",
                    marker=dict(size=foc_sizes, color="#06B6D4", opacity=0.35),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_layout(
        paper_bgcolor="#111B2E",
        plot_bgcolor="#111B2E",
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )

    center_lat = 21.5
    center_lon = 80.0
    zoom = 3.6
    map_dict = dict(style="carto-darkmatter", center=dict(lat=center_lat, lon=center_lon), zoom=zoom)

    if hasattr(fig.layout, "map"):
        fig.update_layout(map=map_dict)
    elif hasattr(fig.layout, "mapbox"):
        fig.update_layout(mapbox=map_dict)

    return fig

def create_workforce_trend_chart(
    org_df: pd.DataFrame,
    current_date: str,
    selected_org: str,
    window: Union[int, str] = 60,
    z_threshold: float = 2.0,
    height: int = 300,
) -> go.Figure:
    """Create interactive timeseries chart showing sick-leave % vs historical baseline and threshold."""
    df_hist = org_df[org_df["date"] <= current_date].copy()
    if df_hist.empty:
        df_hist = org_df.head(1).copy()

    if isinstance(window, int) and len(df_hist) > window:
        df_view = df_hist.tail(window).copy()
    else:
        df_view = df_hist.copy()

    thresh_curve = df_view["sick_leave_mean"] + z_threshold * df_view["sick_leave_std"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_view["date"],
            y=df_view["sick_leave_mean"],
            mode="lines",
            name="28-Day Baseline (Mean)",
            line=dict(color="#64748B", width=1.5, dash="dash"),
            hovertemplate="Baseline: %{y:.2f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_view["date"],
            y=thresh_curve,
            mode="lines",
            name=f"Anomaly Threshold (+{z_threshold:.1f}σ)",
            line=dict(color="#EF4444", width=1.2, dash="dot"),
            hovertemplate="Threshold: %{y:.2f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_view["date"],
            y=df_view["sick_leave"],
            mode="lines+markers",
            name="Reported Sick Leave %",
            line=dict(color="#06B6D4", width=2.4),
            marker=dict(size=4, color="#06B6D4"),
            hovertemplate="Sick Leave: <b>%{y:.2f}%</b><extra></extra>",
        )
    )

    fig.add_vline(
        x=current_date,
        line=dict(color="#F8FAFC", width=1.2, dash="dot"),
        annotation_text="Today",
        annotation_position="bottom right",
        annotation_font=dict(color="#F8FAFC", size=10),
    )

    fig.update_layout(
        paper_bgcolor="#111B2E",
        plot_bgcolor="#111B2E",
        height=height,
        margin=dict(l=35, r=25, t=25, b=25),
        title=dict(
            text=f"Absenteeism Trajectory & Threshold — {selected_org}",
            font=dict(size=13, color="#F8FAFC"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color="#94A3B8", size=10),
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#1E293B",
        linecolor="#233149",
        tickfont=dict(color="#94A3B8", size=10),
    )
    fig.update_yaxes(
        title="Sick Leave Rate (%)",
        title_font=dict(color="#94A3B8", size=10),
        showgrid=True,
        gridcolor="#1E293B",
        linecolor="#233149",
        tickfont=dict(color="#94A3B8", size=10),
    )
    return fig

def create_org_comparison_chart(day_df: pd.DataFrame, height: int = 220) -> go.Figure:
    """Create horizontal bar chart ranking organizations by current workforce risk."""
    df_sorted = day_df.sort_values("workforce_risk", ascending=True)

    colors = [SEV_COLOR_MAP.get(str(sev).capitalize(), "#22C55E") for sev in df_sorted["severity"]]
    labels = df_sorted["organization"].tolist()
    risks = df_sorted["workforce_risk"].tolist()
    leaves = df_sorted["sick_leave"].tolist()
    caps = df_sorted["operational_capacity"].tolist()

    hover_texts = [
        f"<b>{o}</b><br>Workforce Risk: {r:.1f}/100<br>Sick Leave: {l:.1f}%<br>Operational Capacity: {c:.1f}%"
        for o, r, l, c in zip(labels, risks, leaves, caps)
    ]

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=risks,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#233149", width=1)),
            text=[f"{r:.1f} pts ({l:.1f}%)" for r, l in zip(risks, leaves)],
            textposition="auto",
            textfont=dict(color="#F8FAFC", size=11),
            hoverinfo="text",
            hovertext=hover_texts,
        )
    )

    fig.update_layout(
        paper_bgcolor="#111B2E",
        plot_bgcolor="#111B2E",
        height=height,
        margin=dict(l=10, r=20, t=20, b=25),
        xaxis=dict(
            title="Workforce Risk Score (0–100)",
            title_font=dict(color="#94A3B8", size=10),
            range=[0, 100],
            showgrid=True,
            gridcolor="#1E293B",
            tickfont=dict(color="#94A3B8", size=10),
        ),
        yaxis=dict(tickfont=dict(color="#F8FAFC", size=11)),
    )
    return fig

def prepare_workforce_alert_feed(
    alert_events: List[Any],
    processed_df: pd.DataFrame,
    selected_date: str,
    status_filter: str = "All",
) -> List[Dict[str, Any]]:
    """Format and filter workforce alert episodes up to selected_date with no-lookahead."""
    feed = []
    for ev in alert_events:
        ep_dict = ev.to_dict() if hasattr(ev, "to_dict") else dict(ev)
        first_d = ep_dict.get("first_alert_date", "")
        last_d = ep_dict.get("last_alert_date", "")

        if first_d > selected_date:
            continue

        is_active = (first_d <= selected_date <= last_d)

        org = ep_dict.get("organization")
        curr_row = processed_df[(processed_df["organization"] == org) & (processed_df["date"] == selected_date)]
        if not curr_row.empty:
            curr_r = float(curr_row["workforce_risk"].iloc[0])
            curr_sev = str(curr_row["severity"].iloc[0])
            curr_leave = float(curr_row["sick_leave"].iloc[0])
            curr_z = float(curr_row["z_absence"].iloc[0])
        else:
            curr_r = ep_dict.get("peak_risk_score", 0.0)
            curr_sev = ep_dict.get("severity", "Yellow")
            curr_leave = ep_dict.get("peak_sick_leave", 0.0)
            curr_z = 2.0

        item = {
            "episode_id": ep_dict.get("episode_id"),
            "organization": org,
            "site": ep_dict.get("site"),
            "region": ep_dict.get("region"),
            "first_alert_date": first_d,
            "last_alert_date": last_d,
            "current_risk_score": curr_r,
            "current_severity": curr_sev,
            "current_sick_leave": curr_leave,
            "current_z": curr_z,
            "is_active": is_active,
            "explanation": ep_dict.get("explanation", ""),
        }

        sf = status_filter.lower()
        if "active" in sf and not is_active:
            continue
        if "resolved" in sf and is_active:
            continue
        if "red" in sf and curr_sev.upper() != "RED":
            continue
        if "orange" in sf and curr_sev.upper() != "ORANGE":
            continue
        if "yellow" in sf and curr_sev.upper() != "YELLOW":
            continue

        feed.append(item)

    feed.sort(key=lambda x: x["first_alert_date"], reverse=True)
    return feed

def render_workforce_alert_card(alert: Dict[str, Any]) -> str:
    """Render a styled Clean Command Center card for a workforce alert."""
    sev = alert["current_severity"].capitalize()
    pill_class = f"pill-{sev.lower()}"
    status_label = "ACTIVE" if alert["is_active"] else "RESOLVED"
    status_color = "#EF4444" if alert["is_active"] else "#64748B"
    bg_border = "#EF4444" if (alert["is_active"] and sev == "Red") else "#233149"

    return f"""
    <div style="background: #111B2E; border: 1px solid {bg_border}; border-radius: 8px; padding: 14px 16px; margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div>
                <span style="font-size: 15px; font-weight: 700; color: #F8FAFC;">{alert['organization']}</span>
                <span class="severity-pill {pill_class}" style="margin-left: 6px;">{sev}</span>
            </div>
            <span style="font-size: 11px; font-weight: 700; color: {status_color}; background: rgba(255,255,255,0.05); padding: 3px 8px; border-radius: 4px;">
                {status_label}
            </span>
        </div>
        <div style="font-size: 12px; color: #94A3B8; margin-bottom: 6px;">
            Site: <strong style="color: #F8FAFC;">{alert['site']}</strong> • Region: <strong style="color: #06B6D4;">{alert['region']}</strong>
        </div>
        <div style="display: flex; gap: 16px; font-size: 12px; color: #94A3B8; margin-bottom: 6px;">
            <div>Sick Leave: <strong style="color: #F8FAFC;">{alert['current_sick_leave']:.1f}%</strong> (+{alert['current_z']:.1f}σ)</div>
            <div>Workforce Risk: <strong style="color: #F8FAFC;">{alert['current_risk_score']:.1f}/100</strong></div>
            <div>Alert Date: <strong style="color: #06B6D4; font-family: monospace;">{alert['first_alert_date']}</strong></div>
        </div>
        <div style="font-size: 12px; color: #CBD5E1; line-height: 1.4; border-top: 1px solid #1E293B; padding-top: 6px;">
            {alert['explanation']}
        </div>
    </div>
    """

def create_lead_time_comparison_chart(
    regional_summary: pd.DataFrame,
    height: int = 240,
) -> go.Figure:
    """Create bar chart visualizing lead time advantage (days) across outbreak regions."""
    df_outbreak = regional_summary[regional_summary["Known Outbreak"] == "Yes"].copy()
    df_outbreak["Lead Days Safe"] = df_outbreak["Lead Days"].fillna(0).astype(float)

    colors = ["#06B6D4" if ld > 0 else "#EF4444" for ld in df_outbreak["Lead Days Safe"]]

    fig = go.Figure(
        go.Bar(
            x=df_outbreak["Region"],
            y=df_outbreak["Lead Days Safe"],
            marker=dict(color=colors, line=dict(color="#233149", width=1)),
            text=[f"+{int(ld)} Days Earlier" if ld > 0 else "0 Days" for ld in df_outbreak["Lead Days Safe"]],
            textposition="auto",
            textfont=dict(color="#F8FAFC", size=12),
            hovertemplate="<b>%{x}</b><br>Lead Time: %{y:.0f} Days Earlier<extra></extra>",
        )
    )

    fig.update_layout(
        paper_bgcolor="#111B2E",
        plot_bgcolor="#111B2E",
        height=height,
        margin=dict(l=30, r=20, t=30, b=25),
        title=dict(
            text="Early-Warning Lead Time Advantage (Days Before Official Confirmation)",
            font=dict(size=13, color="#F8FAFC"),
        ),
        yaxis=dict(
            title="Lead Time (Days)",
            title_font=dict(color="#94A3B8", size=11),
            showgrid=True,
            gridcolor="#1E293B",
            tickfont=dict(color="#94A3B8", size=10),
        ),
        xaxis=dict(tickfont=dict(color="#F8FAFC", size=11)),
    )
    return fig

def create_historical_risk_progression_chart(
    processed_df: pd.DataFrame,
    outbreak_config: Optional[Dict[str, Any]] = None,
    height: int = 300,
) -> go.Figure:
    """Create retrospective multi-region risk progression chart across complete timeline."""
    regions_to_plot = ["Region 3", "Region 4", "Region 5", "Region 1"]
    palette = {
        "Region 3": "#EF4444",
        "Region 4": "#F97316",
        "Region 5": "#EAB308",
        "Region 1": "#64748B",
    }

    fig = go.Figure()

    for reg in regions_to_plot:
        reg_df = processed_df[processed_df["region"] == reg].sort_values("date")
        if not reg_df.empty:
            color = palette.get(reg, "#38BDF8")
            is_control = " (Control / Baseline)" if reg == "Region 1" else ""
            fig.add_trace(
                go.Scatter(
                    x=reg_df["date"],
                    y=reg_df["risk_score"],
                    mode="lines",
                    name=f"{reg}{is_control}",
                    line=dict(color=color, width=2.2 if reg == "Region 3" else 1.5),
                    hovertemplate=f"<b>{reg}</b>: %{{y:.1f}}/100<extra></extra>",
                )
            )

    fig.add_vline(
        x="2025-03-10",
        line=dict(color="#38BDF8", width=1.5, dash="dash"),
        annotation_text="Injected Wave Onset (Mar 10)",
        annotation_position="top left",
        annotation_font=dict(color="#38BDF8", size=10),
    )
    fig.add_vline(
        x="2025-03-13",
        line=dict(color="#06B6D4", width=1.8, dash="dot"),
        annotation_text="Whisper Alert (Mar 13)",
        annotation_position="top right",
        annotation_font=dict(color="#06B6D4", size=10),
    )
    fig.add_vline(
        x="2025-03-21",
        line=dict(color="#EF4444", width=1.5, dash="dash"),
        annotation_text="Official Clinical Report (Mar 21)",
        annotation_position="bottom right",
        annotation_font=dict(color="#EF4444", size=10),
    )

    fig.update_layout(
        paper_bgcolor="#111B2E",
        plot_bgcolor="#111B2E",
        height=height,
        margin=dict(l=35, r=25, t=30, b=25),
        title=dict(
            text="Retrospective Risk Progression Across 2-Year Surveillance History",
            font=dict(size=13, color="#F8FAFC"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#94A3B8", size=10),
        ),
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#1E293B",
        linecolor="#233149",
        tickfont=dict(color="#94A3B8", size=10),
    )
    fig.update_yaxes(
        title="Risk Score (0–100)",
        title_font=dict(color="#94A3B8", size=10),
        range=[0, 100],
        showgrid=True,
        gridcolor="#1E293B",
        linecolor="#233149",
        tickfont=dict(color="#94A3B8", size=10),
    )
    return fig

def create_sensitivity_heatmap(
    sensitivity_df: pd.DataFrame,
    metric: str = "f1_score",
    height: int = 260,
) -> go.Figure:
    """Create heatmap showing sensitivity analysis metric across Z and alert threshold combinations."""
    pivot_df = sensitivity_df.pivot(index="alert_threshold", columns="z_threshold", values=metric)

    metric_titles = {
        "f1_score": "F1 Score Matrix",
        "detection_rate": "Detection Rate (%)",
        "false_alarms": "False Alarm Episodes",
        "avg_lead_time": "Average Lead Time (Days)",
    }
    title_text = metric_titles.get(metric, metric)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_df.values,
            x=[f"{z:.1f}σ" for z in pivot_df.columns],
            y=[f"Risk {int(a)}" for a in pivot_df.index],
            colorscale="Teal",
            text=np.round(pivot_df.values, 2),
            texttemplate="%{text}",
            textfont=dict(color="#F8FAFC", size=11),
            hovertemplate="Sensitivity: %{x}<br>Alert Threshold: %{y}<br>Value: <b>%{z:.2f}</b><extra></extra>",
            showscale=False,
        )
    )

    fig.update_layout(
        paper_bgcolor="#111B2E",
        plot_bgcolor="#111B2E",
        height=height,
        margin=dict(l=40, r=20, t=30, b=25),
        title=dict(
            text=f"Threshold Sensitivity Trade-Off: {title_text}",
            font=dict(size=13, color="#F8FAFC"),
        ),
        xaxis=dict(
            title="Z-Score Sensitivity Threshold",
            title_font=dict(color="#94A3B8", size=10),
            tickfont=dict(color="#F8FAFC", size=10),
        ),
        yaxis=dict(
            title="Alert Qualification Threshold",
            title_font=dict(color="#94A3B8", size=10),
            tickfont=dict(color="#F8FAFC", size=10),
        ),
    )
    return fig
