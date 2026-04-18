# visualizations.py — All Plotly charts for the Startup Country Advisor

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import FEATURE_LABELS, CLUSTER_COLORS, FEATURE_COLS

# ── Color palette ─────────────────────────────────────────────────────────────
PRIMARY = "#1f77b4"
ACCENT = "#ff7f0e"
SUCCESS = "#2ca02c"
DANGER = "#d62728"
NEUTRAL = "#7f7f7f"

BG_COLOR = "#0e1117"
CARD_COLOR = "#1c1f26"
TEXT_COLOR = "#fafafa"

PLOTLY_TEMPLATE = "plotly_dark"


def _base_layout(title: str = "", height: int = 400) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR)),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=CARD_COLOR,
        font=dict(color=TEXT_COLOR, size=12),
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. OPPORTUNITY GAUGE
# ─────────────────────────────────────────────────────────────────────────────

def opportunity_gauge(score: float, country: str, rank: int, total: int) -> go.Figure:
    """
    Circular gauge showing the opportunity score (0-100) for the target country.
    Color: red < 33, orange 33-66, green > 66.
    """
    color = SUCCESS if score >= 66 else (ACCENT if score >= 33 else DANGER)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number={"suffix": "/100", "font": {"size": 36, "color": TEXT_COLOR}},
        title={"text": f"Opportunity Score<br><span style='font-size:13px'>#{rank} of {total} countries</span>",
               "font": {"size": 15, "color": TEXT_COLOR}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT_COLOR},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0, 33], "color": "#3a1a1a"},
                {"range": [33, 66], "color": "#3a2f0d"},
                {"range": [66, 100], "color": "#0d2e1a"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))

    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. RADAR CHART — Country Feature Profile
# ─────────────────────────────────────────────────────────────────────────────

def radar_chart(
    country: str,
    percentiles: dict,
    alternatives: list[tuple[str, dict]] = None,
) -> go.Figure:
    """
    Spider/radar chart showing the percentile of each feature for target country.
    Optionally overlays 1-2 alternative countries.
    """
    features = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS if f in percentiles]
    values = [percentiles.get(f, 0) for f in FEATURE_COLS if f in percentiles]

    # Close the polygon
    features_closed = features + [features[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=features_closed,
        fill="toself",
        name=country,
        line=dict(color=PRIMARY, width=2),
        fillcolor=f"rgba(31, 119, 180, 0.25)",
    ))

    if alternatives:
        alt_colors = [ACCENT, SUCCESS]
        for i, (alt_name, alt_pct) in enumerate(alternatives[:2]):
            alt_vals = [alt_pct.get(f, 0) for f in FEATURE_COLS if f in alt_pct]
            alt_closed = alt_vals + [alt_vals[0]]
            color = alt_colors[i % len(alt_colors)]
            fig.add_trace(go.Scatterpolar(
                r=alt_closed,
                theta=features_closed,
                fill="toself",
                name=alt_name,
                line=dict(color=color, width=2, dash="dash"),
                fillcolor=f"rgba(255, 127, 14, 0.15)" if i == 0 else "rgba(44, 160, 44, 0.15)",
            ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10, color=TEXT_COLOR),
                            gridcolor="#333"),
            angularaxis=dict(tickfont=dict(size=11, color=TEXT_COLOR), gridcolor="#333"),
            bgcolor=CARD_COLOR,
        ),
        showlegend=True,
        legend=dict(font=dict(color=TEXT_COLOR)),
        **_base_layout(f"Feature Profile: {country} (Percentile Rank)", height=420),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. TOP COUNTRIES BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def top_countries_bar(df: pd.DataFrame, sector: str, target_country: str = None, n: int = 15) -> go.Figure:
    """
    Horizontal bar chart of top N countries by opportunity score.
    Target country highlighted in accent color.
    """
    top = df.sort_values("opportunity_score", ascending=False).head(n).copy()

    colors = [ACCENT if row["country"] == target_country else PRIMARY
              for _, row in top.iterrows()]

    fig = go.Figure(go.Bar(
        x=top["opportunity_score"],
        y=top["country"],
        orientation="h",
        marker_color=colors,
        text=top["opportunity_score"].round(1).astype(str),
        textposition="inside",
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
    ))

    fig.update_layout(
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        xaxis=dict(title="Opportunity Score (0-100)", range=[0, 100]),
        **_base_layout(f"Top {n} Countries — {sector.title()} Sector", height=max(350, n * 28)),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. WORLD MAP (CHOROPLETH)
# ─────────────────────────────────────────────────────────────────────────────

def world_map(df: pd.DataFrame, sector: str, target_country: str = None) -> go.Figure:
    """
    Choropleth world map colored by opportunity score.
    Target country outlined with a marker.
    """
    fig = px.choropleth(
        df,
        locations="country",
        locationmode="country names",
        color="opportunity_score",
        hover_name="country",
        hover_data={
            "opportunity_score": ":.1f",
            "rank": True,
            "cluster_name": True,
        },
        color_continuous_scale="RdYlGn",
        range_color=[0, 100],
        labels={"opportunity_score": "Score", "cluster_name": "Cluster"},
        title=f"Global Opportunity Map — {sector.title()} Sector",
        template=PLOTLY_TEMPLATE,
    )

    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        geo=dict(bgcolor=BG_COLOR, showframe=False, showcoastlines=True,
                 coastlinecolor="#444", showland=True, landcolor="#1c2733",
                 showocean=True, oceancolor="#0a1628"),
        coloraxis_colorbar=dict(title="Score", tickfont=dict(color=TEXT_COLOR)),
        height=480,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLUSTER SCATTER (PCA 2D)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_scatter(df: pd.DataFrame, target_country: str = None) -> go.Figure:
    """
    2D scatter plot of countries using PCA-reduced features.
    Countries colored by cluster. Target country highlighted.
    """
    fig = px.scatter(
        df,
        x="pca_x",
        y="pca_y",
        color="cluster_name",
        hover_name="country",
        hover_data={"opportunity_score": ":.1f", "pca_x": False, "pca_y": False},
        color_discrete_map=CLUSTER_COLORS,
        title="Country Clusters (PCA 2D Projection)",
        template=PLOTLY_TEMPLATE,
        labels={"pca_x": "Principal Component 1", "pca_y": "Principal Component 2"},
    )

    # Highlight target country
    if target_country:
        target_row = df[df["country"] == target_country]
        if not target_row.empty:
            fig.add_trace(go.Scatter(
                x=target_row["pca_x"],
                y=target_row["pca_y"],
                mode="markers+text",
                marker=dict(size=18, color="white", symbol="star",
                            line=dict(color=ACCENT, width=2)),
                text=[target_country],
                textposition="top center",
                textfont=dict(color=ACCENT, size=12),
                name=f"▶ {target_country}",
                showlegend=True,
            ))

    fig.update_layout(
        **_base_layout("Country Clusters — PCA Visualization", height=450),
        legend=dict(font=dict(color=TEXT_COLOR, size=11)),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. FEATURE COMPARISON BAR
# ─────────────────────────────────────────────────────────────────────────────

def feature_comparison_bar(
    target_country: str,
    country_scores: dict,   # {country_name: {feature: normalized_val}}
) -> go.Figure:
    """
    Grouped horizontal bar chart comparing feature scores across countries.
    """
    features = [FEATURE_LABELS.get(f, f) for f in FEATURE_COLS]
    fig = go.Figure()

    countries = list(country_scores.keys())
    palette = [PRIMARY, ACCENT, SUCCESS, "#e377c2", "#17becf"]

    for i, (country, scores) in enumerate(country_scores.items()):
        vals = [scores.get(f, 0) * 100 for f in FEATURE_COLS]  # scale to 0-100
        color = palette[i % len(palette)]
        fig.add_trace(go.Bar(
            name=country,
            x=features,
            y=vals,
            marker_color=color,
            hovertemplate=f"<b>{country}</b><br>%{{x}}: %{{y:.1f}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(title="Normalized Score (0-100)", range=[0, 100]),
        legend=dict(font=dict(color=TEXT_COLOR)),
        **_base_layout("Feature Comparison Across Countries", height=420),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. FEATURE HEATMAP (bonus)
# ─────────────────────────────────────────────────────────────────────────────

def feature_heatmap(country_list: list, df: pd.DataFrame) -> go.Figure:
    """
    Heatmap of normalized feature values for a list of countries.
    """
    available_feats = [f for f in FEATURE_COLS if f in df.columns]
    rows = []
    labels = []
    for c in country_list:
        row_data = df[df["country"] == c]
        if not row_data.empty:
            rows.append(row_data[available_feats].values[0] * 100)
            labels.append(c)

    if not rows:
        return go.Figure()

    z = np.array(rows)
    feat_labels = [FEATURE_LABELS.get(f, f) for f in available_feats]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=feat_labels,
        y=labels,
        colorscale="RdYlGn",
        zmin=0, zmax=100,
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}<extra></extra>",
        colorbar=dict(title="Score (0-100)", tickfont=dict(color=TEXT_COLOR)),
    ))

    fig.update_layout(
        xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=11)),
        **_base_layout("Feature Heatmap — Country Comparison", height=max(350, len(labels) * 35 + 100)),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. DATA QUALITY SUMMARY CHART
# ─────────────────────────────────────────────────────────────────────────────

def data_quality_chart(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing data completeness before and after cleaning.
    """
    from config import NUMERIC_COLS

    cols = [c for c in NUMERIC_COLS if c in df_raw.columns or c in df_clean.columns]

    before = [round((1 - df_raw[c].isna().mean()) * 100, 1) if c in df_raw.columns else 0 for c in cols]
    after = [round((1 - df_clean[c].isna().mean()) * 100, 1) if c in df_clean.columns else 100 for c in cols]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before Cleaning", x=cols, y=before,
                         marker_color=DANGER, opacity=0.7))
    fig.add_trace(go.Bar(name="After Cleaning", x=cols, y=after,
                         marker_color=SUCCESS, opacity=0.9))

    fig.update_layout(
        barmode="group",
        xaxis=dict(tickangle=-40, tickfont=dict(size=9)),
        yaxis=dict(title="Completeness (%)", range=[0, 105]),
        **_base_layout("Data Quality: Completeness Before & After Cleaning", height=380),
    )
    return fig
