# app.py — Streamlit Frontend for Startup Country Advisor

import json
import logging
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load .env before importing config (config also loads it, but this ensures it)
load_dotenv()

from config import (
    APP_TITLE, APP_SUBTITLE, DATA_PATH, FEATURE_LABELS,
    FEATURE_COLS, SECTORS, GLOSSARY, CLUSTER_COLORS,
)
from data_pipeline import run_pipeline, clean_data, load_raw_data
from groq_agent import parse_intent, synthesize_results, followup_chat
from results_engine import (
    get_country_profile, get_top_countries, get_similar_countries,
    get_feature_percentiles, compare_countries, build_llm_context,
    fuzzy_find_country,
)
from visualizations import (
    opportunity_gauge, radar_chart, top_countries_bar, world_map,
    cluster_scatter, feature_comparison_bar, feature_heatmap,
    data_quality_chart,
)

logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Startup Country Advisor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS STYLING
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Dark background tweaks */
    .stApp { background-color: #0e1117; }
    .stChatMessage { border-radius: 10px; margin-bottom: 8px; }
    
    /* Chat input styling */
    .stChatInput > div { border-radius: 20px; }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1c1f26;
        border: 1px solid #2a2d36;
        border-radius: 10px;
        padding: 12px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e0e0e0;
        border-left: 3px solid #1f77b4;
        padding-left: 10px;
        margin: 16px 0 10px 0;
    }
    
    /* Cluster badge */
    .cluster-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 4px;
    }

    /* Glossary term */
    .glossary-term {
        background: #1c1f26;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.85rem;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "messages": [],            # Chat history [{role, content}]
        "analysis_context": None,  # Last LLM context dict
        "df": None,                # Cleaned + scored dataframe
        "pipeline_run": False,     # Whether pipeline has been run
        "last_country": None,
        "last_sector": "general",
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),  # Pre-load from .env
        "data_loaded": False,
        "raw_df": None,            # For data quality chart
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (CACHED)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_and_clean_data(sector: str, data_path: str) -> dict:
    """Load + run full pipeline. Cached per sector."""
    return run_pipeline(sector=sector, data_path=data_path)


def check_data_file() -> bool:
    """Check if the dataset file exists."""
    return os.path.exists(DATA_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"# {APP_TITLE}")
    st.markdown(f"*{APP_SUBTITLE}*")
    st.divider()

    # Groq API Key
    st.markdown("### 🔑 Groq API Key")
    api_key_input = st.text_input(
        "Enter your Groq API key",
        type="password",
        value=st.session_state.groq_api_key,
        placeholder="gsk_...",
        help="Get a free key at https://console.groq.com. Or set GROQ_API_KEY in .env",
    )
    if api_key_input:
        st.session_state.groq_api_key = api_key_input
        st.success("✅ API key set")
    elif st.session_state.groq_api_key:
        st.success("✅ API key loaded from .env")

    st.divider()

    # Dataset status
    st.markdown("### 📂 Dataset")
    if check_data_file():
        st.success(f"✅ `{DATA_PATH}` found")
    else:
        st.error(f"❌ `{DATA_PATH}` not found")
        st.markdown("""
        **Setup:**
        1. Download `factbook.csv` from [Kaggle](https://www.kaggle.com/datasets/lucafrance/the-world-factbook-by-cia/data)
        2. Place it in `data/factbook.csv`
        """)

    st.divider()

    # Sector override
    st.markdown("### ⚙️ Default Sector")
    default_sector = st.selectbox(
        "Startup sector (auto-detected from chat)",
        options=SECTORS,
        index=SECTORS.index("general"),
        key="default_sector_select",
    )

    st.divider()

    # Dataset info
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("### 📊 Dataset Info")
        st.metric("Countries", len(df))
        st.metric("Features", len([f for f in FEATURE_COLS if f in df.columns]))
        st.metric("Clusters", df["cluster_name"].nunique() if "cluster_name" in df.columns else 0)

        # Cluster breakdown
        st.markdown("**Cluster Distribution:**")
        cluster_counts = df["cluster_name"].value_counts()
        for cluster, count in cluster_counts.items():
            color = CLUSTER_COLORS.get(cluster, "#666")
            st.markdown(
                f'<span class="cluster-badge" style="background:{color}22; color:{color}; border:1px solid {color}">'
                f'{cluster}: {count}</span>',
                unsafe_allow_html=True,
            )

    st.divider()

    # Data Mining Glossary
    with st.expander("📖 Data Mining Glossary"):
        for term, definition in GLOSSARY.items():
            st.markdown(
                f'<div class="glossary-term"><b>{term}:</b> {definition}</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT AREA — TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_chat, tab_analysis, tab_map, tab_data = st.tabs([
    "💬 Chat", "📊 Analysis", "🗺️ Map & Clusters", "🔬 Data Pipeline"
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: CHAT
# ─────────────────────────────────────────────────────────────────────────────

with tab_chat:
    st.markdown("### 💬 Startup Advisor Chat")
    st.markdown(
        "Describe your startup and the country you want to launch in. "
        "I'll run a full data mining analysis using CIA World Factbook data."
    )

    # Example prompts
    with st.expander("💡 Example prompts"):
        examples = [
            "I want to launch a fintech startup in Tunisia",
            "My edtech platform targets students in Nigeria",
            "We're building a logistics SaaS for the Southeast Asian market, starting with Vietnam",
            "I have a healthtech startup for rural areas — which African country is best?",
            "Compare Morocco and Egypt for an e-commerce business",
        ]
        for ex in examples:
            if st.button(f"→ {ex}", key=f"ex_{ex[:20]}"):
                st.session_state.messages.append({"role": "user", "content": ex})
                st.rerun()

    st.divider()

    # Chat history display
    chat_container = st.container(height=480)
    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                """
                <div style='text-align:center; color:#888; padding:80px 20px;'>
                    <div style='font-size:3rem'>🌍</div>
                    <div style='font-size:1.1rem; margin-top:10px;'>
                        Tell me about your startup and where you want to launch it.
                    </div>
                    <div style='font-size:0.9rem; color:#555; margin-top:8px;'>
                        Example: "I want to launch a fintech startup in Tunisia"
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Describe your startup and target country...")

    if user_input:
        # Validate prerequisites
        if not st.session_state.groq_api_key:
            st.error("⚠️ Please enter your Groq API key in the sidebar first.")
            st.stop()

        if not check_data_file():
            st.error("⚠️ Dataset not found. Please download `factbook.csv` and place it in `data/`.")
            st.stop()

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Determine if this is a new analysis or a follow-up
        is_new_analysis = (
            st.session_state.analysis_context is None
            or any(kw in user_input.lower() for kw in [
                "launch", "start", "compare", "analyze", "what about", "tell me about", "consider"
            ])
        )

        with st.spinner("🔍 Analyzing..."):
            try:
                if is_new_analysis:
                    # ── Parse intent ──────────────────────────────────────────
                    intent = parse_intent(user_input, st.session_state.groq_api_key)
                    sector = intent.get("sector", default_sector)
                    target_country = intent.get("target_country", "")
                    startup_desc = intent.get("startup_description", user_input)

                    # ── Run data mining pipeline ──────────────────────────────
                    pipeline_result = load_and_clean_data(sector, DATA_PATH)
                    df = pipeline_result["df"]
                    st.session_state.df = df
                    st.session_state.last_sector = sector
                    st.session_state.last_country = target_country

                    # ── Build LLM context ─────────────────────────────────────
                    context = build_llm_context(
                        target_country=target_country,
                        sector=sector,
                        startup_description=startup_desc,
                        df=df,
                        n_alternatives=5,
                    )
                    st.session_state.analysis_context = context

                    # ── Synthesize response ───────────────────────────────────
                    response = synthesize_results(context, st.session_state.groq_api_key)

                    # Prepend context line
                    header = (
                        f"**📍 Analysis:** `{target_country}` | "
                        f"**Sector:** `{sector.title()}` | "
                        f"**Score:** `{context.get('opportunity_score', 'N/A')}/100` | "
                        f"**Rank:** `#{context.get('global_rank', '?')}`\n\n"
                    )
                    response = header + response

                else:
                    # ── Follow-up question ────────────────────────────────────
                    response = followup_chat(
                        user_message=user_input,
                        chat_history=st.session_state.messages[:-1],
                        context=st.session_state.analysis_context,
                        api_key=st.session_state.groq_api_key,
                    )

            except Exception as e:
                response = f"❌ Error: {str(e)}\n\nPlease check your API key and dataset file."

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

with tab_analysis:
    df = st.session_state.df
    ctx = st.session_state.analysis_context
    country = st.session_state.last_country
    sector = st.session_state.last_sector

    if df is None or ctx is None:
        st.info("💬 Start a conversation in the Chat tab to see analysis results here.")
    elif "error" in ctx:
        st.error(ctx["error"])
    else:
        st.markdown(f"### 📊 Analysis: **{country}** — {sector.title()} Sector")

        # ── Row 1: Key metrics ────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Opportunity Score", f"{ctx['opportunity_score']:.1f}/100")
        with col2:
            st.metric("🏆 Global Rank", f"#{ctx['global_rank']} / {ctx['total_countries']}")
        with col3:
            st.metric("🌐 Cluster", ctx.get("cluster", "—"))
        with col4:
            pct_rank = round((1 - ctx["global_rank"] / ctx["total_countries"]) * 100, 1)
            st.metric("📈 Better Than", f"{pct_rank}% of countries")

        st.divider()

        # ── Row 2: Gauge + Radar ──────────────────────────────────────────────
        col_g, col_r = st.columns([1, 1.6])

        with col_g:
            st.markdown('<div class="section-header">Opportunity Score</div>', unsafe_allow_html=True)
            fig_gauge = opportunity_gauge(
                ctx["opportunity_score"], country,
                ctx["global_rank"], ctx["total_countries"]
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_r:
            st.markdown('<div class="section-header">Feature Profile (Percentile Rank)</div>', unsafe_allow_html=True)
            percentiles = get_feature_percentiles(country, df)

            # Get top alternative for overlay
            alt_pcts = []
            alts = ctx.get("top_alternative_countries", [])
            if alts:
                top_alt = alts[0]["country"]
                top_alt_pcts = get_feature_percentiles(top_alt, df)
                alt_pcts = [(top_alt, top_alt_pcts)]

            fig_radar = radar_chart(country, percentiles, alt_pcts)
            st.plotly_chart(fig_radar, use_container_width=True)

        st.divider()

        # ── Row 3: Top countries bar ──────────────────────────────────────────
        st.markdown('<div class="section-header">Top Countries by Opportunity Score</div>', unsafe_allow_html=True)
        fig_bar = top_countries_bar(df, sector, target_country=country, n=15)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # ── Row 4: Feature comparison heatmap ────────────────────────────────
        st.markdown('<div class="section-header">Feature Comparison — Target vs Alternatives</div>', unsafe_allow_html=True)

        alt_countries = [a["country"] for a in ctx.get("top_alternative_countries", [])[:4]]
        compare_list = [country] + alt_countries
        compare_list = [c for c in compare_list if fuzzy_find_country(c, df)]

        if len(compare_list) >= 2:
            fig_heat = feature_heatmap(compare_list, df)
            st.plotly_chart(fig_heat, use_container_width=True)

        # ── Row 5: Raw indicators table ───────────────────────────────────────
        st.markdown('<div class="section-header">Raw Economic Indicators</div>', unsafe_allow_html=True)
        raw = ctx.get("raw_indicators", {})
        if raw:
            raw_df = pd.DataFrame([
                {"Indicator": k.replace("_", " "), "Value": v}
                for k, v in raw.items() if v is not None
            ])
            st.dataframe(
                raw_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Indicator": st.column_config.TextColumn("Indicator", width=250),
                    "Value": st.column_config.NumberColumn("Value", format="%.2f"),
                },
            )

        # ── Row 6: Alternative countries table ────────────────────────────────
        st.markdown('<div class="section-header">Top Alternative Countries</div>', unsafe_allow_html=True)
        alts = ctx.get("top_alternative_countries", [])
        if alts:
            alts_df = pd.DataFrame(alts)
            alts_df["opportunity_score"] = alts_df["opportunity_score"].round(1)
            st.dataframe(
                alts_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "country": "Country",
                    "opportunity_score": st.column_config.ProgressColumn(
                        "Opportunity Score", min_value=0, max_value=100, format="%.1f"
                    ),
                    "rank": "Global Rank",
                    "cluster_name": "Cluster",
                },
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: MAP & CLUSTERS
# ─────────────────────────────────────────────────────────────────────────────

with tab_map:
    df = st.session_state.df
    country = st.session_state.last_country
    sector = st.session_state.last_sector

    if df is None:
        st.info("💬 Start a conversation in the Chat tab to see maps here.")
    else:
        st.markdown(f"### 🗺️ Global Map — {sector.title()} Opportunity Scores")

        # World map
        fig_map = world_map(df, sector, target_country=country)
        st.plotly_chart(fig_map, use_container_width=True)

        st.divider()

        # Cluster scatter
        st.markdown("### 🔵 Country Clusters (PCA 2D)")
        st.markdown(
            "Countries are grouped into clusters using K-Means on 7 economic features. "
            "PCA reduces the 7D feature space to 2D for visualization."
        )
        fig_scatter = cluster_scatter(df, target_country=country)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Cluster details table
        st.divider()
        st.markdown("### 📋 Countries in Same Cluster")
        if country and "cluster_name" in df.columns:
            matched = fuzzy_find_country(country, df)
            if matched:
                target_cluster = df[df["country"] == matched]["cluster_name"].values[0]
                cluster_df = df[df["cluster_name"] == target_cluster][
                    ["country", "opportunity_score", "rank"] +
                    [f for f in FEATURE_COLS[:4] if f in df.columns]
                ].sort_values("opportunity_score", ascending=False)

                st.markdown(f"**Cluster: {target_cluster}** — {len(cluster_df)} countries")
                st.dataframe(
                    cluster_df.reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "opportunity_score": st.column_config.ProgressColumn(
                            "Score", min_value=0, max_value=100, format="%.1f"
                        ),
                        "rank": "Rank",
                    },
                )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

with tab_data:
    st.markdown("### 🔬 Data Mining Pipeline")
    st.markdown("This tab shows the data mining process used to analyze countries.")

    # Pipeline steps visualization
    steps = [
        ("1️⃣", "Data Loading", "Load CIA World Factbook CSV with pandas"),
        ("2️⃣", "Data Cleaning", "Extract numeric values, handle missing data (drop >60% missing, impute median)"),
        ("3️⃣", "Feature Engineering", "Create 7 composite scores: Market Size, Digital Penetration, Economic Stability, Growth Potential, Infrastructure, Trade Openness, Human Capital"),
        ("4️⃣", "Normalization", "MinMaxScaler → all features in [0, 1]"),
        ("5️⃣", "K-Means Clustering", f"Group countries into 5 clusters based on economic profiles"),
        ("6️⃣", "Opportunity Scoring", "Weighted sum of features tuned to your startup sector"),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div style="background:#1c1f26; border:1px solid #2a2d36; border-radius:10px;
                            padding:16px; margin-bottom:12px; min-height:120px;">
                    <div style="font-size:1.5rem">{icon}</div>
                    <div style="font-weight:600; color:#e0e0e0; margin:6px 0;">{title}</div>
                    <div style="font-size:0.85rem; color:#aaa;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Feature weights for current sector
    st.markdown(f"### ⚖️ Feature Weights — {st.session_state.last_sector.title()} Sector")
    from config import SECTOR_WEIGHTS
    weights = SECTOR_WEIGHTS.get(st.session_state.last_sector, SECTOR_WEIGHTS["general"])
    weight_df = pd.DataFrame([
        {"Feature": FEATURE_LABELS.get(k, k), "Weight": v * 100}
        for k, v in weights.items() if v > 0
    ]).sort_values("Weight", ascending=False)

    st.dataframe(
        weight_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Feature": "Feature",
            "Weight": st.column_config.ProgressColumn("Weight (%)", min_value=0, max_value=100, format="%.0f%%"),
        },
    )

    st.divider()

    # Full country dataset table
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown(f"### 📋 Full Country Dataset ({len(df)} countries)")

        # Filter controls
        col_search, col_cluster_filter = st.columns([2, 1])
        with col_search:
            search_term = st.text_input("🔍 Search country", placeholder="e.g. France")
        with col_cluster_filter:
            cluster_filter = st.selectbox(
                "Filter by cluster",
                ["All"] + sorted(df["cluster_name"].unique().tolist()),
            )

        display_df = df.copy()
        if search_term:
            display_df = display_df[display_df["country"].str.contains(search_term, case=False, na=False)]
        if cluster_filter != "All":
            display_df = display_df[display_df["cluster_name"] == cluster_filter]

        show_cols = ["country", "opportunity_score", "rank", "cluster_name"] + \
                    [f for f in FEATURE_COLS if f in display_df.columns]

        st.dataframe(
            display_df[show_cols].sort_values("rank").reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            column_config={
                "opportunity_score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=100, format="%.1f"
                ),
                "rank": "Rank",
                "cluster_name": "Cluster",
                **{f: st.column_config.NumberColumn(FEATURE_LABELS.get(f, f), format="%.3f")
                   for f in FEATURE_COLS if f in display_df.columns},
            },
        )
