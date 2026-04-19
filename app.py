# app.py — Streamlit Frontend for Startup Country Advisor

import json
import logging
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import plotly.express as px

# Load .env before importing config (config also loads it, but this ensures it)
load_dotenv()

from config import (
    APP_TITLE, APP_SUBTITLE, DATA_PATH, FEATURE_LABELS,
    FEATURE_COLS, SECTORS, GLOSSARY, CLUSTER_COLORS,
    SECTOR_WEIGHTS,
)
from data_pipeline import run_pipeline, clean_data, load_raw_data, detect_outliers_iqr
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
from ml_models import (
    select_features, create_target, get_task_labels,
    train_models, get_performance_summary,
    ModelManager, TASK_DEFINITIONS, MODEL_DESCRIPTIONS,
)
from seaborn_viz import (
    correlation_heatmap, feature_distributions, boxplot_outliers,
    pairplot_features, confusion_matrix_plot, feature_importance_plot,
    cv_scores_plot, feature_selection_plot,
    roc_curve_plot, model_comparison_plot,
    opportunity_score_distribution, top_bottom_countries_plot,
    cluster_composition_plot, violin_score_by_cluster,
)
from results_engine import build_llm_context

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
    /* Global background white, text always black */
    .stApp { background-color: #FFFFFF; color: #000000; }
    p, span, h1, h2, h3, h4, h5, h6, label, div { color: #000000; }
    
    /* ═══ TOP NAVIGATION BAR ═══ */
    /* Remove any background color from the nav container so it stays white */
    [data-testid="stMainBlockContainer"] > div > div > div > [data-testid="stVerticalBlockBorderWrapper"]:first-child [data-testid="stHorizontalBlock"] {
        padding: 12px 24px;
        margin-bottom: 8px;
    }
    [data-testid="stMainBlockContainer"] > div > div > div > [data-testid="stVerticalBlockBorderWrapper"]:first-child [data-testid="stHorizontalBlock"] h3,
    [data-testid="stMainBlockContainer"] > div > div > div > [data-testid="stVerticalBlockBorderWrapper"]:first-child [data-testid="stHorizontalBlock"] span,
    [data-testid="stMainBlockContainer"] > div > div > div > [data-testid="stVerticalBlockBorderWrapper"]:first-child [data-testid="stHorizontalBlock"] p,
    [data-testid="stMainBlockContainer"] > div > div > div > [data-testid="stVerticalBlockBorderWrapper"]:first-child [data-testid="stHorizontalBlock"] div,
    [data-testid="stMainBlockContainer"] > div > div > div > [data-testid="stVerticalBlockBorderWrapper"]:first-child [data-testid="stHorizontalBlock"] label {
        color: #000000 !important;
    }

    /* Sections style cards */
    [data-testid="metric-container"], 
    div.stChatMessage, 
    div[data-testid="stExpanderDetails"] > div,
    .stChatInput > div,
    [data-testid="stForm"],
    .stDataFrame,
    [data-testid="stTable"] {
        background-color: #f5f5f5 !important;
        border-radius: 8px;
    }
    
    /* Chat input styling */
    .stChatInput > div { border: 1px solid #ccc; }
    
    /* Metric cards border */
    [data-testid="metric-container"] {
        border: 1px solid #ddd;
        padding: 12px;
    }
    
    /* Navigation radio buttons — horizontal, no circles */
    [data-testid="stRadio"] > div {
        flex-direction: row;
        align-items: center;
        gap: 20px;
        background-color: transparent;
    }
    
    /* Hide radio circles (boules) */
    [data-testid="stRadio"] label span[data-testid="stMarkdownContainer"] ~ div,
    [data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }

    /* Radio labels base style — BLACK text */
    [data-testid="stRadio"] label {
        font-weight: normal !important;
        border-bottom: 2px solid transparent !important;
        padding-bottom: 3px !important;
        cursor: pointer;
        color: #000000 !important;
    }
    [data-testid="stRadio"] label p {
        font-weight: normal !important;
        font-size: 1rem !important;
        color: #000000 !important;
    }
    
    /* Radio labels hover */
    [data-testid="stRadio"] label:hover {
        border-bottom: 2px solid #305CDE !important;
    }
    [data-testid="stRadio"] label:hover p {
        font-weight: bold !important;
        color: #305CDE !important;
    }

    /* Radio labels selected */
    [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
        border-bottom: 2px solid #305CDE !important;
    }
    [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] p {
        font-weight: bold !important;
        color: #305CDE !important;
    }
    
    /* Charts border */
    [data-testid="stPlotlyChart"],
    [data-testid="stImage"] > img {
        border: 1px solid #000000;
        border-radius: 4px;
        padding: 4px;
    }
    
    /* Glossary scrollable area */
    .glossary-scroll {
        max-height: 250px;
        overflow-y: auto;
        padding-right: 10px;
    }

    /* Glossary term */
    .glossary-term {
        background: #f5f5f5;
        border-bottom: 1px solid #ccc;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.85rem;
        color: #000000;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #000000;
        border-left: 4px solid #305CDE;
        padding-left: 10px;
        margin: 16px 0 10px 0;
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
        "ml_results": None,        # ML model training results
        "ml_manager": None,        # ModelManager instance
        "outlier_info": None,      # Outlier detection results
        "feature_sel": None,       # Feature selection results
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
# ─────────────────────────────────────────────────────────────────────────────
# TOP NAVIGATION BAR (Replaces Sidebar & Tabs)
# ─────────────────────────────────────────────────────────────────────────────

# --- Header & Nav row ---
nav_container = st.container()
with nav_container:
    col_logo, col_nav, col_settings = st.columns([3, 6, 2], vertical_alignment="center")
    
    with col_logo:
        st.markdown(f"### {APP_TITLE}")
        st.markdown(f"<span style='color:#A3BFF5; font-size:0.9rem;'>{APP_SUBTITLE}</span>", unsafe_allow_html=True)
    
    with col_nav:
        pages = [
            "Accueil", "AI StartUp Advisor", "Exploration",
            "Modèles Machine Learning", "Données"
        ]
        selected_page = st.radio(
            "Navigation",
            pages,
            horizontal=True,
            label_visibility="collapsed",
        )
    
    with col_settings:
        with st.expander("📖 Glossaire"):
            glossary_html = '<div class="glossary-scroll">'
            for term, definition in GLOSSARY.items():
                glossary_html += f'<div class="glossary-term"><b>{term}:</b> {definition}</div>'
            glossary_html += '</div>'
            st.markdown(glossary_html, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE ROUTING
# ─────────────────────────────────────────────────────────────────────────────

if selected_page == "AI StartUp Advisor":
    st.markdown("### 💬 AI StartUp Advisor")
    st.markdown(
        "Décrivez votre startup et le pays dans lequel vous souhaitez vous lancer. "
        "J'exécuterai une analyse complète de data mining en utilisant les données du CIA World Factbook."
    )

    # Example prompts
    with st.expander("💡 Exemples de prompts"):
        examples = [
            "Je veux lancer une startup fintech en Tunisie",
            "Ma plateforme edtech cible les étudiants au Nigeria",
            "Nous développons un SaaS logistique pour le marché de l'Asie du Sud-Est, en commençant par le Vietnam",
            "J'ai une startup healthtech pour les zones rurales — quel pays africain est le meilleur ?",
            "Comparez le Maroc et l'Égypte pour une entreprise de commerce électronique",
        ]
        for ex in examples:
            if st.button(f"→ {ex}", key=f"ex_{ex[:20]}"):
                # Clicking an example prompt sends it directly as user input
                st.session_state["_pending_prompt"] = ex
                st.rerun()

    st.divider()

    # Chat history display
    chat_container = st.container(height=480)
    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                """
                <div style='text-align:center; color:#475569; padding:80px 20px;'>
                    <div style='font-size:3rem'>🌍</div>
                    <div style='font-size:1.1rem; margin-top:10px; color:#1E293B; font-weight:600;'>
                        Parlez-moi de votre startup et du pays ciblé.
                    </div>
                    <div style='font-size:0.9rem; color:#64748B; margin-top:8px;'>
                        Exemple : "Je veux lancer une startup fintech en Tunisie"
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
    user_input = st.chat_input("Décrivez votre startup et le pays ciblé...")

    # Handle pending prompt from example buttons
    _from_example = False
    if st.session_state.get("_pending_prompt"):
        user_input = st.session_state.pop("_pending_prompt")
        st.session_state.messages.append({"role": "user", "content": user_input})
        _from_example = True

    if user_input:
        # Validate prerequisites
        if not st.session_state.groq_api_key:
            st.error("⚠️ Please enter your Groq API key in the sidebar first.")
            st.stop()

        if not check_data_file():
            st.error("⚠️ Dataset not found. Please download `factbook.csv` and place it in `data/`.")
            st.stop()

        # Add user message to history (only if not already added from example)
        if not _from_example:
            st.session_state.messages.append({"role": "user", "content": user_input})

        # Determine if this is a new analysis or a follow-up
        is_new_analysis = (
            st.session_state.analysis_context is None
            or any(kw in user_input.lower() for kw in [
                "launch", "start", "compare", "analyze", "what about", "tell me about", "consider"
            ])
        )

        with st.spinner("Analyzing..."):
            try:
                if is_new_analysis:
                    # ── Parse intent ──────────────────────────────────────────
                    intent = parse_intent(user_input, st.session_state.groq_api_key)
                    sector = intent.get("sector", "general")
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

        # Clean <think>...</think> tags and their content from the response
        import re as _re
        response = _re.sub(r'<think>.*?</think>', '', response, flags=_re.DOTALL).strip()
        # Bold all percentages (e.g. 45.2%, 100%)
        response = _re.sub(r'(\d+\.?\d*%)', r'**\1**', response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

elif selected_page == "Accueil":
    st.markdown("### 🌍 Accueil - Dashboard Startup Global")
    
    # ── SAFE DATA LOADING ──
    if st.session_state.df is None:
        if not check_data_file():
            st.error("⚠️ Dataset `factbook.csv` introuvable.")
            st.stop()
        from data_pipeline import run_pipeline
        st.session_state.df = run_pipeline(sector="general", data_path=DATA_PATH)["df"]
        st.session_state.last_sector = "general"
        country_list = st.session_state.df["country"].sort_values().tolist()
        st.session_state.last_country = "France" if "France" in country_list else country_list[0]
        
    df = st.session_state.df
    
    # Selection Controls
    st.markdown('<div class="section-header">Paramètres de Visualisation</div>', unsafe_allow_html=True)
    col_sel_country, col_sel_sector = st.columns(2)
    with col_sel_country:
        country_list = df["country"].sort_values().tolist()
        current_country = st.session_state.last_country if st.session_state.last_country in country_list else country_list[0]
        selected_country = st.selectbox("Sélectionnez un pays cible :", country_list, index=country_list.index(current_country))
    
    with col_sel_sector:
        current_sector = st.session_state.last_sector if st.session_state.last_sector in SECTORS else "general"
        selected_sector = st.selectbox("Secteur d'activité :", SECTORS, index=SECTORS.index(current_sector))
        
    # Recalculate if changed or context missing
    if selected_country != st.session_state.last_country or selected_sector != st.session_state.last_sector or st.session_state.analysis_context is None:
        with st.spinner("Mise à jour des données..."):
            from data_pipeline import run_pipeline
            st.session_state.last_country = selected_country
            st.session_state.last_sector = selected_sector
            st.session_state.df = run_pipeline(sector=selected_sector, data_path=DATA_PATH)["df"]
            df = st.session_state.df
            
            
            st.session_state.analysis_context = build_llm_context(
                target_country=selected_country,
                sector=selected_sector,
                startup_description="Exploration du dataset",
                df=df,
                n_alternatives=5
            )
        st.rerun()

    ctx = st.session_state.analysis_context
    country = selected_country
    sector = selected_sector
    
    if "error" in ctx:
        st.error(ctx["error"])
    else:
        st.divider()
        st.markdown(f"### 📊 Aperçu des Performances : **{country}** — Secteur {sector.title()}")

        # ── Row 1: Key metrics ────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Score d'Opportunité", f"{ctx['opportunity_score']:.1f}/100")
        with col2:
            st.metric("🏆 Rang Mondial", f"#{ctx['global_rank']} / {ctx['total_countries']}")
        with col3:
            st.metric("🌐 Cluster", ctx.get("cluster", "—"))
        with col4:
            pct_rank = round((1 - ctx["global_rank"] / ctx["total_countries"]) * 100, 1)
            st.metric("📈 Meilleur Que", f"{pct_rank}% des pays")

        st.divider()

        # ── World Map ─────────────────────────────────────────────────────────
        st.markdown(f"### Carte Mondiale — Scores d'Opportunité ({sector.title()})")
        fig_map = world_map(df, sector, target_country=country)
        st.plotly_chart(fig_map, use_container_width=True)

        st.divider()

        # ── Cluster Scatter ───────────────────────────────────────────────────
        st.markdown("### Clusters de Pays (PCA 2D)")
        st.markdown(
            "Les pays sont regroupés en clusters avec K-Means sur 7 caractéristiques. "
            "La PCA réduit l'espace 7D en 2D pour la visualisation."
        )
        fig_scatter = cluster_scatter(df, target_country=country)
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.divider()

        # ── Gauge + Radar ─────────────────────────────────────────────────────
        col_g, col_r = st.columns([1, 1.6])
        with col_g:
            st.markdown('<div class="section-header">Score d\'Opportunité</div>', unsafe_allow_html=True)
            fig_gauge = opportunity_gauge(
                ctx["opportunity_score"], country,
                ctx["global_rank"], ctx["total_countries"]
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_r:
            st.markdown('<div class="section-header">Profil des Caractéristiques (Rang Percentile)</div>', unsafe_allow_html=True)
            percentiles = get_feature_percentiles(country, df)
            alt_pcts = []
            alts = ctx.get("top_alternative_countries", [])
            if alts:
                top_alt = alts[0]["country"]
                top_alt_pcts = get_feature_percentiles(top_alt, df)
                alt_pcts = [(top_alt, top_alt_pcts)]
            fig_radar = radar_chart(country, percentiles, alt_pcts)
            st.plotly_chart(fig_radar, use_container_width=True)

        st.divider()

        # ── Top 10 Countries Bar (light blue) ──────────────────────────────────
        st.markdown('<div class="section-header">Meilleurs Pays par Score d\'Opportunité</div>', unsafe_allow_html=True)
        fig_bar = top_countries_bar(df, sector, target_country=country, n=10)
        # Override colors to light blue
        fig_bar.update_traces(marker_color="#87CEFA", selector=dict(type="bar"))
        # Re-highlight target country in darker blue
        top10 = df.sort_values("opportunity_score", ascending=False).head(10)
        bar_colors = ["#305CDE" if c == country else "#87CEFA" for c in top10["country"]]
        fig_bar.update_traces(marker_color=bar_colors)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # ── Feature comparison heatmap ────────────────────────────────────────
        st.markdown('<div class="section-header">Comparaison — Cible vs Alternatives</div>', unsafe_allow_html=True)
        from results_engine import fuzzy_find_country
        alt_countries = [a["country"] for a in ctx.get("top_alternative_countries", [])[:4]]
        compare_list = [country] + alt_countries
        compare_list = [c for c in compare_list if fuzzy_find_country(c, df)]
        if len(compare_list) >= 2:
            fig_heat = feature_heatmap(compare_list, df)
            st.plotly_chart(fig_heat, use_container_width=True)

        st.divider()

        # ── Raw indicators (full width, single row of metrics) ─────────────────
        st.markdown('<div class="section-header">Indicateurs Économiques Bruts</div>', unsafe_allow_html=True)
        raw = ctx.get("raw_indicators", {})
        if raw:
            keys = [k for k in raw.keys() if raw[k] is not None]
            n_keys = len(keys)
            if n_keys > 0:
                metric_cols = st.columns(min(n_keys, 6))
                for idx, k in enumerate(keys[:6]):
                    v = raw[k]
                    metric_cols[idx % min(n_keys, 6)].metric(k.replace("_", " ").title(), f"{v:,.2f}")
                # If more than 6, add a second row
                if n_keys > 6:
                    metric_cols2 = st.columns(min(n_keys - 6, 6))
                    for idx, k in enumerate(keys[6:12]):
                        v = raw[k]
                        metric_cols2[idx].metric(k.replace("_", " ").title(), f"{v:,.2f}")

        st.divider()

        # ── Meilleures Alternatives (full width, below indicators) ────────────
        st.markdown('<div class="section-header">Meilleures Alternatives</div>', unsafe_allow_html=True)
        alts = ctx.get("top_alternative_countries", [])
        if alts:
            alts_df = pd.DataFrame(alts).head(5)
            fig_alts = px.bar(
                alts_df, x="opportunity_score", y="country", orientation="h",
                text="opportunity_score", color_discrete_sequence=["#6C9CFF"]
            )
            fig_alts.update_layout(
                yaxis=dict(autorange="reversed", title=""),
                xaxis=dict(title="Score", range=[0, 100]),
                margin=dict(l=0, r=0, t=0, b=0),
                height=280,
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font=dict(color="#000000")
            )
            fig_alts.update_traces(texttemplate='%{text:.1f}', textposition='inside', marker_line_color='black', marker_line_width=1)
            st.plotly_chart(fig_alts, use_container_width=True)

        st.divider()

        # ── Cluster Bar Chart ─────────────────────────────────────────────────
        st.markdown('<div class="section-header">Groupe de pays de référence</div>', unsafe_allow_html=True)
        if country and "cluster_name" in df.columns:
            matched = fuzzy_find_country(country, df)
            if matched:
                target_cluster = df[df["country"] == matched]["cluster_name"].values[0]
                cluster_df = df[df["cluster_name"] == target_cluster]
                if len(cluster_df) > 0:
                    fig_cluster_bar = top_countries_bar(cluster_df, sector, target_country=country, n=10)
                    fig_cluster_bar.update_layout(title=f"Top 10 Pays — Cluster: {target_cluster}")
                    fig_cluster_bar.update_traces(marker_color="#87CEFA", selector=dict(type="bar"))
                    st.plotly_chart(fig_cluster_bar, use_container_width=True)



# ─────────────────────────────────────────────────────────────────────────────
# TAB 3b: EXPLORATION (SEABORN)
# ─────────────────────────────────────────────────────────────────────────────

elif selected_page == "Exploration":
    df = st.session_state.df

    # ─────────────────────────────────────────────────────────────
    # SAFE LOADING (indépendant du Chat)
    # ─────────────────────────────────────────────────────────────
    if df is None:
        st.info("⚠️ Dataset non chargé. Chargement automatique en cours...")

        from data_pipeline import run_pipeline
        df = run_pipeline(sector="general", data_path=DATA_PATH)["df"]
        st.session_state.df = df

    # ─────────────────────────────────────────────────────────────
    # TITLE
    # ─────────────────────────────────────────────────────────────
    st.markdown("### Exploration des Données avec Seaborn")
    st.markdown(
        "Analyse exploratoire des données : distributions, corrélations, outliers et pairplots."
    )

    # ─────────────────────────────────────────────────────────────
    # 1. DISTRIBUTIONS
    # ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Distribution des Features</div>', unsafe_allow_html=True)

    fig_dist = feature_distributions(df)
    st.pyplot(fig_dist)
    plt.close(fig_dist)

    st.divider()

    # ─────────────────────────────────────────────────────────────
    # 2. CORRELATION
    # ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Matrice de Corrélation</div>', unsafe_allow_html=True)

    fig_corr = correlation_heatmap(df)
    st.pyplot(fig_corr)
    plt.close(fig_corr)

    st.divider()

    # ─────────────────────────────────────────────────────────────
    # 3. OUTLIERS
    # ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Détection des Outliers (IQR)</div>', unsafe_allow_html=True)

    fig_box = boxplot_outliers(df)
    st.pyplot(fig_box)
    plt.close(fig_box)

    outlier_info = detect_outliers_iqr(df)
    st.session_state.outlier_info = outlier_info

    outlier_rows = [
        {
            "Feature": FEATURE_LABELS.get(feat, feat),
            "Nb Outliers": info["n_outliers"],
            "% Outliers": info["pct_outliers"],
            "Q1": info["Q1"],
            "Q3": info["Q3"],
            "IQR": info["IQR"],
            "Lower Bound": info["lower_bound"],
            "Upper Bound": info["upper_bound"],
        }
        for feat, info in outlier_info.items()
    ]

    st.dataframe(
        pd.DataFrame(outlier_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ─────────────────────────────────────────────────────────────
    # 4. OPPORTUNITY SCORE DISTRIBUTION
    # ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Distribution des Scores d\'Opportunité</div>', unsafe_allow_html=True)

    fig_opp = opportunity_score_distribution(df)
    st.pyplot(fig_opp)
    plt.close(fig_opp)

    st.divider()

    # ─────────────────────────────────────────────────────────────
    # 5. TOP vs BOTTOM COUNTRIES
    # ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏆 Top 10 vs Bottom 10 Pays</div>', unsafe_allow_html=True)
    st.markdown("Comparaison directe des meilleurs et des pires pays selon le score d'opportunité.")

    fig_tb = top_bottom_countries_plot(df, n=10)
    st.pyplot(fig_tb)
    plt.close(fig_tb)

    st.divider()

    # ─────────────────────────────────────────────────────────────
    # 6. CLUSTER COMPOSITION
    # ─────────────────────────────────────────────────────────────
    col_pie, col_violin = st.columns(2)
    with col_pie:
        st.markdown('<div class="section-header">Répartition des Clusters</div>', unsafe_allow_html=True)
        fig_pie = cluster_composition_plot(df)
        st.pyplot(fig_pie)
        plt.close(fig_pie)

    with col_violin:
        st.markdown('<div class="section-header">Scores par Cluster (Violin)</div>', unsafe_allow_html=True)
        fig_vio = violin_score_by_cluster(df)
        st.pyplot(fig_vio)
        plt.close(fig_vio)

    st.divider()

    # ─────────────────────────────────────────────────────────────
    # 7. PAIRPLOT
    # ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Pairplot des Features</div>', unsafe_allow_html=True)
    st.markdown("Analyse multivariée sur les principales features.")

    fig_pair = pairplot_features(df, max_features=4)
    st.pyplot(fig_pair)
    plt.close(fig_pair)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: ML MODELS (Advanced — ModelManager)
# ─────────────────────────────────────────────────────────────────────────────

elif selected_page == "Modèles Machine Learning":
    df = st.session_state.df

    # ── Safe data loading ─────────────────────────────────────────────────
    if df is None:
        st.info("⚠️ Dataset non chargé. Chargement automatique...")
        df = run_pipeline(sector="general", data_path=DATA_PATH)["df"]
        st.session_state.df = df

    st.markdown("### Apprentissage Supervisé — Model Manager")
    st.markdown(
        "Architecture ML avancée avec **3 tâches de classification**, "
        "**3 modèles** (Random Forest, SVM, XGBoost), "
        "**GridSearchCV (CV=4)** et sélection automatique du meilleur modèle."
    )

    # ── Task Selection ────────────────────────────────────────────────────
    task_key = st.selectbox(
        "Tâche de classification",
        options=list(TASK_DEFINITIONS.keys()),
        format_func=lambda t: f"{TASK_DEFINITIONS[t]['name']} ({len(TASK_DEFINITIONS[t]['classes'])} classes)",
        key="ml_task_select",
    )

    # Show task description
    task_info = TASK_DEFINITIONS[task_key]
    st.info(f"**{task_info['name']}** — {task_info['description']}  \nClasses : `{', '.join(task_info['classes'])}`")

    # ── Prepare data ──────────────────────────────────────────────────────
    available = [f for f in FEATURE_COLS if f in df.columns]
    X = df[available].fillna(0)
    y = create_target(df, task=task_key)

    # ── Feature Selection ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Sélection des Caractéristiques (ANOVA F-test)</div>', unsafe_allow_html=True)
    feat_sel = select_features(X, y, k=5)
    st.session_state.feature_sel = feat_sel

    fig_sel = feature_selection_plot(feat_sel["scores"], feat_sel["pvalues"])
    st.pyplot(fig_sel)
    plt.close(fig_sel)

    sel_rows = []
    for feat, score in feat_sel["feature_ranking"]:
        sel_rows.append({
            "Feature": FEATURE_LABELS.get(feat, feat),
            "F-Score": round(score, 2),
            "p-value": f"{feat_sel['pvalues'].get(feat, 1):.4f}",
            "Sélectionné": "✅" if feat in feat_sel["selected_features"] else "❌",
        })
    st.dataframe(pd.DataFrame(sel_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Train Models (ModelManager) ───────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ Entraînement des Modèles (GridSearchCV, CV=4)</div>', unsafe_allow_html=True)

    if st.button("Lancer l'entraînement des modèles", type="primary"):
        with st.spinner(f"Entraînement en cours — Tâche : {task_info['name']}..."):
            mgr = ModelManager(X, y, task=task_key, cv=4)
            mgr.run_feature_selection(k=5)
            mgr.train_all()
            st.session_state.ml_results = mgr.results
            st.session_state.ml_manager = mgr
        st.success("✅ Entraînement terminé !")

    results = st.session_state.ml_results
    mgr = st.session_state.get("ml_manager", None)

    if results is not None and mgr is not None:
        # ── Best Model Auto-Selection ─────────────────────────────────────
        best_info = mgr.get_best_model(metric="auto")
        auto_metric = best_info["metric"].upper()
        
    

        st.divider()

        # ── Model Descriptions ────────────────────────────────────────────
        for name, res in results.items():
            desc = res.get("description", {})
            icon = {"Random Forest": "🌲", "SVM (SVC)": "⚙️", "XGBoost": "🚀"}.get(name, "📖")
            with st.expander(f"{icon} Description du modèle : {name}", expanded=False):
                st.markdown(f"**Définition :** {desc.get('definition', 'N/A')}")
                st.markdown("**Étapes :**")
                for step in desc.get("steps", []):
                    st.markdown(f"- {step}")
                st.markdown(f"**Paramètres optimaux :** `{res['best_params']}`")

        st.divider()

        # ── Confusion Matrices ────────────────────────────────────────────
        st.markdown('<div class="section-header">Matrices de Confusion</div>', unsafe_allow_html=True)
        cm_cols = st.columns(len(results))
        for i, (name, res) in enumerate(results.items()):
            with cm_cols[i]:
                fig_cm = confusion_matrix_plot(
                    res["confusion_matrix"], res["labels"],
                    title=f"Confusion — {name}",
                )
                st.pyplot(fig_cm)
                plt.close(fig_cm)

        st.divider()

        # ── ROC Curves ────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Courbes ROC</div>', unsafe_allow_html=True)
        roc_data = mgr.get_roc_data()
        if roc_data:
            fig_roc = roc_curve_plot(roc_data)
            st.pyplot(fig_roc)
            plt.close(fig_roc)

            # AUC summary
            auc_cols = st.columns(len(roc_data))
            for i, (name, data) in enumerate(roc_data.items()):
                with auc_cols[i]:
                    st.metric(f"AUC — {name}", f"{data['auc']:.3f}")
        else:
            st.warning("Les courbes ROC nécessitent des probabilités (SVM sans probability=True).")

        st.divider()

        # ── Model Comparison ──────────────────────────────────────────────
        st.markdown('<div class="section-header">Comparaison des Modèles</div>', unsafe_allow_html=True)
        fig_comp = model_comparison_plot(results)
        st.pyplot(fig_comp)
        plt.close(fig_comp)

        st.divider()

        # ── Cross-Validation Comparison ───────────────────────────────────
        st.markdown('<div class="section-header">Scores de Validation Croisée (CV=4)</div>', unsafe_allow_html=True)
        fig_cv = cv_scores_plot(results)
        st.pyplot(fig_cv)
        plt.close(fig_cv)

        st.divider()

        # ── Feature Importance ────────────────────────────────────────────
        st.markdown('<div class="section-header">Importance des Features</div>', unsafe_allow_html=True)
        imp_models = [n for n, r in results.items() if r.get("feature_importances")]
        if imp_models:
            imp_cols = st.columns(len(imp_models))
            for i, name in enumerate(imp_models):
                with imp_cols[i]:
                    fig_imp = feature_importance_plot(
                        results[name]["feature_importances"],
                        title=f"Importance — {name}",
                    )
                    st.pyplot(fig_imp)
                    plt.close(fig_imp)
        else:
            st.info("Aucun modèle ne fournit d'importance des features (SVM n'en a pas).")

        st.divider()

        # ── Performance Summary Table ─────────────────────────────────────
        st.markdown('<div class="section-header">Tableau Récapitulatif des Performances</div>', unsafe_allow_html=True)
        summary_df = mgr.get_performance_summary()

        # Display labels with icons
        summary_df["Model"] = summary_df["Model"].replace({
            "Random Forest": "🌲 Random Forest",
            "SVM (SVC)": "⚙️ SVM",
            "XGBoost": "🚀 XGBoost",
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.success(
            f"🏆 **Le modèle gagnant est {best_info['name']}**, "
            f"sélectionné automatiquement via la métrique d'évaluation la plus adaptée "
            f"à cette tâche ({auto_metric}), avec un score de **{best_info['score']:.4f}**."
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

elif selected_page == "Données":
    st.markdown("### Pipeline de Data Mining")
    st.markdown("Cet onglet montre le processus de data mining utilisé pour analyser les pays.")

    # Pipeline steps visualization
    steps = [
        ("1️⃣", "Chargement", "Chargement du CSV CIA World Factbook via pandas"),
        ("2️⃣", "Nettoyage", "Extraction des valeurs, gestion des manquants (suppression >60%, imputation médiane)"),
        ("3️⃣", "Ingénierie des features", "Création de 7 scores composites : Taille Marché, Numérique, Stabilité, etc."),
        ("4️⃣", "Normalisation", "MinMaxScaler → toutes les caractéristiques dans [0, 1]"),
        ("5️⃣", "Clustering K-Means", f"Regroupement des pays en clusters selon leurs profils économiques"),
        ("6️⃣", "Scoring", "Somme pondérée des caractéristiques ajustée au secteur ciblé"),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div style="background:#f5f5f5; border:1px solid #ccc; border-radius:10px;
                            padding:16px; margin-bottom:12px; min-height:120px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                    <div style="font-size:1.5rem">{icon}</div>
                    <div style="font-weight:600; color:#1E293B; margin:6px 0;">{title}</div>
                    <div style="font-size:0.85rem; color:#475569;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Feature weights for current sector
    st.markdown(f"### Poids des Caractéristiques — Secteur {st.session_state.last_sector.title()}")
    weights = SECTOR_WEIGHTS.get(st.session_state.last_sector, SECTOR_WEIGHTS["general"])
    weight_df = pd.DataFrame([
        {"Caractéristique": FEATURE_LABELS.get(k, k), "Poids": v * 100}
        for k, v in weights.items() if v > 0
    ]).sort_values("Weight", ascending=False)

    st.dataframe(
        weight_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Caractéristique": "Caractéristique",
            "Poids": st.column_config.ProgressColumn("Poids (%)", min_value=0, max_value=100, format="%.0f%%"),
        },
    )

    st.divider()

    # Full country dataset table
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown(f"### Jeu de Données Complet ({len(df)} pays)")

        # Filter controls
        col_search, col_cluster_filter = st.columns([2, 1])
        with col_search:
            search_term = st.text_input("Chercher un pays", placeholder="ex. France")
        with col_cluster_filter:
            cluster_filter = st.selectbox(
                "Filtrer par cluster",
                ["Tous"] + sorted(df["cluster_name"].unique().tolist()),
            )

        display_df = df.copy()
        if search_term:
            display_df = display_df[display_df["country"].str.contains(search_term, case=False, na=False)]
        if cluster_filter != "Tous":
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
                "rank": "Rang",
                "cluster_name": "Cluster",
                **{f: st.column_config.NumberColumn(FEATURE_LABELS.get(f, f), format="%.3f")
                   for f in FEATURE_COLS if f in display_df.columns},
            },
        )
