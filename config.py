# config.py — Constants and configuration for Startup Country Advisor

import os
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ── Groq Model ──────────────────────────────────────────────────────────────
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen-qwq-32b")  # Free tier model on Groq
GROQ_MAX_TOKENS = 1024

# ── Dataset ──────────────────────────────────────────────────────────────────
DATA_PATH = "data/countries.csv"

# Raw column names from CIA World Factbook CSV (will be mapped to clean names)
# The Kaggle "countries.csv" uses "Section: Field" naming convention.
COLUMN_MAP = {
    "Country": "country",
    "Economy: Real GDP (purchasing power parity)": "gdp_ppp",
    "Economy: Real GDP per capita": "gdp_per_capita",
    "Economy: Real GDP growth rate": "gdp_growth",
    "Economy: Inflation rate (consumer prices)": "inflation",
    "Economy: Unemployment rate": "unemployment",
    "People and Society: Population - total": "population",
    "Communications: Internet users - percent of population": "internet_users_pct",
    "Economy: Taxes and other revenues": "tax_revenue",
    "Economy: Public debt": "public_debt",
    "Economy: Exports": "exports",
    "Economy: Imports": "imports",
    "Economy: Labor force": "labor_force",
    "Energy: Electricity - consumption": "electricity_production",
    "Communications: Telephones - mobile cellular - total subscriptions": "mobile_phones",
    "People and Society: Literacy - total population": "literacy",
}

# Numeric columns to extract float values from
NUMERIC_COLS = [
    "gdp_ppp", "gdp_per_capita", "gdp_growth", "inflation", "unemployment",
    "population", "internet_users_pct", "tax_revenue", "public_debt",
    "exports", "imports", "labor_force", "electricity_production",
    "mobile_phones", "literacy",
]

# Engineered feature columns
FEATURE_COLS = [
    "market_size_score",
    "digital_score",
    "economic_stability",
    "growth_potential",
    "infrastructure_score",
    "trade_openness",
    "human_capital",
]

FEATURE_LABELS = {
    "market_size_score": "Market Size",
    "digital_score": "Digital Penetration",
    "economic_stability": "Economic Stability",
    "growth_potential": "Growth Potential",
    "infrastructure_score": "Infrastructure",
    "trade_openness": "Trade Openness",
    "human_capital": "Human Capital",
}

# ── Clustering ────────────────────────────────────────────────────────────────
N_CLUSTERS = 5
CLUSTER_NAMES = {
    0: "Advanced Economies",
    1: "Emerging Markets",
    2: "Developing Economies",
    3: "Digital Leapfrog Markets",
    4: "Frontier Markets",
}

CLUSTER_COLORS = {
    "Advanced Economies": "#2196F3",
    "Emerging Markets": "#4CAF50",
    "Developing Economies": "#FF9800",
    "Digital Leapfrog Markets": "#9C27B0",
    "Frontier Markets": "#F44336",
}

# ── Sector Weights ────────────────────────────────────────────────────────────
SECTOR_WEIGHTS = {
    "fintech": {
        "digital_score": 0.30,
        "economic_stability": 0.25,
        "market_size_score": 0.20,
        "trade_openness": 0.15,
        "human_capital": 0.10,
        "growth_potential": 0.00,
        "infrastructure_score": 0.00,
    },
    "e-commerce": {
        "digital_score": 0.35,
        "market_size_score": 0.25,
        "infrastructure_score": 0.20,
        "trade_openness": 0.15,
        "human_capital": 0.05,
        "economic_stability": 0.00,
        "growth_potential": 0.00,
    },
    "edtech": {
        "human_capital": 0.30,
        "digital_score": 0.25,
        "market_size_score": 0.20,
        "economic_stability": 0.15,
        "growth_potential": 0.10,
        "infrastructure_score": 0.00,
        "trade_openness": 0.00,
    },
    "healthtech": {
        "human_capital": 0.25,
        "economic_stability": 0.25,
        "market_size_score": 0.20,
        "infrastructure_score": 0.20,
        "growth_potential": 0.10,
        "digital_score": 0.00,
        "trade_openness": 0.00,
    },
    "agritech": {
        "growth_potential": 0.30,
        "market_size_score": 0.25,
        "trade_openness": 0.20,
        "infrastructure_score": 0.15,
        "economic_stability": 0.10,
        "digital_score": 0.00,
        "human_capital": 0.00,
    },
    "saas": {
        "digital_score": 0.30,
        "human_capital": 0.25,
        "economic_stability": 0.20,
        "market_size_score": 0.15,
        "trade_openness": 0.10,
        "growth_potential": 0.00,
        "infrastructure_score": 0.00,
    },
    "logistics": {
        "infrastructure_score": 0.30,
        "trade_openness": 0.25,
        "market_size_score": 0.20,
        "economic_stability": 0.15,
        "growth_potential": 0.10,
        "digital_score": 0.00,
        "human_capital": 0.00,
    },
    "general": {
        "market_size_score": 0.20,
        "digital_score": 0.20,
        "economic_stability": 0.20,
        "growth_potential": 0.15,
        "human_capital": 0.15,
        "infrastructure_score": 0.05,
        "trade_openness": 0.05,
    },
}

SECTORS = list(SECTOR_WEIGHTS.keys())

# ── UI ────────────────────────────────────────────────────────────────────────
APP_TITLE = "🌍 Startup Market Advisor"
APP_SUBTITLE = "Votre guide intelligent pour choisir le bon marché"

GLOSSARY = {
    "Data Cleaning": "Removing/imputing missing values and standardizing data formats.",
    "Feature Engineering": "Creating composite scores (e.g. Market Size, Digital Penetration) from raw indicators.",
    "Normalization": "Scaling all features to [0, 1] so they can be compared fairly.",
    "K-Means Clustering": "Grouping countries with similar economic profiles into clusters.",
    "Opportunity Scoring": "Weighted sum of features tuned to your startup sector.",
    "PCA": "Principal Component Analysis — reducing 7D feature space to 2D for visualization.",
    "Percentile Rank": "Country's position relative to all countries (0 = worst, 100 = best).",
}
