# data_pipeline.py — Full Data Mining Pipeline
# Steps: Load → Clean → Feature Engineering → Normalize → Cluster → Score

import re
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from config import (
    DATA_PATH, COLUMN_MAP, NUMERIC_COLS, FEATURE_COLS,
    N_CLUSTERS, CLUSTER_NAMES, SECTOR_WEIGHTS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the CIA World Factbook CSV."""
    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)

    logger.info(f"[LOAD] Raw shape: {df.shape}")
    logger.info(f"[LOAD] Columns: {list(df.columns[:10])}...")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def extract_numeric(value) -> float:
    """
    Extract the first numeric value from a messy string.
    Handles: '$1.2 trillion', '3.5%', '1,234,567', '2.3 (2020 est.)', etc.
    Returns NaN if no numeric found.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip()

    # Handle trillion / billion / million abbreviations
    multipliers = {"trillion": 1e12, "billion": 1e9, "million": 1e6, "thousand": 1e3}
    s_lower = s.lower()
    multiplier = 1.0
    for word, mult in multipliers.items():
        if word in s_lower:
            multiplier = mult
            break

    # Extract first number (possibly with decimal)
    match = re.search(r"[-+]?\d[\d,]*\.?\d*", s)
    if match:
        num_str = match.group().replace(",", "")
        try:
            return float(num_str) * multiplier
        except ValueError:
            return np.nan
    return np.nan


def select_and_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select columns using the COLUMN_MAP from config.
    Falls back to keyword-based detection if exact names are not found.
    """
    # Try exact match first via COLUMN_MAP
    found = {}
    for raw_name, clean_name in COLUMN_MAP.items():
        if raw_name in df.columns:
            found[raw_name] = clean_name

    # For any missing columns, try keyword-based detection
    keyword_map = {
        "country": ["country", "name"],
        "gdp_ppp": ["purchasing power parity", "gdp (ppp)", "gdp ppp", "real gdp (purchasing"],
        "gdp_per_capita": ["per capita", "gdp per capita", "real gdp per capita"],
        "gdp_growth": ["real growth rate", "gdp growth", "real gdp growth"],
        "inflation": ["inflation rate", "consumer prices"],
        "unemployment": ["unemployment rate"],
        "population": ["population - total", "population"],
        "internet_users_pct": ["internet users - percent", "internet users"],
        "tax_revenue": ["taxes and other revenues", "tax revenue"],
        "public_debt": ["public debt"],
        "exports": ["economy: exports"],
        "imports": ["economy: imports"],
        "labor_force": ["labor force"],
        "electricity_production": ["electricity - consumption", "electricity - production", "electricity production"],
        "mobile_phones": ["mobile cellular - total", "mobile cellular"],
        "literacy": ["literacy - total population", "literacy"],
    }

    found_clean_names = set(found.values())
    for clean_name, keywords in keyword_map.items():
        if clean_name in found_clean_names:
            continue  # already found via exact match
        for col in df.columns:
            col_lower = col.lower()
            if any(kw.lower() in col_lower for kw in keywords):
                # Avoid columns that have "note" or "partner" in them
                if "note" in col_lower or "partner" in col_lower or "commodit" in col_lower:
                    continue
                found[col] = clean_name
                found_clean_names.add(clean_name)
                break

    if not found:
        raise ValueError("Could not find any matching columns in the dataset.")

    df2 = df[list(found.keys())].rename(columns=found)
    logger.info(f"[SELECT] Retained columns: {list(df2.columns)}")
    return df2


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full data cleaning:
    1. Select and rename columns
    2. Convert numeric strings to float
    3. Drop columns with >60% missing
    4. Impute remaining missing with median
    5. Remove rows with null country
    6. Standardize country names
    """
    logger.info("[CLEAN] Starting data cleaning...")

    df = select_and_rename_columns(df)

    # Drop rows with no country
    if "country" in df.columns:
        df = df.dropna(subset=["country"])
        df["country"] = df["country"].str.strip().str.title()
    else:
        raise ValueError("No 'country' column found in dataset.")

    # Convert numeric columns
    numeric_cols_present = [c for c in NUMERIC_COLS if c in df.columns]
    for col in numeric_cols_present:
        df[col] = df[col].apply(extract_numeric)

    # Drop columns with >60% missing
    threshold = 0.60
    missing_frac = df[numeric_cols_present].isna().mean()
    cols_to_drop = missing_frac[missing_frac > threshold].index.tolist()
    if cols_to_drop:
        logger.info(f"[CLEAN] Dropping high-missing columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Impute remaining missing with median
    numeric_cols_present = [c for c in NUMERIC_COLS if c in df.columns]
    for col in numeric_cols_present:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Reset index
    df = df.reset_index(drop=True)

    logger.info(f"[CLEAN] Clean shape: {df.shape}, Countries: {len(df)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 7 composite features for startup opportunity analysis.
    All raw columns must be numeric at this point.
    """
    logger.info("[FEATURES] Engineering composite features...")

    eps = 1e-9  # avoid log(0)

    def safe_log(series):
        return np.log1p(series.clip(lower=0))

    # 1. Market Size Score: log(population) * log(gdp_ppp)
    if "population" in df.columns and "gdp_ppp" in df.columns:
        df["market_size_score"] = safe_log(df["population"]) * safe_log(df["gdp_ppp"])
    else:
        df["market_size_score"] = safe_log(df.get("population", pd.Series([0] * len(df))))

    # 2. Digital Penetration:
    #    If we have internet_users_pct (already a percentage), use directly.
    #    Otherwise fall back to internet_users / population * 100.
    if "internet_users_pct" in df.columns:
        df["digital_score"] = df["internet_users_pct"].clip(0, 100)
    elif "internet_users" in df.columns and "population" in df.columns:
        df["digital_score"] = (df["internet_users"] / (df["population"] + eps)) * 100
        df["digital_score"] = df["digital_score"].clip(0, 100)
    else:
        df["digital_score"] = 0.0

    # 3. Economic Stability: 100 - inflation - public_debt/10
    stability = 100.0
    if "inflation" in df.columns:
        stability = stability - df["inflation"].clip(0, 100)
    if "public_debt" in df.columns:
        stability = stability - (df["public_debt"] / 10).clip(0, 50)
    df["economic_stability"] = stability.clip(0, 100) if isinstance(stability, pd.Series) else pd.Series([stability] * len(df))

    # 4. Growth Potential: gdp_growth * (100 - unemployment) / 100
    growth = pd.Series([5.0] * len(df), index=df.index)
    if "gdp_growth" in df.columns:
        growth = df["gdp_growth"].clip(-5, 20)
    unemp_factor = 1.0
    if "unemployment" in df.columns:
        unemp_factor = (100 - df["unemployment"].clip(0, 100)) / 100
    df["growth_potential"] = (growth * unemp_factor).clip(0, 20)

    # 5. Infrastructure Score: (electricity + mobile_phones) / population
    infra = pd.Series([0.0] * len(df), index=df.index)
    if "electricity_production" in df.columns:
        infra = infra + df["electricity_production"].clip(lower=0)
    if "mobile_phones" in df.columns:
        infra = infra + df["mobile_phones"].clip(lower=0)
    if "population" in df.columns:
        df["infrastructure_score"] = (infra / (df["population"] + eps)).clip(0, 100)
    else:
        df["infrastructure_score"] = infra

    # 6. Trade Openness: (exports + imports) / gdp_ppp * 100
    trade = pd.Series([0.0] * len(df), index=df.index)
    if "exports" in df.columns:
        trade = trade + df["exports"].clip(lower=0)
    if "imports" in df.columns:
        trade = trade + df["imports"].clip(lower=0)
    if "gdp_ppp" in df.columns:
        df["trade_openness"] = (trade / (df["gdp_ppp"] + eps) * 100).clip(0, 200)
    else:
        df["trade_openness"] = trade

    # 7. Human Capital: literacy * (1 + gdp_per_capita / 100000)
    literacy = pd.Series([70.0] * len(df), index=df.index)
    if "literacy" in df.columns:
        literacy = df["literacy"].clip(0, 100)
    gdp_pc_factor = 1.0
    if "gdp_per_capita" in df.columns:
        gdp_pc_factor = 1 + (df["gdp_per_capita"].clip(0, 100000) / 100000)
    df["human_capital"] = (literacy * gdp_pc_factor).clip(0, 200)

    available_features = [f for f in FEATURE_COLS if f in df.columns]
    logger.info(f"[FEATURES] Engineered: {available_features}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_features(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Apply MinMaxScaler to all FEATURE_COLS.
    Returns the updated dataframe and the fitted scaler.
    """
    logger.info("[NORMALIZE] Applying MinMaxScaler to feature columns...")
    available = [f for f in FEATURE_COLS if f in df.columns]
    scaler = MinMaxScaler()
    df[available] = scaler.fit_transform(df[available].fillna(0))
    logger.info(f"[NORMALIZE] Normalized {len(available)} features to [0, 1]")
    return df, scaler


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — COUNTRY CLUSTERING (K-MEANS)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_countries(df: pd.DataFrame) -> tuple[pd.DataFrame, KMeans, PCA]:
    """
    K-Means clustering on normalized feature columns.
    Also runs PCA for 2D visualization.
    Labels clusters heuristically by average feature values.
    """
    logger.info(f"[CLUSTER] Running KMeans with k={N_CLUSTERS}...")
    available = [f for f in FEATURE_COLS if f in df.columns]
    X = df[available].fillna(0).values

    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df["cluster_id"] = km.fit_predict(X)

    # Heuristic cluster naming based on centroid averages
    centroids = pd.DataFrame(km.cluster_centers_, columns=available)
    cluster_label_map = _name_clusters(centroids)
    df["cluster_name"] = df["cluster_id"].map(cluster_label_map)

    # PCA for 2D scatter
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    df["pca_x"] = coords[:, 0]
    df["pca_y"] = coords[:, 1]

    logger.info(f"[CLUSTER] Cluster distribution:\n{df['cluster_name'].value_counts()}")
    return df, km, pca


def _name_clusters(centroids: pd.DataFrame) -> dict:
    """
    Assign human-readable names to clusters based on centroid values.
    """
    names = {}
    remaining = list(centroids.index)

    # Advanced Economies: highest gdp-related (economic_stability + human_capital)
    if "economic_stability" in centroids.columns and "human_capital" in centroids.columns:
        advanced_idx = centroids.loc[remaining, ["economic_stability", "human_capital"]].mean(axis=1).idxmax()
        names[advanced_idx] = "Advanced Economies"
        remaining.remove(advanced_idx)

    # Frontier Markets: lowest overall average
    if remaining:
        frontier_idx = centroids.loc[remaining].mean(axis=1).idxmin()
        names[frontier_idx] = "Frontier Markets"
        remaining.remove(frontier_idx)

    # Digital Leapfrog: highest digital_score among remaining
    if remaining and "digital_score" in centroids.columns:
        digital_idx = centroids.loc[remaining, "digital_score"].idxmax()
        names[digital_idx] = "Digital Leapfrog Markets"
        remaining.remove(digital_idx)

    # Emerging Markets: highest growth_potential among remaining
    if remaining and "growth_potential" in centroids.columns:
        emerging_idx = centroids.loc[remaining, "growth_potential"].idxmax()
        names[emerging_idx] = "Emerging Markets"
        remaining.remove(emerging_idx)

    # Remaining → Developing Economies
    for idx in remaining:
        names[idx] = "Developing Economies"

    return names


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — OPPORTUNITY SCORING
# ─────────────────────────────────────────────────────────────────────────────

def score_countries(df: pd.DataFrame, sector: str) -> pd.DataFrame:
    """
    Compute weighted opportunity score for a sector.
    Score = Σ(weight_i × feature_i) for all features.
    Adds 'opportunity_score' and 'rank' columns.
    """
    sector = sector.lower()
    if sector not in SECTOR_WEIGHTS:
        logger.warning(f"[SCORE] Unknown sector '{sector}', using 'general'")
        sector = "general"

    weights = SECTOR_WEIGHTS[sector]
    logger.info(f"[SCORE] Scoring for sector: {sector}")

    score = pd.Series(0.0, index=df.index)
    for feature, weight in weights.items():
        if weight > 0 and feature in df.columns:
            score += weight * df[feature]

    df["opportunity_score"] = (score * 100).round(2)  # scale to 0-100
    df["rank"] = df["opportunity_score"].rank(ascending=False, method="min").astype(int)
    df["sector"] = sector

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(sector: str = "general", data_path: str = DATA_PATH) -> dict:
    """
    Run the complete data mining pipeline.
    Returns a dict with the cleaned dataframe and pipeline artifacts.
    """
    logger.info("=" * 60)
    logger.info("[PIPELINE] Starting full data mining pipeline...")
    logger.info("=" * 60)

    # Step 1: Load
    df_raw = load_raw_data(data_path)

    # Step 2: Clean
    df_clean = clean_data(df_raw)

    # Step 3: Feature Engineering
    df_feat = engineer_features(df_clean)

    # Step 4: Normalize
    df_norm, scaler = normalize_features(df_feat)

    # Step 5: Cluster
    df_clustered, km_model, pca_model = cluster_countries(df_norm)

    # Step 6: Score
    df_scored = score_countries(df_clustered, sector)

    logger.info("[PIPELINE] ✅ Pipeline complete.")
    logger.info(f"[PIPELINE] Total countries processed: {len(df_scored)}")

    return {
        "df": df_scored,
        "scaler": scaler,
        "kmeans": km_model,
        "pca": pca_model,
        "sector": sector,
        "n_countries": len(df_scored),
    }
