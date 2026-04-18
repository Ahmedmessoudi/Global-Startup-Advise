# results_engine.py — Scoring, Ranking, and Country Comparison Engine

import numpy as np
import pandas as pd
from difflib import get_close_matches

from config import FEATURE_COLS, FEATURE_LABELS, SECTOR_WEIGHTS


def fuzzy_find_country(name: str, df: pd.DataFrame, threshold: float = 0.6) -> str | None:
    """
    Find the closest matching country name in the dataframe.
    Returns the matched name or None if not found.
    """
    name = name.strip().title()
    countries = df["country"].tolist()

    # Exact match first
    if name in countries:
        return name

    # Fuzzy match
    matches = get_close_matches(name, countries, n=1, cutoff=threshold)
    if matches:
        return matches[0]
    return None


def get_country_profile(country_name: str, df: pd.DataFrame) -> dict | None:
    """
    Return a full profile dict for a country including:
    - raw indicators (GDP, population, etc.)
    - engineered features (normalized)
    - opportunity score and rank
    - cluster name
    """
    matched = fuzzy_find_country(country_name, df)
    if matched is None:
        return None

    row = df[df["country"] == matched].iloc[0]
    profile = {"country": matched}

    # Raw indicators (if present)
    raw_cols = [
        "gdp_ppp", "gdp_per_capita", "gdp_growth", "inflation", "unemployment",
        "population", "internet_users_pct", "public_debt", "exports", "imports",
        "labor_force", "literacy",
    ]
    for col in raw_cols:
        if col in df.columns:
            profile[col] = round(float(row[col]), 4) if not pd.isna(row[col]) else None

    # Engineered features (normalized 0-1)
    for feat in FEATURE_COLS:
        if feat in df.columns:
            profile[feat] = round(float(row[feat]), 4)

    # Scoring and clustering
    profile["opportunity_score"] = float(row.get("opportunity_score", 0))
    profile["rank"] = int(row.get("rank", 0))
    profile["cluster_name"] = str(row.get("cluster_name", "Unknown"))
    profile["sector"] = str(row.get("sector", "general"))

    return profile


def get_top_countries(df: pd.DataFrame, sector: str, n: int = 10) -> pd.DataFrame:
    """
    Return the top N countries by opportunity score for a given sector.
    """
    top = df.sort_values("opportunity_score", ascending=False).head(n)
    return top[["country", "opportunity_score", "rank", "cluster_name"] +
               [f for f in FEATURE_COLS if f in df.columns]].reset_index(drop=True)


def get_similar_countries(country_name: str, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Return countries in the same cluster as the target country,
    sorted by opportunity score descending (excluding the target itself).
    """
    matched = fuzzy_find_country(country_name, df)
    if matched is None:
        return pd.DataFrame()

    target_cluster = df[df["country"] == matched]["cluster_name"].values[0]
    similar = df[
        (df["cluster_name"] == target_cluster) &
        (df["country"] != matched)
    ].sort_values("opportunity_score", ascending=False).head(n)

    return similar[["country", "opportunity_score", "rank", "cluster_name"]].reset_index(drop=True)


def get_country_rank(country_name: str, df: pd.DataFrame) -> int | None:
    """Return the global rank of a country."""
    matched = fuzzy_find_country(country_name, df)
    if matched is None:
        return None
    row = df[df["country"] == matched]
    if row.empty:
        return None
    return int(row["rank"].values[0])


def get_feature_percentiles(country_name: str, df: pd.DataFrame) -> dict:
    """
    Return percentile rank (0-100) for each feature of the target country.
    Used for radar chart.
    """
    matched = fuzzy_find_country(country_name, df)
    if matched is None:
        return {}

    row = df[df["country"] == matched].iloc[0]
    percentiles = {}

    for feat in FEATURE_COLS:
        if feat in df.columns:
            val = row[feat]
            # Percentile: fraction of countries this country beats
            pct = (df[feat] < val).sum() / len(df) * 100
            percentiles[feat] = round(pct, 1)

    return percentiles


def compare_countries(country_list: list, df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a side-by-side comparison dataframe for a list of countries.
    """
    rows = []
    for name in country_list:
        matched = fuzzy_find_country(name, df)
        if matched:
            row = df[df["country"] == matched].iloc[0]
            entry = {"country": matched, "opportunity_score": row.get("opportunity_score", 0),
                     "rank": row.get("rank", 0), "cluster_name": row.get("cluster_name", "?")}
            for feat in FEATURE_COLS:
                if feat in df.columns:
                    entry[feat] = round(float(row[feat]), 3)
            rows.append(entry)

    return pd.DataFrame(rows)


def build_llm_context(
    target_country: str,
    sector: str,
    startup_description: str,
    df: pd.DataFrame,
    n_alternatives: int = 5,
) -> dict:
    """
    Build a structured context dict to send to the LLM for synthesis.
    Includes target profile, alternatives, cluster info.
    """
    profile = get_country_profile(target_country, df)
    if profile is None:
        return {"error": f"Country '{target_country}' not found in dataset."}

    top = get_top_countries(df, sector, n=20)
    alternatives = top[top["country"] != profile["country"]].head(n_alternatives)

    similar = get_similar_countries(target_country, df, n=3)

    percentiles = get_feature_percentiles(target_country, df)

    # Human-readable feature summaries
    feature_summary = {}
    for feat, label in FEATURE_LABELS.items():
        if feat in profile:
            raw_val = profile[feat]
            pct = percentiles.get(feat, 0)
            feature_summary[label] = {
                "normalized_score": raw_val,
                "percentile": pct,
                "assessment": "Strong" if pct >= 66 else ("Moderate" if pct >= 33 else "Weak"),
            }

    return {
        "startup_description": startup_description,
        "sector": sector,
        "target_country": profile["country"],
        "opportunity_score": profile["opportunity_score"],
        "global_rank": profile["rank"],
        "total_countries": len(df),
        "cluster": profile["cluster_name"],
        "feature_analysis": feature_summary,
        "raw_indicators": {
            "GDP_PPP_USD": profile.get("gdp_ppp"),
            "GDP_per_capita_USD": profile.get("gdp_per_capita"),
            "GDP_growth_pct": profile.get("gdp_growth"),
            "inflation_pct": profile.get("inflation"),
            "unemployment_pct": profile.get("unemployment"),
            "population": profile.get("population"),
            "internet_users_pct": profile.get("internet_users_pct"),
            "public_debt_pct_gdp": profile.get("public_debt"),
            "literacy_pct": profile.get("literacy"),
        },
        "top_alternative_countries": alternatives[["country", "opportunity_score", "rank", "cluster_name"]].to_dict("records"),
        "similar_cluster_countries": similar.to_dict("records") if not similar.empty else [],
    }
