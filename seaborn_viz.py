# seaborn_viz.py — Seaborn-based visualizations for data exploration & ML results
# Professional blue palette on white background — UNICEF / data-platform aesthetic
# Required by project description: "Visualiser les données en utilisant les fonctions de seaborn"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import FEATURE_COLS, FEATURE_LABELS
from sklearn.model_selection import learning_curve

# ── Professional Blue Palette ────────────────────────────────────────────────
PRIMARY = "#305CDE"
DARK_BLUE = "#1B3A8C"
MID_BLUE = "#4A7BF7"
LIGHT_BLUE = "#A3BFF5"
PALE_BLUE = "#E8EEFB"
ACCENT_BLUE = "#6C9CFF"
TEXT_DARK = "#000000"
TEXT_MED = "#000000"
TEXT_LIGHT = "#000000"
BG_WHITE = "#FFFFFF"
GRID_COLOR = "#E2E8F0"
BORDER_COLOR = "#CBD5E1"

# Monochrome blue palette for categorical/sequential data
BLUE_PALETTE = [PRIMARY, MID_BLUE, DARK_BLUE, LIGHT_BLUE, ACCENT_BLUE]
BLUE_SEQ = ["#E8EEFB", "#A3BFF5", "#6C9CFF", "#305CDE", "#1B3A8C"]

# ── Global style ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette=BLUE_PALETTE)
_LIGHT_RC = {
    "figure.facecolor": BG_WHITE,
    "axes.facecolor": BG_WHITE,
    "text.color": TEXT_DARK,
    "axes.labelcolor": TEXT_DARK,
    "xtick.color": TEXT_MED,
    "ytick.color": TEXT_MED,
    "axes.edgecolor": BORDER_COLOR,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _apply_style():
    """Apply professional light theme."""
    plt.rcParams.update(_LIGHT_RC)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def correlation_heatmap(df: pd.DataFrame):
    """Seaborn heatmap of feature correlations — blue gradient."""
    _apply_style()
    features = [f for f in FEATURE_COLS if f in df.columns]
    labels = [FEATURE_LABELS.get(f, f) for f in features]
    corr = df[features].corr()
    corr.index = labels
    corr.columns = labels

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = sns.light_palette(PRIMARY, as_cmap=True)
    sns.heatmap(
        corr, annot=True, cmap="coolwarm", center=0, square=True,
        linewidths=0.8, fmt=".2f", ax=ax,
        linecolor=BG_WHITE,
        cbar_kws={"label": "Corrélation", "shrink": 0.8},
        annot_kws={"fontsize": 9, "color": TEXT_DARK},
    )
    ax.set_title("Matrice de Corrélation des Features", fontsize=15, color=TEXT_DARK, pad=18, fontweight="600")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────

def feature_distributions(df: pd.DataFrame):
    """Histograms + KDE for all features — blue tones."""
    _apply_style()
    features = [f for f in FEATURE_COLS if f in df.columns]
    n = len(features)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, feat in enumerate(features):
        label = FEATURE_LABELS.get(feat, feat)
        sns.histplot(df[feat], kde=True, ax=axes[i], color=PRIMARY, edgecolor=BG_WHITE, alpha=0.75)
        axes[i].set_title(label, fontsize=11, color=TEXT_DARK, fontweight="500")
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribution des Features", fontsize=15, color=TEXT_DARK, y=1.02, fontweight="600")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLASS DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def class_distribution_plot(y: pd.Series):
    """Distribution des classes (target imbalance check)."""
    _apply_style()

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.countplot(x=y, hue=y, palette=BLUE_PALETTE, ax=ax, legend=False, edgecolor=BG_WHITE)

    ax.set_title("Distribution des Classes (Target)", fontsize=13, color=TEXT_DARK, pad=12, fontweight="600")
    ax.set_xlabel("Classe", color=TEXT_DARK)
    ax.set_ylabel("Nombre d'échantillons", color=TEXT_DARK)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. BOXPLOTS FOR OUTLIER DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def boxplot_outliers(df: pd.DataFrame):
    """Horizontal boxplots for outlier visualization — blue tones."""
    _apply_style()
    features = [f for f in FEATURE_COLS if f in df.columns]
    labels = [FEATURE_LABELS.get(f, f) for f in features]

    plot_df = df[features].copy()
    plot_df.columns = labels

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = sns.boxplot(
        data=plot_df, orient="h", palette=BLUE_PALETTE, ax=ax,
        flierprops={"marker": "o", "markerfacecolor": DARK_BLUE, "markersize": 5, "markeredgecolor": BG_WHITE},
        boxprops={"edgecolor": PRIMARY},
        medianprops={"color": DARK_BLUE, "linewidth": 2},
        whiskerprops={"color": TEXT_MED},
        capprops={"color": TEXT_MED},
    )
    ax.set_title("Boxplots — Détection des Valeurs Aberrantes", fontsize=15, color=TEXT_DARK, pad=18, fontweight="600")
    ax.set_xlabel("Valeur Normalisée (0–1)", color=TEXT_DARK)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. PAIRPLOT BY CLUSTER
# ─────────────────────────────────────────────────────────────────────────────

def pairplot_features(df: pd.DataFrame, max_features: int = 4):
    """Seaborn pairplot of top features colored by cluster — blue tones."""
    _apply_style()
    features = [f for f in FEATURE_COLS if f in df.columns][:max_features]
    labels_map = {f: FEATURE_LABELS.get(f, f) for f in features}

    plot_df = df[features + ["cluster_name"]].copy()
    plot_df = plot_df.rename(columns=labels_map)

    g = sns.pairplot(
        plot_df, hue="cluster_name", palette=BLUE_PALETTE,
        diag_kind="kde", plot_kws={"alpha": 0.55, "s": 28, "edgecolor": BG_WHITE},
    )
    g.figure.suptitle("Pairplot des Features par Cluster", y=1.02, fontsize=15, color=TEXT_DARK, fontweight="600")
    g.figure.set_facecolor(BG_WHITE)
    return g.figure


# ─────────────────────────────────────────────────────────────────────────────
# 6. OPPORTUNITY SCORE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def opportunity_score_distribution(df: pd.DataFrame):
    """Distribution des opportunity scores des pays."""
    _apply_style()

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(df["opportunity_score"], kde=True, color=PRIMARY, ax=ax, edgecolor=BG_WHITE, alpha=0.75)

    ax.set_title("Distribution des Scores d'Opportunité", fontsize=13, color=TEXT_DARK, pad=12, fontweight="600")
    ax.set_xlabel("Score d'Opportunité", color=TEXT_DARK)
    ax.set_ylabel("Nombre de pays", color=TEXT_DARK)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def confusion_matrix_plot(cm, labels, title: str = "Matrice de Confusion"):
    """Seaborn heatmap-based confusion matrix — blue cmap."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = sns.light_palette(PRIMARY, as_cmap=True)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap, square=True,
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=1.5, linecolor=BG_WHITE,
        cbar_kws={"label": "Nombre", "shrink": 0.8},
        annot_kws={"fontsize": 12, "fontweight": "600", "color": TEXT_DARK},
    )
    ax.set_xlabel("Prédit", fontsize=12, color=TEXT_DARK, fontweight="500")
    ax.set_ylabel("Réel", fontsize=12, color=TEXT_DARK, fontweight="500")
    ax.set_title(title, fontsize=14, color=TEXT_DARK, pad=15, fontweight="600")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. FEATURE IMPORTANCE (Random Forest / XGBoost)
# ─────────────────────────────────────────────────────────────────────────────

def feature_importance_plot(importances: dict, title: str = "Importance des Features"):
    """Horizontal bar plot of feature importances — blue gradient."""
    _apply_style()
    features = list(importances.keys())
    values = list(importances.values())
    labels = [FEATURE_LABELS.get(f, f) for f in features]

    sorted_idx = np.argsort(values)
    n = len(sorted_idx)
    colors = [sns.light_palette(PRIMARY, n_colors=n + 2)[i + 2] for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=[values[i] for i in sorted_idx],
        y=[labels[i] for i in sorted_idx],
        hue=[labels[i] for i in sorted_idx],
        palette=colors, ax=ax, legend=False, edgecolor=BG_WHITE,
    )
    ax.set_title(title, fontsize=14, color=TEXT_DARK, pad=15, fontweight="600")
    ax.set_xlabel("Score d'importance", color=TEXT_DARK)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 9. CROSS-VALIDATION SCORES COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def cv_scores_plot(results: dict):
    """Boxplot comparing CV scores across models — blue palette."""
    _apply_style()
    data = []
    for name, res in results.items():
        for score in res["cv_scores"]:
            data.append({"Modèle": name, "Score CV": score})

    plot_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=plot_df, x="Modèle", y="Score CV", hue="Modèle", palette=BLUE_PALETTE[:3], ax=ax, legend=False)
    sns.stripplot(data=plot_df, x="Modèle", y="Score CV", color=DARK_BLUE, size=7, alpha=0.6, ax=ax)
    ax.set_title("Comparaison des Scores de Validation Croisée (CV=4)", fontsize=14, color=TEXT_DARK, pad=15, fontweight="600")
    ax.set_ylabel("Accuracy", color=TEXT_DARK)
    ax.set_xlabel("")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 10. FEATURE SELECTION SCORES
# ─────────────────────────────────────────────────────────────────────────────

def feature_selection_plot(scores: dict, pvalues: dict):
    """Bar chart of ANOVA F-scores from feature selection — blue significant, light insignificant."""
    _apply_style()
    features = list(scores.keys())
    f_scores = list(scores.values())
    labels = [FEATURE_LABELS.get(f, f) for f in features]

    sorted_idx = np.argsort(f_scores)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PRIMARY if pvalues.get(features[i], 1) < 0.05 else LIGHT_BLUE for i in sorted_idx]
    sns.barplot(
        x=[f_scores[i] for i in sorted_idx],
        y=[labels[i] for i in sorted_idx],
        hue=[labels[i] for i in sorted_idx],
        palette=colors, ax=ax, legend=False, edgecolor=BG_WHITE,
    )
    ax.set_title("Sélection des Features — Scores ANOVA F-test", fontsize=14, color=TEXT_DARK, pad=15, fontweight="600")
    ax.set_xlabel("F-Score (foncé = p < 0.05, clair = p ≥ 0.05)", color=TEXT_DARK)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 11. ROC CURVE
# ─────────────────────────────────────────────────────────────────────────────

def roc_curve_plot(roc_data: dict):
    """
    Plot ROC curves for all models — blue palette.
    roc_data: {model_name: {"fpr": array, "tpr": array, "auc": float}}
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    palette = [PRIMARY, MID_BLUE, DARK_BLUE]
    for i, (name, data) in enumerate(roc_data.items()):
        color = palette[i % len(palette)]
        ax.plot(
            data["fpr"], data["tpr"],
            color=color, lw=2.5,
            label=f"{name} (AUC = {data['auc']:.3f})",
        )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], color=TEXT_LIGHT, linestyle="--", lw=1, alpha=0.7, label="Aléatoire (AUC = 0.5)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Taux de Faux Positifs (FPR)", fontsize=12, color=TEXT_DARK)
    ax.set_ylabel("Taux de Vrais Positifs (TPR)", fontsize=12, color=TEXT_DARK)
    ax.set_title("Courbes ROC — Comparaison des Modèles", fontsize=14, color=TEXT_DARK, pad=15, fontweight="600")
    ax.legend(loc="lower right", fontsize=10, facecolor=BG_WHITE, edgecolor=BORDER_COLOR)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 12. LEARNING CURVE (ML DIAGNOSTIC)
# ─────────────────────────────────────────────────────────────────────────────

def learning_curve_plot(estimator, X, y):
    """Learning curve to detect overfitting / underfitting — blue tones."""
    _apply_style()

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y,
        cv=4, scoring="f1_macro",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(train_sizes, train_mean, label="Score Entraînement", color=PRIMARY, lw=2)
    ax.plot(train_sizes, test_mean, label="Score Validation", color=MID_BLUE, lw=2, linestyle="--")
    ax.fill_between(train_sizes, train_mean, test_mean, alpha=0.08, color=PRIMARY)

    ax.set_title("Courbe d'Apprentissage — Diagnostic", fontsize=13, color=TEXT_DARK, pad=12, fontweight="600")
    ax.set_xlabel("Taille du jeu de données", color=TEXT_DARK)
    ax.set_ylabel("F1-Score", color=TEXT_DARK)

    ax.legend(facecolor=BG_WHITE, edgecolor=BORDER_COLOR)
    plt.tight_layout()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 13. MODEL COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def model_comparison_plot(results: dict):
    """
    Grouped bar chart comparing Accuracy, Precision, Recall, F1 — blue palette.
    """
    _apply_style()
    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    data = []
    for name, res in results.items():
        for metric, label in zip(metrics, metric_labels):
            data.append({"Modèle": name, "Métrique": label, "Score": res[metric]})

    plot_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=plot_df, x="Métrique", y="Score", hue="Modèle",
        palette=BLUE_PALETTE[:3], ax=ax, edgecolor=BG_WHITE,
    )
    ax.set_title("Comparaison des Modèles — Métriques de Performance", fontsize=14, color=TEXT_DARK, pad=15, fontweight="600")
    ax.set_ylabel("Score", color=TEXT_DARK)
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Modèle", loc="lower right", fontsize=10, facecolor=BG_WHITE, edgecolor=BORDER_COLOR)

    # Add value annotations on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, color=TEXT_DARK, padding=3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 14. TOP & BOTTOM COUNTRIES BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def top_bottom_countries_plot(df: pd.DataFrame, n: int = 10):
    """Side-by-side bar chart of top-N and bottom-N countries by opportunity score."""
    _apply_style()
    top = df.nlargest(n, "opportunity_score")[["country", "opportunity_score"]]
    bottom = df.nsmallest(n, "opportunity_score")[["country", "opportunity_score"]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top N
    sns.barplot(data=top, y="country", x="opportunity_score", ax=axes[0],
                color=PRIMARY, edgecolor=BG_WHITE)
    axes[0].set_title(f"Top {n} Pays", fontsize=13, color=TEXT_DARK, fontweight="600")
    axes[0].set_xlabel("Score d'Opportunité")
    axes[0].set_xlim(0, 100)

    # Bottom N
    sns.barplot(data=bottom, y="country", x="opportunity_score", ax=axes[1],
                color=LIGHT_BLUE, edgecolor=BG_WHITE)
    axes[1].set_title(f"Bottom {n} Pays", fontsize=13, color=TEXT_DARK, fontweight="600")
    axes[1].set_xlabel("Score d'Opportunité")
    axes[1].set_xlim(0, 100)

    fig.suptitle("Comparaison Top vs Bottom Pays", fontsize=15, color=TEXT_DARK, y=1.02, fontweight="600")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 15. CLUSTER COMPOSITION PIE
# ─────────────────────────────────────────────────────────────────────────────

def cluster_composition_plot(df: pd.DataFrame):
    """Pie chart showing the number of countries per cluster."""
    _apply_style()
    counts = df["cluster_name"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette(BLUE_PALETTE, n_colors=len(counts))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, autopct="%1.1f%%",
        colors=colors, startangle=90, pctdistance=0.85,
        wedgeprops=dict(edgecolor=BG_WHITE, linewidth=2),
    )
    for t in texts + autotexts:
        t.set_color(TEXT_DARK)
    ax.set_title("Répartition des Pays par Cluster", fontsize=14, color=TEXT_DARK, pad=15, fontweight="600")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 16. VIOLIN PLOT — SCORE BY CLUSTER
# ─────────────────────────────────────────────────────────────────────────────

def violin_score_by_cluster(df: pd.DataFrame):
    """Violin plot of opportunity scores grouped by cluster."""
    _apply_style()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(
        data=df, x="cluster_name", y="opportunity_score",
        hue="cluster_name", palette=BLUE_PALETTE, ax=ax, legend=False,
        inner="box", linewidth=1.2,
    )
    ax.set_title("Distribution des Scores d'Opportunité par Cluster", fontsize=14, color=TEXT_DARK, pad=15, fontweight="600")
    ax.set_xlabel("Cluster", color=TEXT_DARK)
    ax.set_ylabel("Score d'Opportunité", color=TEXT_DARK)
    plt.tight_layout()
    return fig
