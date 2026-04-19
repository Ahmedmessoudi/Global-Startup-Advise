# ml_models.py — Advanced Supervised Learning Layer with Model Manager
# ═══════════════════════════════════════════════════════════════════════
# Architecture:
#   ModelManager  → centralizes model registry, training, and evaluation
#   3 Tasks       → sector, opportunity level, risk level
#   3 Models      → Random Forest, SVM, XGBoost
#   GridSearchCV  → CV=4, auto-select best model by F1 or accuracy
# ═══════════════════════════════════════════════════════════════════════
# NOTE: All preprocessing is handled by data_pipeline.py.
#       This module ONLY handles supervised learning.

import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_curve, auc,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder, label_binarize
from xgboost import XGBClassifier

from config import FEATURE_COLS, FEATURE_LABELS, SECTORS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SELECTION (SelectKBest — ANOVA F-test)
# ─────────────────────────────────────────────────────────────────────────────

def select_features(X: pd.DataFrame, y: pd.Series, k: int = 5) -> dict:
    """
    Perform feature selection using ANOVA F-test (SelectKBest).
    Returns scores, p-values, and selected features.
    """
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)

    scores = dict(zip(X.columns, selector.scores_))
    pvalues = dict(zip(X.columns, selector.pvalues_))
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    logger.info(f"[FEATURE SEL] Selected {len(selected_features)} features: {selected_features}")
    return {
        "scores": scores,
        "pvalues": pvalues,
        "selected_features": selected_features,
        "feature_ranking": ranking,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION TASKS — Target Builders
# ─────────────────────────────────────────────────────────────────────────────

TASK_DEFINITIONS = {
    "opportunity": {
        "name": "Niveau d'Opportunité",
        "description": "Classification 3 classes basée sur les quantiles du score d'opportunité.",
        "classes": ["Low", "Medium", "High"],
    },
    "risk": {
        "name": "Niveau de Risque",
        "description": (
            "Classification 3 classes basée sur un score de risque inversé "
            "(instabilité économique + chômage + dette publique)."
        ),
        "classes": ["Low Risk", "Medium Risk", "High Risk"],
    },
    "sector": {
        "name": "Secteur Recommandé",
        "description": (
            "Classification multi-classes : prédire le secteur le plus adapté "
            "pour chaque pays."
        ),
        "classes": [s for s in SECTORS if s != "general"],
    },
}


def create_target(df: pd.DataFrame, task: str = "opportunity") -> pd.Series:
    """
    Create classification target for the specified task.

    Tasks:
      - 'opportunity': Low / Medium / High (quantile-based, 3 classes)
      - 'risk': Low Risk / Medium Risk / High Risk (inverse stability score)
      - 'sector': best sector per country (multi-class)

    Returns integer-encoded labels (0, 1, 2, ...).
    """
    if task == "opportunity":
        score = df["opportunity_score"]
        q33 = score.quantile(0.33)
        q66 = score.quantile(0.66)
        y = pd.cut(
            score,
            bins=[-np.inf, q33, q66, np.inf],
            labels=[0, 1, 2],
        ).astype(int)
        return y

    elif task == "risk":
        # Risk score = combination of instability factors (higher = riskier)
        risk = pd.Series(0.0, index=df.index)
        if "economic_stability" in df.columns:
            risk += (1 - df["economic_stability"])  # low stability = high risk
        if "growth_potential" in df.columns:
            risk += (1 - df["growth_potential"])
        if "human_capital" in df.columns:
            risk += (1 - df["human_capital"])
        risk = risk / 3  # normalize to [0, 1]

        q33 = risk.quantile(0.33)
        q66 = risk.quantile(0.66)
        y = pd.cut(
            risk,
            bins=[-np.inf, q33, q66, np.inf],
            labels=[0, 1, 2],
        ).astype(int)
        return y

    elif task == "sector":
        # Best sector per country based on which sector weight maximizes score
        from config import SECTOR_WEIGHTS
        sector_list = [s for s in SECTORS if s != "general"]
        available = [f for f in FEATURE_COLS if f in df.columns]

        best_sectors = []
        for _, row in df[available].iterrows():
            best_score = -1
            best_sec = sector_list[0]
            for sec in sector_list:
                weights = SECTOR_WEIGHTS[sec]
                score = sum(weights.get(f, 0) * row.get(f, 0) for f in available)
                if score > best_score:
                    best_score = score
                    best_sec = sec
            best_sectors.append(best_sec)

        le = LabelEncoder()
        y = pd.Series(le.fit_transform(best_sectors), index=df.index)
        return y

    else:
        raise ValueError(f"Unknown task: {task}. Use 'opportunity', 'risk', or 'sector'.")


def get_task_labels(task: str) -> list:
    """Return human-readable labels for a given task."""
    return TASK_DEFINITIONS[task]["classes"]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DESCRIPTIONS (for academic presentation)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DESCRIPTIONS = {
    "Random Forest": {
        "definition": (
            "Random Forest is an ensemble learning method that constructs multiple "
            "decision trees during training and outputs the class that is the mode "
            "of the individual trees. It reduces overfitting by averaging many trees "
            "trained on random subsets of data and features."
        ),
        "steps": [
            "1. Bootstrap sampling: Draw N random samples with replacement",
            "2. At each node, select a random subset of features",
            "3. Split using the best feature among the subset (Gini or Entropy)",
            "4. Grow many trees independently (no pruning)",
            "5. Aggregate predictions via majority vote (classification)",
        ],
    },
    "SVM (SVC)": {
        "definition": (
            "Support Vector Machine (SVM) finds the optimal hyperplane that maximizes "
            "the margin between classes. Using kernel functions (RBF, linear), it can "
            "handle non-linearly separable data by mapping to a higher-dimensional space."
        ),
        "steps": [
            "1. Map data to a feature space (via kernel function)",
            "2. Find the hyperplane with maximum margin between classes",
            "3. Identify support vectors (closest points to the boundary)",
            "4. Use the regularization parameter C to control margin softness",
            "5. Classify new points based on which side of the hyperplane they fall",
        ],
    },
    "XGBoost": {
        "definition": (
            "XGBoost (Extreme Gradient Boosting) is an optimized ensemble learning method "
            "based on gradient boosting. It builds models sequentially where each new tree "
            "corrects the errors of the previous ones using gradient descent optimization. "
            "It is highly efficient, regularized, and widely used for structured/tabular data."
        ),
        "steps": [
            "1. Initialize predictions with a simple base value",
            "2. Compute residual errors between predictions and true labels",
            "3. Fit a decision tree to predict these residuals",
            "4. Optimize the tree using gradient descent and regularization",
            "5. Update predictions by adding the new tree (with learning rate)",
            "6. Repeat the process for multiple boosting rounds",
            "7. Final prediction is the weighted sum of all trees",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL MANAGER — Centralizes model registry, training, and evaluation
# ─────────────────────────────────────────────────────────────────────────────

class ModelManager:
    """
    Centralized Model Manager for supervised learning.

    Usage:
        mgr = ModelManager(X, y, task="opportunity", cv=4)
        mgr.run_feature_selection(k=5)
        mgr.train_all()
        best = mgr.get_best_model(metric="f1")
        summary = mgr.get_performance_summary()
        roc = mgr.get_roc_data()
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, task: str = "opportunity", cv: int = 4):
        self.X = X.fillna(0)
        self.y = y
        self.task = task
        self.cv = cv
        self.task_info = TASK_DEFINITIONS[task]
        self.labels = self.task_info["classes"]
        self.n_classes = len(self.labels)
        self.feature_names = list(X.columns)

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.25, random_state=42, stratify=self.y
        )

        # Results storage
        self.results = {}
        self.feature_selection = None
        self._best_model_name = None

        # Model registry
        self._build_model_registry()

        logger.info(
            f"[MODEL MGR] Initialized — Task: {task}, "
            f"Classes: {self.n_classes}, Samples: {len(X)}, "
            f"Train: {len(self.X_train)}, Test: {len(self.X_test)}"
        )

    def _build_model_registry(self):
        """Define model configurations depending on the task type."""
        is_binary = self.n_classes == 2

        # XGBoost objective depends on number of classes
        xgb_objective = "binary:logistic" if is_binary else "multi:softmax"
        xgb_extra = {"num_class": self.n_classes} if not is_binary else {}

        self.model_registry = {
            "Random Forest": {
                "estimator": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5],
                },
            },
            "SVM (SVC)": {
                "estimator": SVC(random_state=42, probability=True),
                "params": {
                    "C": [0.1, 1, 10],
                    "kernel": ["rbf", "linear"],
                    "gamma": ["scale", "auto"],
                },
            },
            "XGBoost": {
                "estimator": XGBClassifier(
                    objective=xgb_objective,
                    eval_metric="logloss" if is_binary else "mlogloss",
                    random_state=42,
                    use_label_encoder=False,
                    **xgb_extra,
                ),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5],
                    "learning_rate": [0.05, 0.1],
                    "subsample": [0.8, 1.0],
                },
            },
        }

    # ── Feature Selection ─────────────────────────────────────────────────

    def run_feature_selection(self, k: int = 5) -> dict:
        """Run ANOVA F-test feature selection."""
        self.feature_selection = select_features(self.X, self.y, k=k)
        return self.feature_selection

    # ── Training ──────────────────────────────────────────────────────────

    def train_all(self) -> dict:
        """Train all registered models with GridSearchCV."""
        unique_labels = sorted(self.y.unique())

        for name, config in self.model_registry.items():
            logger.info(f"[MODEL MGR] Training {name} (task={self.task}, cv={self.cv})...")

            grid = GridSearchCV(
                estimator=config["estimator"],
                param_grid=config["params"],
                cv=self.cv,
                scoring="accuracy",
                n_jobs=-1,
                return_train_score=True,
            )
            grid.fit(self.X_train, self.y_train)

            best = grid.best_estimator_
            y_pred = best.predict(self.X_test)

            cm = confusion_matrix(self.y_test, y_pred, labels=unique_labels)
            cv_scores = cross_val_score(best, self.X, self.y, cv=self.cv, scoring="accuracy")

            # Probability predictions for ROC (if model supports it)
            y_proba = None
            if hasattr(best, "predict_proba"):
                y_proba = best.predict_proba(self.X_test)

            # Feature importances (if model supports it)
            feat_imp = None
            if hasattr(best, "feature_importances_"):
                feat_imp = dict(zip(self.feature_names, best.feature_importances_))

            self.results[name] = {
                "model": best,
                "best_params": grid.best_params_,
                "best_cv_score": grid.best_score_,
                "test_accuracy": accuracy_score(self.y_test, y_pred),
                "test_precision": precision_score(
                    self.y_test, y_pred, average="weighted", zero_division=0
                ),
                "test_recall": recall_score(
                    self.y_test, y_pred, average="weighted", zero_division=0
                ),
                "test_f1": f1_score(
                    self.y_test, y_pred, average="weighted", zero_division=0
                ),
                "confusion_matrix": cm,
                "cv_scores": cv_scores,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "labels": self.labels,
                "y_pred": y_pred,
                "y_proba": y_proba,
                "feature_importances": feat_imp,
                "description": MODEL_DESCRIPTIONS.get(name, {}),
            }

            logger.info(
                f"[MODEL MGR] {name} — Best CV: {grid.best_score_:.4f}, "
                f"Test Acc: {accuracy_score(self.y_test, y_pred):.4f}"
            )

        return self.results

    # ── Best Model Selection ──────────────────────────────────────────────

    def get_best_model(self, metric: str = "auto") -> dict:
        """
        Return the best model based on the chosen metric.
        If metric is 'auto', it is automatically selected based on the task:
          - 'sector' (imbalanced/multiclass): F1-Score
          - 'opportunity', 'risk' (ordinal/balanced): Accuracy
        """
        if metric == "auto":
            if self.task == "sector":
                metric = "f1"
            else:
                metric = "accuracy"

        metric_map = {
            "f1": "test_f1",
            "accuracy": "test_accuracy",
            "precision": "test_precision",
            "recall": "test_recall",
        }
        key = metric_map.get(metric, "test_f1")
        best_name = max(self.results, key=lambda n: self.results[n][key])
        self._best_model_name = best_name

        return {
            "name": best_name,
            "metric": metric,
            "score": self.results[best_name][key],
            "model": self.results[best_name]["model"],
            "params": self.results[best_name]["best_params"],
        }

    # ── ROC Data ──────────────────────────────────────────────────────────

    def get_roc_data(self) -> dict:
        """
        Compute ROC curve data for each model.
        For multiclass, uses one-vs-rest macro average.
        """
        roc_data = {}
        unique_labels = sorted(self.y.unique())

        for name, res in self.results.items():
            y_proba = res.get("y_proba")
            if y_proba is None:
                continue

            if self.n_classes == 2:
                # Binary: use probability of positive class
                fpr, tpr, _ = roc_curve(self.y_test, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
            else:
                # Multiclass: macro-average one-vs-rest
                y_test_bin = label_binarize(self.y_test, classes=unique_labels)
                all_fpr = np.linspace(0, 1, 100)
                mean_tpr = np.zeros_like(all_fpr)

                for i in range(self.n_classes):
                    if i < y_proba.shape[1]:
                        fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)

                mean_tpr /= self.n_classes
                roc_auc = auc(all_fpr, mean_tpr)
                roc_data[name] = {"fpr": all_fpr, "tpr": mean_tpr, "auc": roc_auc}

        return roc_data

    # ── Performance Summary ───────────────────────────────────────────────

    def get_performance_summary(self) -> pd.DataFrame:
        """Create a summary table of all model performances."""
        rows = []
        for name, res in self.results.items():
            rows.append({
                "Model": name,
                "Best CV Accuracy": round(res["best_cv_score"], 4),
                "Test Accuracy": round(res["test_accuracy"], 4),
                "Precision": round(res["test_precision"], 4),
                "Recall": round(res["test_recall"], 4),
                "F1-Score": round(res["test_f1"], 4),
                "CV Mean ± Std": f"{res['cv_mean']:.4f} ± {res['cv_std']:.4f}",
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# BACKWARD COMPATIBILITY — keep old function signatures working
# ─────────────────────────────────────────────────────────────────────────────

def train_models(X: pd.DataFrame, y: pd.Series, cv: int = 4) -> dict:
    """Legacy wrapper: trains all models and returns results dict."""
    mgr = ModelManager(X, y, task="opportunity", cv=cv)
    return mgr.train_all()


def get_performance_summary(results: dict) -> pd.DataFrame:
    """Legacy wrapper: creates summary table from results dict."""
    rows = []
    for name, res in results.items():
        rows.append({
            "Model": name,
            "Best CV Accuracy": round(res["best_cv_score"], 4),
            "Test Accuracy": round(res["test_accuracy"], 4),
            "Precision": round(res["test_precision"], 4),
            "Recall": round(res["test_recall"], 4),
            "F1-Score": round(res["test_f1"], 4),
            "CV Mean ± Std": f"{res['cv_mean']:.4f} ± {res['cv_std']:.4f}",
        })
    return pd.DataFrame(rows)
