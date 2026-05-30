"""
Churn Prediction Model — Logistic Regression, Random Forest, and XGBoost
=========================================================================

Author: Sanman Kadam
Description:
    This script builds and evaluates three machine learning models to predict
    customer churn risk at the circle-provider level. It generates model
    comparison visualizations, feature importance charts, and ROC curves.

Usage:
    python notebooks/churn_prediction.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)
import warnings
import joblib

warnings.filterwarnings("ignore")

# Try importing xgboost; fall back gracefully
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Skipping XGBoost model.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Cleaned_Telecom_Subscriptions.csv"
IMAGE_DIR = PROJECT_ROOT / "images"
IMAGE_DIR.mkdir(exist_ok=True)

COLORS = {
    "primary_blue": "#1B4F72",
    "light_blue": "#5DADE2",
    "info_blue": "#2E86C1",
    "risk_red": "#E74C3C",
    "risk_orange": "#E67E22",
    "safe_green": "#27AE60",
    "neutral_grey": "#95A5A6",
    "dark_grey": "#2C3E50",
    "bg_white": "#FAFAFA",
}

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

plt.rcParams.update({
    "figure.facecolor": COLORS["bg_white"],
    "axes.facecolor": COLORS["bg_white"],
    "axes.edgecolor": COLORS["dark_grey"],
    "axes.labelcolor": COLORS["dark_grey"],
    "text.color": COLORS["dark_grey"],
    "xtick.color": COLORS["dark_grey"],
    "ytick.color": COLORS["dark_grey"],
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})


# ---------------------------------------------------------------------------
# Data Preparation for Modeling
# ---------------------------------------------------------------------------
def prepare_modeling_data():
    """
    Transform raw subscription data into a supervised learning dataset.

    Each observation = one (circle, provider, connection_type, period) tuple.
    Target: 1 if subscribers declined in the next period, 0 otherwise.
    """
    print("Preparing modeling data...")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["value"] > 0]

    # Standardize
    circle_mapping = {
        "All india": "All India",
        "Andaman and Nicobar Islands": "Andaman and Nicobar",
        "Chattisgarh": "Chhattisgarh",
        "North East1": "North East",
        "North East2": "North East",
        "North East 1": "North East",
        "North East 2": "North East",
    }
    df["circle"] = df["circle"].replace(circle_mapping)
    df["type_of_connection"] = df["type_of_connection"].str.strip().str.lower()
    df["month_num"] = df["month"].map({m: i + 1 for i, m in enumerate(MONTH_ORDER)})
    df = df.dropna(subset=["month_num"])
    df["month_num"] = df["month_num"].astype(int)
    df["period"] = df["year"].astype(str) + "-" + df["month_num"].apply(lambda x: f"{x:02d}")

    # Filter to circles only
    df = df[df["circle"] != "All India"]

    # Sort and compute changes
    df = df.sort_values(["circle", "service_provider", "type_of_connection", "period"])
    df["prev_value"] = df.groupby(["circle", "service_provider", "type_of_connection"])["value"].shift(1)
    df["subscriber_change"] = df["value"] - df["prev_value"]
    df["change_pct"] = (df["subscriber_change"] / df["prev_value"]) * 100
    df = df.dropna(subset=["prev_value"])

    # Target: did subscribers decline?
    df["churn"] = (df["subscriber_change"] < 0).astype(int)

    # Feature engineering
    # Rolling averages
    group_cols = ["circle", "service_provider", "type_of_connection"]
    df["rolling_avg_3"] = df.groupby(group_cols)["value"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df["rolling_std_3"] = df.groupby(group_cols)["value"].transform(
        lambda x: x.rolling(3, min_periods=1).std().fillna(0)
    )
    df["value_to_avg_ratio"] = df["value"] / df["rolling_avg_3"]

    # Encode categoricals
    le_circle = LabelEncoder()
    le_provider = LabelEncoder()
    le_conn = LabelEncoder()

    df["circle_enc"] = le_circle.fit_transform(df["circle"])
    df["provider_enc"] = le_provider.fit_transform(df["service_provider"])
    df["conn_enc"] = le_conn.fit_transform(df["type_of_connection"])

    # Feature set
    features = [
        "value", "prev_value", "change_pct", "month_num",
        "rolling_avg_3", "rolling_std_3", "value_to_avg_ratio",
        "circle_enc", "provider_enc", "conn_enc",
    ]

    X = df[features].copy()
    y = df["churn"].copy()

    # Handle infinities and NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    feature_names = features

    print(f"  Dataset shape: {X.shape}")
    print(f"  Churn rate: {y.mean() * 100:.1f}%")
    print(f"  Features: {feature_names}")

    return X, y, feature_names, le_circle, le_provider, le_conn


# ---------------------------------------------------------------------------
# Model Training and Evaluation
# ---------------------------------------------------------------------------
def train_and_evaluate(X, y, feature_names):
    """Train LR, RF, and XGBoost; return results dictionary."""
    print("\nTraining models...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric="logloss", verbosity=0,
        )

    results = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")

        # Use scaled data for LR, raw for tree models
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # Cross-validation
        if name == "Logistic Regression":
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="roc_auc")
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        # Feature importance
        if name == "Logistic Regression":
            importance = np.abs(model.coef_[0])
        else:
            importance = model.feature_importances_

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(),
            "fpr": fpr,
            "tpr": tpr,
            "confusion_matrix": cm,
            "feature_importance": importance,
            "y_test": y_test,
            "y_pred": y_pred,
        }

        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    F1-Score:  {f1:.4f}")
        print(f"    ROC-AUC:   {auc:.4f}")
        print(f"    CV AUC:    {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return results, feature_names, scaler


# ---------------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------------
def plot_model_comparison(results):
    """Bar chart comparing all models across metrics."""
    print("\n  Plotting model comparison...")

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    model_names = list(results.keys())

    model_colors = [COLORS["info_blue"], COLORS["safe_green"], COLORS["risk_orange"]]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)

    for i, name in enumerate(model_names):
        values = [results[name][m] for m in metrics]
        bars = ax.bar(
            x + i * width - (len(model_names) - 1) * width / 2,
            values,
            width,
            label=name,
            color=model_colors[i % len(model_colors)],
            edgecolor="white",
        )
        # Value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", pad=15)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curves(results):
    """ROC curves for all models on one chart."""
    print("  Plotting ROC curves...")

    model_colors = [COLORS["info_blue"], COLORS["safe_green"], COLORS["risk_orange"]]

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, (name, res) in enumerate(results.items()):
        ax.plot(
            res["fpr"], res["tpr"],
            color=model_colors[i % len(model_colors)],
            linewidth=2.5,
            label=f"{name} (AUC = {res['roc_auc']:.3f})",
        )

    ax.plot([0, 1], [0, 1], color=COLORS["neutral_grey"], linewidth=1, linestyle="--", label="Random Baseline")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison", pad=15)
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "model_comparison_roc.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(results):
    """Confusion matrices for all models."""
    print("  Plotting confusion matrices...")

    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = res["confusion_matrix"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Active", "Churn"],
            yticklabels=["Active", "Churn"],
            ax=ax, cbar=False,
            annot_kws={"fontsize": 14, "fontweight": "bold"},
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(results, feature_names):
    """Feature importance for the best model."""
    print("  Plotting feature importance...")

    # Pick the model with highest AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    importance = results[best_name]["feature_importance"]

    # Sort features
    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=True)

    # Clean feature names for display
    name_map = {
        "value": "Subscriber Count",
        "prev_value": "Previous Period Count",
        "change_pct": "Change Percentage",
        "month_num": "Month",
        "rolling_avg_3": "3-Period Rolling Avg",
        "rolling_std_3": "3-Period Volatility",
        "value_to_avg_ratio": "Value/Avg Ratio",
        "circle_enc": "Circle (Region)",
        "provider_enc": "Service Provider",
        "conn_enc": "Connection Type",
    }
    feat_imp["feature_label"] = feat_imp["feature"].map(name_map).fillna(feat_imp["feature"])

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = [COLORS["risk_red"] if i >= len(feat_imp) - 3 else COLORS["info_blue"]
              for i in range(len(feat_imp))]

    ax.barh(
        feat_imp["feature_label"],
        feat_imp["importance"],
        color=colors,
        edgecolor="white",
    )

    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top Churn Drivers — {best_name}", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


def print_model_table(results):
    """Print a formatted model comparison table for the README."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON TABLE (for README)")
    print("=" * 80)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 80)
    for name, res in results.items():
        print(f"{name:<25} {res['accuracy']:>10.4f} {res['precision']:>10.4f} "
              f"{res['recall']:>10.4f} {res['f1']:>10.4f} {res['roc_auc']:>10.4f}")
    print("=" * 80)

    best = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\nBest Model: {best} (ROC-AUC: {results[best]['roc_auc']:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("CHURN PREDICTION MODEL")
    print("Author: Sanman Kadam")
    print("=" * 60)

    # Prepare data
    X, y, feature_names, le_circle, le_provider, le_conn = prepare_modeling_data()

    # Train and evaluate
    results, features, scaler = train_and_evaluate(X, y, feature_names)

    # Save encoders, scaler and models
    print("\nSaving encoders, scaler and models...")
    joblib.dump(scaler, PROJECT_ROOT / "notebooks" / "scaler.pkl")
    joblib.dump({
        "circle": le_circle,
        "provider": le_provider,
        "connection": le_conn
    }, PROJECT_ROOT / "notebooks" / "label_encoders.pkl")

    for name, res in results.items():
        model_filename = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(res["model"], PROJECT_ROOT / "notebooks" / f"model_{model_filename}")
        print(f"  Saved model_{model_filename}")

    # Generate visualizations
    print("\nGenerating model visualizations...")
    plot_model_comparison(results)
    plot_roc_curves(results)
    plot_confusion_matrices(results)
    plot_feature_importance(results, features)

    # Print summary
    print_model_table(results)

    print("\nAll model visualizations saved to images/")
    print("Done.")


if __name__ == "__main__":
    main()
