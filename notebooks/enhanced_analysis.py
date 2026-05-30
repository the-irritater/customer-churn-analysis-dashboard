"""
Enhanced Churn Analysis — Revenue Impact, Risk Segmentation, and Trend Analysis
================================================================================

Author: Sanman Kadam
Description:
    This script performs advanced analysis on Indian telecom subscription data
    to quantify revenue impact, segment circles by risk level, and identify
    temporal churn trends. All visualizations follow a professional color scheme:
    Grey (neutral), Blue (informational), Red/Orange (churn risk).

Usage:
    python notebooks/enhanced_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Cleaned_Telecom_Subscriptions.csv"
IMAGE_DIR = PROJECT_ROOT / "images"
IMAGE_DIR.mkdir(exist_ok=True)

# Professional color palette
COLORS = {
    "primary_blue": "#1B4F72",
    "light_blue": "#5DADE2",
    "info_blue": "#2E86C1",
    "risk_red": "#E74C3C",
    "risk_orange": "#E67E22",
    "warning_yellow": "#F4D03F",
    "safe_green": "#27AE60",
    "neutral_grey": "#95A5A6",
    "dark_grey": "#2C3E50",
    "light_grey": "#ECF0F1",
    "bg_white": "#FAFAFA",
}

RISK_COLORS = {
    "Critical": COLORS["risk_red"],
    "High": COLORS["risk_orange"],
    "Medium": COLORS["warning_yellow"],
    "Low": COLORS["safe_green"],
}

# Matplotlib defaults
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
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
})

# Month ordering
MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


# ---------------------------------------------------------------------------
# Data Loading and Preparation
# ---------------------------------------------------------------------------
def load_and_prepare_data():
    """Load and clean the telecom subscription dataset."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Remove rows with missing or invalid values
    df = df.dropna(subset=["value"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["value"] > 0]

    # Standardize circle names
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

    # Standardize connection type
    df["type_of_connection"] = df["type_of_connection"].str.strip().str.lower()

    # Create month order for sorting
    df["month_num"] = df["month"].map({m: i + 1 for i, m in enumerate(MONTH_ORDER)})
    df = df.dropna(subset=["month_num"])
    df["month_num"] = df["month_num"].astype(int)

    # Create period column
    df["period"] = df["year"].astype(str) + "-" + df["month_num"].apply(lambda x: f"{x:02d}")

    # Filter out "All India" aggregates for circle-level analysis
    df_circles = df[df["circle"] != "All India"].copy()

    print(f"  Loaded {len(df):,} records across {df['circle'].nunique()} circles")
    print(f"  Time range: {df['period'].min()} to {df['period'].max()}")
    print(f"  Providers: {df['service_provider'].nunique()}")

    return df, df_circles


# ---------------------------------------------------------------------------
# Churn Rate Calculation
# ---------------------------------------------------------------------------
def calculate_churn_metrics(df_circles):
    """
    Calculate churn-like metrics by comparing subscriber counts across periods.

    Since this is subscription count data (not individual customer records),
    'churn' is approximated as the net subscriber loss between consecutive
    periods for each circle-provider-connection combination.
    """
    print("\nCalculating churn metrics...")

    # Sort by period
    df_sorted = df_circles.sort_values(["circle", "service_provider", "type_of_connection", "period"])

    # Calculate period-over-period change
    df_sorted["prev_value"] = df_sorted.groupby(
        ["circle", "service_provider", "type_of_connection"]
    )["value"].shift(1)

    df_sorted["subscriber_change"] = df_sorted["value"] - df_sorted["prev_value"]
    df_sorted["change_pct"] = (df_sorted["subscriber_change"] / df_sorted["prev_value"]) * 100

    # Flag subscriber loss (proxy for churn)
    df_sorted["is_declining"] = df_sorted["subscriber_change"] < 0
    df_sorted["subscribers_lost"] = df_sorted["subscriber_change"].clip(upper=0).abs()

    df_sorted = df_sorted.dropna(subset=["prev_value"])

    return df_sorted


# ---------------------------------------------------------------------------
# Circle-Level Risk Scoring
# ---------------------------------------------------------------------------
def calculate_circle_risk(df_metrics):
    """Segment circles into risk tiers based on subscriber loss patterns."""
    print("Calculating circle risk scores...")

    circle_stats = df_metrics.groupby("circle").agg(
        total_subscribers=("value", "sum"),
        total_lost=("subscribers_lost", "sum"),
        avg_change_pct=("change_pct", "mean"),
        decline_periods=("is_declining", "sum"),
        total_periods=("is_declining", "count"),
    ).reset_index()

    circle_stats["loss_rate"] = (circle_stats["total_lost"] / circle_stats["total_subscribers"]) * 100
    circle_stats["decline_frequency"] = (circle_stats["decline_periods"] / circle_stats["total_periods"]) * 100

    # Composite risk score (0-100)
    circle_stats["risk_score"] = (
        circle_stats["loss_rate"].rank(pct=True) * 40 +
        circle_stats["decline_frequency"].rank(pct=True) * 35 +
        circle_stats["avg_change_pct"].rank(ascending=True, pct=True) * 25
    )

    # Risk tiers
    circle_stats["risk_tier"] = pd.cut(
        circle_stats["risk_score"],
        bins=[0, 25, 50, 75, 100],
        labels=["Low", "Medium", "High", "Critical"],
    )

    return circle_stats.sort_values("risk_score", ascending=False)


# ---------------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------------
def plot_revenue_impact_by_circle(df_metrics, top_n=15):
    """Revenue at risk by circle — top N circles."""
    print(f"  Plotting revenue impact by circle (top {top_n})...")

    circle_loss = (
        df_metrics.groupby("circle")["subscribers_lost"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = [COLORS["risk_red"] if i < 5 else COLORS["risk_orange"] if i < 10 else COLORS["neutral_grey"]
              for i in range(len(circle_loss))]

    bars = ax.barh(
        range(len(circle_loss)),
        circle_loss.values,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_yticks(range(len(circle_loss)))
    ax.set_yticklabels(circle_loss.index)
    ax.invert_yaxis()
    ax.set_xlabel("Subscribers Lost (Cumulative)")
    ax.set_title("Top 15 Circles by Subscriber Attrition", pad=15)

    # Add value labels
    for bar, val in zip(bars, circle_loss.values):
        ax.text(
            bar.get_width() + circle_loss.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}",
            va="center",
            fontsize=9,
            color=COLORS["dark_grey"],
        )

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "revenue_impact_by_circle.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_revenue_impact_by_connection(df_metrics):
    """Revenue at risk split by wireless vs wireline."""
    print("  Plotting revenue impact by connection type...")

    conn_stats = df_metrics.groupby("type_of_connection").agg(
        total_subscribers=("value", "sum"),
        subscribers_lost=("subscribers_lost", "sum"),
        avg_change=("change_pct", "mean"),
    ).reset_index()

    conn_stats["loss_rate"] = (conn_stats["subscribers_lost"] / conn_stats["total_subscribers"]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Chart 1: Subscriber loss volume
    colors = [COLORS["risk_red"] if x == "wireline" else COLORS["info_blue"] for x in conn_stats["type_of_connection"]]
    bars1 = axes[0].bar(
        conn_stats["type_of_connection"].str.capitalize(),
        conn_stats["subscribers_lost"],
        color=colors,
        edgecolor="white",
        width=0.5,
    )
    axes[0].set_title("Subscribers Lost by Connection Type")
    axes[0].set_ylabel("Subscribers Lost")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M"))
    for bar, val in zip(bars1, conn_stats["subscribers_lost"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val / 1e6:.1f}M", ha="center", va="bottom", fontweight="bold", fontsize=11)

    # Chart 2: Loss rate
    bars2 = axes[1].bar(
        conn_stats["type_of_connection"].str.capitalize(),
        conn_stats["loss_rate"],
        color=colors,
        edgecolor="white",
        width=0.5,
    )
    axes[1].set_title("Loss Rate by Connection Type")
    axes[1].set_ylabel("Loss Rate (%)")
    for bar, val in zip(bars2, conn_stats["loss_rate"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Revenue Impact Analysis: Wireless vs Wireline", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "revenue_impact_by_connection.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_monthly_churn_trend(df_metrics):
    """Monthly trend of subscriber losses over time."""
    print("  Plotting monthly churn trend...")

    monthly = df_metrics.groupby("period").agg(
        total_lost=("subscribers_lost", "sum"),
        total_subscribers=("value", "sum"),
        net_change=("subscriber_change", "sum"),
    ).reset_index()

    monthly = monthly.sort_values("period")
    monthly["loss_rate"] = (monthly["total_lost"] / monthly["total_subscribers"]) * 100

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Chart 1: Net subscriber change
    colors_net = [COLORS["safe_green"] if x > 0 else COLORS["risk_red"] for x in monthly["net_change"]]
    axes[0].bar(range(len(monthly)), monthly["net_change"], color=colors_net, edgecolor="white", linewidth=0.5)
    axes[0].axhline(y=0, color=COLORS["dark_grey"], linewidth=0.8, linestyle="--")
    axes[0].set_title("Net Subscriber Change by Period")
    axes[0].set_ylabel("Net Change")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))

    # Chart 2: Loss rate trend
    axes[1].plot(range(len(monthly)), monthly["loss_rate"],
                 color=COLORS["risk_red"], linewidth=2.5, marker="o", markersize=5)
    axes[1].fill_between(range(len(monthly)), monthly["loss_rate"],
                         alpha=0.15, color=COLORS["risk_red"])
    axes[1].set_title("Subscriber Loss Rate Trend")
    axes[1].set_ylabel("Loss Rate (%)")

    # X-axis labels
    axes[1].set_xticks(range(len(monthly)))
    axes[1].set_xticklabels(monthly["period"], rotation=45, ha="right", fontsize=9)
    axes[1].set_xlabel("Period")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Monthly Churn Trend Analysis", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "monthly_churn_trend.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_risk_segmentation(circle_risk):
    """Risk segmentation matrix — risk score vs subscriber volume."""
    print("  Plotting risk segmentation matrix...")

    # Filter to meaningful circles
    cr = circle_risk[circle_risk["total_subscribers"] > 0].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Chart 1: Risk tier distribution
    tier_counts = cr["risk_tier"].value_counts().reindex(["Critical", "High", "Medium", "Low"])
    tier_colors = [RISK_COLORS.get(t, COLORS["neutral_grey"]) for t in tier_counts.index]
    bars = axes[0].bar(tier_counts.index, tier_counts.values, color=tier_colors, edgecolor="white", width=0.6)
    axes[0].set_title("Circles by Risk Tier")
    axes[0].set_ylabel("Number of Circles")
    for bar, val in zip(bars, tier_counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     str(int(val)), ha="center", fontweight="bold", fontsize=12)

    # Chart 2: Scatter — risk score vs subscriber volume
    tier_color_map = cr["risk_tier"].map(RISK_COLORS)
    axes[1].scatter(
        cr["risk_score"],
        cr["total_subscribers"],
        c=tier_color_map,
        s=cr["loss_rate"] * 20 + 50,
        alpha=0.75,
        edgecolors="white",
        linewidth=0.8,
    )

    # Label top risk circles
    top_risk = cr.nlargest(5, "risk_score")
    for _, row in top_risk.iterrows():
        axes[1].annotate(
            row["circle"],
            (row["risk_score"], row["total_subscribers"]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=8,
            color=COLORS["dark_grey"],
        )

    axes[1].set_xlabel("Risk Score (0-100)")
    axes[1].set_ylabel("Total Subscribers")
    axes[1].set_title("Risk Score vs Subscriber Volume")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M"))

    # Add quadrant lines
    axes[1].axvline(x=50, color=COLORS["neutral_grey"], linewidth=0.8, linestyle="--", alpha=0.5)
    axes[1].axhline(y=cr["total_subscribers"].median(), color=COLORS["neutral_grey"],
                    linewidth=0.8, linestyle="--", alpha=0.5)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Customer Risk Segmentation Analysis", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "risk_segmentation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_provider_market_share(df_circles):
    """Provider market share analysis — who is gaining/losing."""
    print("  Plotting provider market share...")

    # Get top providers by total subscribers
    top_providers = (
        df_circles.groupby("service_provider")["value"]
        .sum()
        .nlargest(6)
        .index.tolist()
    )

    df_top = df_circles[df_circles["service_provider"].isin(top_providers)]

    # Market share by period
    period_total = df_top.groupby("period")["value"].sum()
    provider_period = df_top.groupby(["period", "service_provider"])["value"].sum().unstack(fill_value=0)
    market_share = provider_period.div(period_total, axis=0) * 100
    market_share = market_share.sort_index()

    fig, ax = plt.subplots(figsize=(14, 7))

    provider_colors = [
        COLORS["primary_blue"], COLORS["info_blue"], COLORS["risk_red"],
        COLORS["safe_green"], COLORS["risk_orange"], COLORS["neutral_grey"],
    ]

    for i, provider in enumerate(market_share.columns):
        ax.plot(
            range(len(market_share)),
            market_share[provider],
            label=provider,
            linewidth=2,
            color=provider_colors[i % len(provider_colors)],
            marker="o",
            markersize=4,
        )

    ax.set_xticks(range(len(market_share)))
    ax.set_xticklabels(market_share.index, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Period")
    ax.set_ylabel("Market Share (%)")
    ax.set_title("Provider Market Share Trends", pad=15)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "provider_market_share.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_executive_kpi_dashboard(df, df_circles, circle_risk):
    """Generate a KPI summary dashboard image for the README."""
    print("  Plotting executive KPI dashboard...")

    # Calculate KPIs
    total_records = len(df_circles)
    total_circles = df_circles["circle"].nunique()
    total_providers = df_circles["service_provider"].nunique()
    latest_period = df_circles["period"].max()
    latest_subs = df_circles[df_circles["period"] == latest_period]["value"].sum()
    critical_circles = len(circle_risk[circle_risk["risk_tier"] == "Critical"])
    high_risk_circles = len(circle_risk[circle_risk["risk_tier"] == "High"])

    fig = plt.figure(figsize=(14, 4))
    fig.patch.set_facecolor(COLORS["bg_white"])

    # KPI Cards
    kpis = [
        ("Total Circles", f"{total_circles}", COLORS["primary_blue"]),
        ("Service Providers", f"{total_providers}", COLORS["info_blue"]),
        ("Latest Subscribers", f"{latest_subs / 1e6:.0f}M", COLORS["safe_green"]),
        ("Critical Risk Circles", f"{critical_circles}", COLORS["risk_red"]),
        ("High Risk Circles", f"{high_risk_circles}", COLORS["risk_orange"]),
        ("Data Points", f"{total_records:,}", COLORS["neutral_grey"]),
    ]

    for i, (label, value, color) in enumerate(kpis):
        ax = fig.add_axes([i / len(kpis) + 0.01, 0.1, 1 / len(kpis) - 0.02, 0.8])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Card background
        rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, linewidth=1.5,
                              edgecolor=color, facecolor=color, alpha=0.08,
                              transform=ax.transAxes, zorder=0)
        ax.add_patch(rect)

        # Top color bar
        top_bar = plt.Rectangle((0.05, 0.85), 0.9, 0.1, linewidth=0,
                                facecolor=color, alpha=0.9,
                                transform=ax.transAxes, zorder=1)
        ax.add_patch(top_bar)

        ax.text(0.5, 0.55, value, ha="center", va="center",
                fontsize=22, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.25, label, ha="center", va="center",
                fontsize=10, color=COLORS["dark_grey"], transform=ax.transAxes)
        ax.axis("off")

    plt.savefig(IMAGE_DIR / "kpi_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Summary Statistics for README
# ---------------------------------------------------------------------------
def print_summary_stats(df_circles, df_metrics, circle_risk):
    """Print key statistics that will be used in the README."""
    print("\n" + "=" * 60)
    print("KEY METRICS FOR README")
    print("=" * 60)

    total_circles = df_circles["circle"].nunique()
    total_providers = df_circles["service_provider"].nunique()
    total_records = len(df_circles)

    latest_period = df_circles["period"].max()
    latest_subs = df_circles[df_circles["period"] == latest_period]["value"].sum()

    total_lost = df_metrics["subscribers_lost"].sum()
    avg_loss_rate = (total_lost / df_metrics["value"].sum()) * 100

    critical = circle_risk[circle_risk["risk_tier"] == "Critical"]
    high = circle_risk[circle_risk["risk_tier"] == "High"]

    print(f"  Total Circles Analyzed:    {total_circles}")
    print(f"  Service Providers:         {total_providers}")
    print(f"  Data Points:               {total_records:,}")
    print(f"  Latest Subscribers:        {latest_subs:,.0f}")
    print(f"  Cumulative Subscribers Lost: {total_lost:,.0f}")
    print(f"  Average Loss Rate:         {avg_loss_rate:.2f}%")
    print(f"  Critical Risk Circles:     {len(critical)}")
    print(f"  High Risk Circles:         {len(high)}")
    print(f"  Critical circles:          {', '.join(critical['circle'].tolist()[:5])}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("ENHANCED CHURN ANALYSIS")
    print("Author: Sanman Kadam")
    print("=" * 60)

    # Load data
    df, df_circles = load_and_prepare_data()

    # Calculate churn metrics
    df_metrics = calculate_churn_metrics(df_circles)

    # Risk segmentation
    circle_risk = calculate_circle_risk(df_metrics)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_revenue_impact_by_circle(df_metrics)
    plot_revenue_impact_by_connection(df_metrics)
    plot_monthly_churn_trend(df_metrics)
    plot_risk_segmentation(circle_risk)
    plot_provider_market_share(df_circles)
    plot_executive_kpi_dashboard(df, df_circles, circle_risk)

    # Print summary
    print_summary_stats(df_circles, df_metrics, circle_risk)

    # Save risk segmentation data
    circle_risk.to_csv(PROJECT_ROOT / "data" / "circle_risk_scores.csv", index=False)
    print(f"\nRisk scores saved to data/circle_risk_scores.csv")

    print("\nAll visualizations saved to images/")
    print("Done.")


if __name__ == "__main__":
    main()
