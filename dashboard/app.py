"""
Customer Churn Analysis Dashboard — Interactive Streamlit Application
=====================================================================

Author: Sanman Kadam
Description:
    Multi-page interactive dashboard for exploring churn patterns,
    revenue impact, risk segmentation, and predictive model results.

Usage:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Analysis Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Cleaned_Telecom_Subscriptions.csv"
RISK_PATH = PROJECT_ROOT / "data" / "circle_risk_scores.csv"

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Color scheme
COLORS = {
    "primary": "#1B4F72",
    "blue": "#2E86C1",
    "light_blue": "#5DADE2",
    "red": "#E74C3C",
    "orange": "#E67E22",
    "green": "#27AE60",
    "grey": "#95A5A6",
    "dark": "#2C3E50",
    "background": "#0E1117",
}

RISK_COLORS = {
    "Critical": "#E74C3C",
    "High": "#E67E22",
    "Medium": "#F4D03F",
    "Low": "#27AE60",
}


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stMetric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E86C1;
    }
    h1, h2, h3 {
        color: #ECF0F1;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load and prepare the subscription dataset."""
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df[df["value"] > 0]

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

    df_circles = df[df["circle"] != "All India"].copy()
    return df, df_circles


@st.cache_data
def load_risk_data():
    """Load precomputed risk scores."""
    if RISK_PATH.exists():
        return pd.read_csv(RISK_PATH)
    return None


@st.cache_data
def calculate_churn_metrics(df_circles):
    """Compute period-over-period subscriber changes."""
    df_sorted = df_circles.sort_values(
        ["circle", "service_provider", "type_of_connection", "period"]
    )
    df_sorted["prev_value"] = df_sorted.groupby(
        ["circle", "service_provider", "type_of_connection"]
    )["value"].shift(1)
    df_sorted["subscriber_change"] = df_sorted["value"] - df_sorted["prev_value"]
    df_sorted["change_pct"] = (df_sorted["subscriber_change"] / df_sorted["prev_value"]) * 100
    df_sorted["subscribers_lost"] = df_sorted["subscriber_change"].clip(upper=0).abs()
    df_sorted = df_sorted.dropna(subset=["prev_value"])
    return df_sorted


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
def page_overview(df, df_circles, df_metrics, risk_data):
    st.title("Customer Churn Analysis Dashboard")
    st.markdown("**Author: Sanman Kadam** | Telecom Subscriber Attrition Analysis")
    st.markdown("---")

    # KPI Row
    latest_period = df_circles["period"].max()
    latest_subs = df_circles[df_circles["period"] == latest_period]["value"].sum()
    total_circles = df_circles["circle"].nunique()
    total_providers = df_circles["service_provider"].nunique()
    total_lost = df_metrics["subscribers_lost"].sum()
    avg_loss_rate = (total_lost / df_metrics["value"].sum()) * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Active Subscribers", f"{latest_subs / 1e9:.2f}B")
    col2.metric("Circles Analyzed", str(total_circles))
    col3.metric("Service Providers", str(total_providers))
    col4.metric("Subscribers Lost", f"{total_lost / 1e9:.2f}B")
    col5.metric("Avg Loss Rate", f"{avg_loss_rate:.2f}%")

    st.markdown("---")

    # Two-column layout
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Subscriber Distribution by Connection Type")
        conn_data = df_circles.groupby("type_of_connection")["value"].sum().reset_index()
        conn_data["type_of_connection"] = conn_data["type_of_connection"].str.capitalize()
        fig = px.pie(
            conn_data, values="value", names="type_of_connection",
            color_discrete_sequence=[COLORS["blue"], COLORS["red"]],
            hole=0.4,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ECF0F1"),
            margin=dict(t=30, b=30, l=30, r=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Top 10 Circles by Subscriber Volume")
        circle_subs = (
            df_circles[df_circles["period"] == latest_period]
            .groupby("circle")["value"]
            .sum()
            .nlargest(10)
            .reset_index()
        )
        fig = px.bar(
            circle_subs, x="value", y="circle", orientation="h",
            color_discrete_sequence=[COLORS["blue"]],
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ECF0F1"),
            xaxis_title="Subscribers",
            yaxis_title="",
            margin=dict(t=30, b=30, l=30, r=30),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Executive summary
    st.markdown("---")
    st.subheader("Executive Summary")
    st.markdown("""
    | Finding | Detail |
    |---|---|
    | **At-Risk Base** | 30% of subscribers (17,499) identified as at-risk for churn |
    | **Wireline Vulnerability** | Wireline connections show disproportionately higher churn than wireless |
    | **Critical Circles** | 5 circles require immediate retention intervention |
    | **Financial Opportunity** | Retention program projected to deliver 5,900% ROI over 3 years |
    | **Low-Value Churn** | Low-value subscribers are 3x more likely to churn |
    """)


# ---------------------------------------------------------------------------
# Page: Geographic Analysis
# ---------------------------------------------------------------------------
def page_geographic(df_circles, df_metrics, risk_data):
    st.title("Geographic Analysis")
    st.markdown("---")

    # Circle selector
    circles = sorted(df_circles["circle"].unique())
    selected_circles = st.multiselect(
        "Filter by Circle", circles, default=None,
        placeholder="All circles shown by default",
    )

    # Subscriber loss by circle
    circle_loss = (
        df_metrics.groupby("circle")["subscribers_lost"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )

    if selected_circles:
        circle_loss = circle_loss[circle_loss["circle"].isin(selected_circles)]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Circles by Subscriber Attrition")
        fig = px.bar(
            circle_loss, x="subscribers_lost", y="circle", orientation="h",
            color="subscribers_lost",
            color_continuous_scale=["#F4D03F", "#E67E22", "#E74C3C"],
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ECF0F1"),
            xaxis_title="Subscribers Lost",
            yaxis_title="",
            coloraxis_showscale=False,
            margin=dict(t=30, b=30, l=30, r=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if risk_data is not None:
            st.subheader("Circle Risk Scores")
            risk_sorted = risk_data.sort_values("risk_score", ascending=False).head(20)
            if selected_circles:
                risk_sorted = risk_sorted[risk_sorted["circle"].isin(selected_circles)]

            fig = px.bar(
                risk_sorted, x="risk_score", y="circle", orientation="h",
                color="risk_tier",
                color_discrete_map=RISK_COLORS,
            )
            fig.update_layout(
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ECF0F1"),
                xaxis_title="Risk Score (0-100)",
                yaxis_title="",
                margin=dict(t=30, b=30, l=30, r=30),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Risk segmentation scatter
    if risk_data is not None:
        st.markdown("---")
        st.subheader("Risk vs Subscriber Volume Matrix")
        fig = px.scatter(
            risk_data, x="risk_score", y="total_subscribers",
            color="risk_tier",
            size="loss_rate",
            hover_data=["circle", "decline_frequency"],
            color_discrete_map=RISK_COLORS,
            text="circle",
        )
        fig.update_traces(textposition="top center", textfont_size=9)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ECF0F1"),
            xaxis_title="Risk Score",
            yaxis_title="Total Subscribers",
            margin=dict(t=30, b=30, l=30, r=30),
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Revenue Impact
# ---------------------------------------------------------------------------
def page_revenue(df_circles, df_metrics):
    st.title("Revenue Impact Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue Impact by Connection Type")
        conn_loss = df_metrics.groupby("type_of_connection").agg(
            subscribers_lost=("subscribers_lost", "sum"),
            total=("value", "sum"),
        ).reset_index()
        conn_loss["loss_rate"] = (conn_loss["subscribers_lost"] / conn_loss["total"]) * 100
        conn_loss["type_of_connection"] = conn_loss["type_of_connection"].str.capitalize()

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Subscribers Lost", "Loss Rate (%)"))
        fig.add_trace(
            go.Bar(
                x=conn_loss["type_of_connection"],
                y=conn_loss["subscribers_lost"],
                marker_color=[COLORS["blue"], COLORS["red"]],
                name="Subscribers Lost",
            ), row=1, col=1
        )
        fig.add_trace(
            go.Bar(
                x=conn_loss["type_of_connection"],
                y=conn_loss["loss_rate"],
                marker_color=[COLORS["blue"], COLORS["red"]],
                name="Loss Rate",
            ), row=1, col=2
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ECF0F1"),
            showlegend=False,
            margin=dict(t=50, b=30, l=30, r=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Providers by Subscriber Loss")
        provider_loss = (
            df_metrics.groupby("service_provider")["subscribers_lost"]
            .sum()
            .nlargest(10)
            .reset_index()
        )
        fig = px.bar(
            provider_loss, x="subscribers_lost", y="service_provider",
            orientation="h",
            color_discrete_sequence=[COLORS["orange"]],
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ECF0F1"),
            xaxis_title="Subscribers Lost",
            yaxis_title="",
            margin=dict(t=30, b=30, l=30, r=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Financial summary
    st.markdown("---")
    st.subheader("Financial Impact Summary")

    fin_col1, fin_col2, fin_col3, fin_col4 = st.columns(4)
    fin_col1.metric("Investment Required", "$1,524.6M")
    fin_col2.metric("Revenue Saved (3-Year)", "$91,476.5M")
    fin_col3.metric("Net Benefit", "$89,951.9M")
    fin_col4.metric("ROI", "5,900%")

    # ROI scenarios
    scenarios = pd.DataFrame({
        "Scenario": ["Conservative (15%)", "Moderate (25%)", "Optimistic (35%)", "Aggressive (50%)"],
        "Retention Rate": [15, 25, 35, 50],
        "Net Benefit ($M)": [53971.1, 89951.9, 125932.7, 179903.8],
        "ROI (%)": [5900, 5900, 5900, 5900],
    })

    fig = px.bar(
        scenarios, x="Scenario", y="Net Benefit ($M)",
        color="Net Benefit ($M)",
        color_continuous_scale=["#27AE60", "#2E86C1"],
        text="Net Benefit ($M)",
    )
    fig.update_traces(texttemplate="$%{text:,.0f}M", textposition="outside")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ECF0F1"),
        coloraxis_showscale=False,
        margin=dict(t=30, b=30, l=30, r=30),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Trend Analysis
# ---------------------------------------------------------------------------
def page_trends(df_circles, df_metrics):
    st.title("Trend Analysis")
    st.markdown("---")

    # Time granularity selector
    granularity = st.radio("Time Granularity", ["Monthly", "Quarterly", "Yearly"], horizontal=True)

    monthly = df_metrics.groupby("period").agg(
        total_lost=("subscribers_lost", "sum"),
        total_subscribers=("value", "sum"),
        net_change=("subscriber_change", "sum"),
    ).reset_index().sort_values("period")

    monthly["loss_rate"] = (monthly["total_lost"] / monthly["total_subscribers"]) * 100

    if granularity == "Quarterly":
        monthly["quarter"] = monthly["period"].str[:4] + "-Q" + (
            (monthly["period"].str[5:7].astype(int) - 1) // 3 + 1
        ).astype(str)
        monthly = monthly.groupby("quarter").agg(
            total_lost=("total_lost", "sum"),
            total_subscribers=("total_subscribers", "sum"),
            net_change=("net_change", "sum"),
        ).reset_index()
        monthly["loss_rate"] = (monthly["total_lost"] / monthly["total_subscribers"]) * 100
        monthly = monthly.rename(columns={"quarter": "period"})
    elif granularity == "Yearly":
        monthly["year"] = monthly["period"].str[:4]
        monthly = monthly.groupby("year").agg(
            total_lost=("total_lost", "sum"),
            total_subscribers=("total_subscribers", "sum"),
            net_change=("net_change", "sum"),
        ).reset_index()
        monthly["loss_rate"] = (monthly["total_lost"] / monthly["total_subscribers"]) * 100
        monthly = monthly.rename(columns={"year": "period"})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Net Subscriber Change")
        colors = ["#27AE60" if x > 0 else "#E74C3C" for x in monthly["net_change"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly["period"],
            y=monthly["net_change"],
            marker_color=colors,
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="#95A5A6")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ECF0F1"),
            xaxis_title="Period",
            yaxis_title="Net Change",
            margin=dict(t=30, b=30, l=30, r=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Loss Rate Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["period"],
            y=monthly["loss_rate"],
            mode="lines+markers",
            line=dict(color=COLORS["red"], width=2.5),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(231, 76, 60, 0.1)",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ECF0F1"),
            xaxis_title="Period",
            yaxis_title="Loss Rate (%)",
            margin=dict(t=30, b=30, l=30, r=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Provider trends
    st.markdown("---")
    st.subheader("Provider Market Share Over Time")

    top_providers = (
        df_circles.groupby("service_provider")["value"]
        .sum()
        .nlargest(6)
        .index.tolist()
    )

    df_top = df_circles[df_circles["service_provider"].isin(top_providers)]
    period_total = df_top.groupby("period")["value"].sum()
    provider_period = df_top.groupby(["period", "service_provider"])["value"].sum().unstack(fill_value=0)
    market_share = provider_period.div(period_total, axis=0) * 100
    market_share = market_share.sort_index().reset_index().melt(
        id_vars="period", var_name="Provider", value_name="Market Share (%)"
    )

    fig = px.line(
        market_share, x="period", y="Market Share (%)",
        color="Provider",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ECF0F1"),
        xaxis_title="Period",
        margin=dict(t=30, b=30, l=30, r=30),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Predictions
# ---------------------------------------------------------------------------
def page_predictions():
    st.title("Predictive Model Results")
    st.markdown("---")

    # Model comparison table
    st.subheader("Model Performance Comparison")

    model_data = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy": [0.8166, 0.9999, 1.0000],
        "Precision": [0.9511, 1.0000, 1.0000],
        "Recall": [0.5215, 0.9997, 1.0000],
        "F1-Score": [0.6737, 0.9999, 1.0000],
        "ROC-AUC": [0.9445, 1.0000, 1.0000],
    })

    st.dataframe(
        model_data.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            color="#27AE60",
        ).format(precision=4),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Selected Model:** XGBoost — highest overall performance across all evaluation metrics.")

    # Display saved charts
    col1, col2 = st.columns(2)

    roc_path = PROJECT_ROOT / "images" / "model_comparison_roc.png"
    feat_path = PROJECT_ROOT / "images" / "feature_importance.png"
    comp_path = PROJECT_ROOT / "images" / "model_comparison.png"
    cm_path = PROJECT_ROOT / "images" / "confusion_matrices.png"

    with col1:
        st.subheader("ROC Curve Comparison")
        if roc_path.exists():
            st.image(str(roc_path))

    with col2:
        st.subheader("Feature Importance")
        if feat_path.exists():
            st.image(str(feat_path))

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Model Metrics Comparison")
        if comp_path.exists():
            st.image(str(comp_path))

    with col4:
        st.subheader("Confusion Matrices")
        if cm_path.exists():
            st.image(str(cm_path))


# ---------------------------------------------------------------------------
# Page: Customer Details (Drill-Through)
# ---------------------------------------------------------------------------
def page_details(df_circles, df_metrics):
    st.title("Customer Details — Drill-Through")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        circle_filter = st.selectbox(
            "Select Circle",
            ["All"] + sorted(df_circles["circle"].unique().tolist()),
        )
    with col2:
        conn_filter = st.selectbox(
            "Connection Type",
            ["All", "Wireless", "Wireline"],
        )
    with col3:
        provider_filter = st.selectbox(
            "Provider",
            ["All"] + sorted(df_circles["service_provider"].unique().tolist()),
        )

    filtered = df_metrics.copy()

    if circle_filter != "All":
        filtered = filtered[filtered["circle"] == circle_filter]
    if conn_filter != "All":
        filtered = filtered[filtered["type_of_connection"] == conn_filter.lower()]
    if provider_filter != "All":
        filtered = filtered[filtered["service_provider"] == provider_filter]

    st.markdown(f"**Showing {len(filtered):,} records**")

    # Summary stats
    if len(filtered) > 0:
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        s_col1.metric("Total Subscribers", f"{filtered['value'].sum():,.0f}")
        s_col2.metric("Total Lost", f"{filtered['subscribers_lost'].sum():,.0f}")
        s_col3.metric("Avg Change %", f"{filtered['change_pct'].mean():.2f}%")
        s_col4.metric("Decline Periods", f"{filtered['subscribers_lost'].gt(0).sum():,}")

    # Data table
    st.markdown("---")
    display_cols = [
        "period", "circle", "service_provider", "type_of_connection",
        "value", "prev_value", "subscriber_change", "change_pct",
    ]
    available_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[available_cols].sort_values("period", ascending=False).head(500),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main():
    # Load data
    df, df_circles = load_data()
    df_metrics = calculate_churn_metrics(df_circles)
    risk_data = load_risk_data()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Select Page",
        [
            "Overview",
            "Geographic Analysis",
            "Revenue Impact",
            "Trend Analysis",
            "Predictions",
            "Customer Details",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Author:** Sanman Kadam")
    st.sidebar.markdown("**Data:** TRAI Telecom Subscriptions")
    st.sidebar.markdown("**Models:** LR, RF, XGBoost")

    # Route to page
    if page == "Overview":
        page_overview(df, df_circles, df_metrics, risk_data)
    elif page == "Geographic Analysis":
        page_geographic(df_circles, df_metrics, risk_data)
    elif page == "Revenue Impact":
        page_revenue(df_circles, df_metrics)
    elif page == "Trend Analysis":
        page_trends(df_circles, df_metrics)
    elif page == "Predictions":
        page_predictions()
    elif page == "Customer Details":
        page_details(df_circles, df_metrics)


if __name__ == "__main__":
    main()
