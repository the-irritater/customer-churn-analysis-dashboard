# Customer Churn Analysis and Retention Strategy

**Author:** Sanman Kadam  
**Status:** Complete  
**License:** MIT

---

## Business Problem

A telecom company operating across 30+ regional circles in India is experiencing subscriber attrition across wireless and wireline segments. Customer churn directly translates to revenue loss, increased acquisition costs, and weakened competitive positioning.

The objective of this project is to:
- Identify the key drivers of subscriber churn at the regional and provider level
- Quantify the financial impact of subscriber attrition
- Build predictive models to flag at-risk segments before churn occurs
- Recommend actionable retention strategies backed by data

---

## Dataset Overview

| Attribute | Detail |
|---|---|
| **Source** | Indian Telecom Regulatory Authority (TRAI) |
| **Records** | 70,000+ subscription records |
| **Time Range** | January 2009 to April 2025 |
| **Circles (Regions)** | 30 telecom circles across India |
| **Service Providers** | 42 operators (Airtel, Jio, BSNL, Vodafone Idea, etc.) |
| **Connection Types** | Wireless, Wireline |
| **Key Field** | Subscriber count per circle-provider-connection-month |

---

## KPIs Tracked

| KPI | Value |
|---|---|
| Total Circles Analyzed | 30 |
| Service Providers | 42 |
| Latest Active Subscribers | 1.19 Billion |
| Cumulative Subscribers Lost | 2.35 Billion |
| Average Loss Rate | 1.17% |
| At-Risk Customers (Model) | 17,499 (30.0% of base) |
| Critical Risk Circles | 5 |
| Projected ROI of Retention Program | 5,900% |
| Net Benefit (3-Year) | $89.9 Billion |

---

## Analysis Process

```
1. Data Collection and Cleaning
   Collected subscription data across all Indian telecom circles.
   Cleaned inconsistencies, standardized circle names, removed duplicates.

2. Feature Engineering
   Computed period-over-period subscriber changes.
   Created rolling averages, volatility measures, and value-to-average ratios.
   Encoded categorical variables for modeling.

3. Exploratory Data Analysis
   Analyzed churn patterns by circle, connection type, and provider.
   Identified geographic hotspots and high-risk segments.

4. Predictive Modeling
   Trained Logistic Regression, Random Forest, and XGBoost classifiers.
   Evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.

5. Risk Segmentation
   Scored circles on a composite risk index (0-100).
   Classified into Critical, High, Medium, and Low risk tiers.

6. Business Translation
   Converted model outputs into executive-level insights.
   Quantified revenue impact and proposed intervention strategies with ROI analysis.
```

---

## Key Insights

### Executive Summary

- **30% of the customer base (17,499 subscribers) are classified as at-risk**, representing a potential revenue loss of $0.15 Billion.

- **Wireline subscribers show disproportionately higher churn** compared to wireless, despite representing a smaller share of the total base.

- **5 circles are classified as Critical Risk** (Himachal Pradesh, Gujarat, Kolkata, Haryana, Uttar Pradesh East) — these regions require immediate intervention.

- **Low-value subscribers (below 56,783) are 3x more likely to churn** compared to high-value segments, indicating that retention efforts should be tiered by customer value.

- **The proposed retention program yields an estimated 5,900% ROI** with a net benefit of $89.9 Billion over a 3-year period.

---

## Revenue Impact Analysis

### Subscribers Lost by Circle

The top 5 circles account for a disproportionate share of total subscriber attrition. Uttar Pradesh (East), Bihar, and Madhya Pradesh lead in absolute subscriber losses.

![Revenue Impact by Circle](images/revenue_impact_by_circle.png)

### Wireless vs Wireline Impact

Wireline connections exhibit a significantly higher loss rate despite lower subscriber volumes, suggesting systemic service quality issues in the fixed-line segment.

![Revenue Impact by Connection Type](images/revenue_impact_by_connection.png)

### Monthly Churn Trend

The trend analysis reveals whether attrition is accelerating or stabilizing over time, enabling proactive resource allocation.

![Monthly Churn Trend](images/monthly_churn_trend.png)

---

## Risk Segmentation

Circles are classified into four risk tiers based on a composite score incorporating loss rate, decline frequency, and average change percentage:

| Risk Tier | Criteria | Count | Action |
|---|---|---|---|
| **Critical** | Risk Score > 75 | 5 | Immediate intervention required |
| **High** | Risk Score 50-75 | 11 | Targeted retention campaigns |
| **Medium** | Risk Score 25-50 | 7 | Monitor and optimize |
| **Low** | Risk Score < 25 | 7 | Standard operations |

![Risk Segmentation Matrix](images/risk_segmentation_matrix.png)

---

## Predictive Model Performance

Three machine learning models were trained to predict subscriber churn at the circle-provider level:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8166 | 0.9511 | 0.5215 | 0.6737 | 0.9445 |
| Random Forest | 0.9999 | 1.0000 | 0.9997 | 0.9999 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**Selected Model:** XGBoost — delivers perfect classification performance with the strongest generalization across cross-validation folds.

### ROC Curve Comparison

![ROC Curves](images/model_comparison_roc.png)

### Feature Importance — Top Churn Drivers

![Feature Importance](images/feature_importance.png)

### Model Performance Comparison

![Model Comparison](images/model_comparison.png)

---

## Business Recommendations

### Immediate Actions (0-3 Months)

1. **Deploy targeted retention campaigns** in the 5 Critical Risk circles (Himachal Pradesh, Gujarat, Kolkata, Haryana, UP East).
2. **Implement an early warning system** using the XGBoost churn prediction model to flag at-risk segments monthly.
3. **Create personalized offers** for at-risk, high-value subscribers to prevent revenue leakage.

### Medium-Term Initiatives (3-6 Months)

4. **Improve wireline service quality** — wireline connections show a disproportionately high loss rate that suggests infrastructure issues.
5. **Develop loyalty programs** targeting low-value subscribers to reduce the 3x churn disparity.
6. **Conduct satisfaction surveys** in high-churn circles to identify root causes beyond what data reveals.

### Long-Term Strategy (6-12 Months)

7. **Invest in network infrastructure** in critical circles where churn correlates with service gaps.
8. **Launch competitive pricing strategies** to counter local market rivals in high-churn regions.
9. **Build real-time churn monitoring** with automated model retraining on a quarterly basis.

---

## Dashboard Previews

### KPI Overview

![KPI Dashboard](images/kpi_dashboard.png)

### Churn Prevention Metrics

![Strategic Dashboard](images/strategic_dashboard.png)

### Customer Distribution and Churn Analysis

![Churn Distribution](images/churn_distribution_analysis.png)

### High-Risk Circle Analysis

![High Risk Circles](images/high_risk_circles_analysis.png)

### Financial Impact and ROI

![ROI Analysis](images/roi_financial_analysis.png)

### Provider Market Share Trends

![Provider Market Share](images/provider_market_share.png)

---

## Analytical Measures

The following analytical computations power the dashboard and model:

**Churn Rate Calculation**
```python
churn_rate = (subscribers_lost / total_subscribers) * 100
```

**Revenue at Risk**
```python
revenue_at_risk = df[df['churn'] == 1]['value'].sum()
```

**Risk Score (Composite Index)**
```python
risk_score = (
    loss_rate_rank * 0.40 +
    decline_frequency_rank * 0.35 +
    change_pct_rank * 0.25
)
```

**ROI of Retention Program**
```python
roi = ((revenue_saved - investment) / investment) * 100
```

---

## Tools and Technologies

| Category | Tools |
|---|---|
| **Language** | Python 3.x |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Dashboard** | Streamlit |
| **Environment** | Jupyter Notebook |

---

## Project Structure

```
customer-churn-analysis-dashboard/
├── data/
│   ├── Cleaned_Telecom_Subscriptions.csv
│   └── circle_risk_scores.csv
├── notebooks/
│   ├── Project.ipynb
│   ├── enhanced_analysis.py
│   └── churn_prediction.py
├── images/
│   ├── kpi_dashboard.png
│   ├── strategic_dashboard.png
│   ├── churn_distribution_analysis.png
│   ├── high_risk_circles_analysis.png
│   ├── roi_financial_analysis.png
│   ├── revenue_impact_by_circle.png
│   ├── revenue_impact_by_connection.png
│   ├── monthly_churn_trend.png
│   ├── risk_segmentation_matrix.png
│   ├── provider_market_share.png
│   ├── model_comparison.png
│   ├── model_comparison_roc.png
│   ├── feature_importance.png
│   └── confusion_matrices.png
├── docs/
│   └── Executive_Summary_Business_Insights.txt
├── dashboard/
│   └── app.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/the-irritater/customer-churn-analysis-dashboard.git
cd customer-churn-analysis-dashboard

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the enhanced analysis
python notebooks/enhanced_analysis.py

# Run the churn prediction model
python notebooks/churn_prediction.py

# Launch the Streamlit dashboard
streamlit run dashboard/app.py

# Or open the Jupyter notebook
jupyter notebook notebooks/Project.ipynb
```

---

## Future Improvements

- **Cohort Analysis** — Track churn behavior by subscriber acquisition cohort to identify lifecycle patterns.
- **Real-Time Monitoring** — Deploy the model as an API endpoint for live churn scoring.
- **A/B Testing Framework** — Measure the effectiveness of different retention interventions.
- **Retention Strategy Simulation** — Build scenario models for different investment levels and their expected ROI.
- **Customer Lifetime Value (CLV) Model** — Integrate predicted CLV to prioritize retention efforts by economic value.

---

**Author:** Sanman Kadam | **License:** MIT
