# Technical Report: Analytical Measures & Model Validation Details

This technical report provides the detailed calculations, formulas, DAX measures, and model validation protocols that power the Customer Churn Analysis Dashboard.

-

## 1. Analytical Measures (DAX & Python)

These calculations are used to measure baseline customer volume changes and compute projected revenue savings.

### Power BI (DAX) Measures

* **Subscriber Churn Rate:**
  ```dax
  Churn Rate = DIVIDE([Churned Customers], [Total Customers], 0)
  ```
* **Annual Revenue at Risk:**
  ```dax
  Annual Revenue at Risk = [Subscribers Lost] * [ARPU] * 12
  ```
* **Projected Revenue Saved:**
  ```dax
  Revenue Recovered = [Annual Revenue at Risk] * [Target Churn Reduction %]
  ```
* **Campaign ROI (%):**
  ```dax
  Campaign ROI = DIVIDE([Revenue Recovered] - [Campaign Cost], [Campaign Cost], 0) * 100
  ```

### Python Calculations

* **Aggregate Churn Rate:**
  ```python
  churn_rate = (subscribers_lost / total_subscribers) * 100
  ```
* **Composite Risk Score (Circle Level):**
  ```python
  risk_score = (
      loss_rate_rank * 0.40 +          # Severity of subscriber loss
      decline_frequency_rank * 0.35 +   # Consistency of decline over time
      change_pct_rank * 0.25            # Monthly rate of change
  )
  ```
* **Sigmoid Churn Probability (Individual Level):**
  ```python
  z = beta_0 + sum(beta_i * x_i)
  churn_probability = 1.0 / (1.0 + np.exp(-z))
  ```

-

## 2. Model Validation & Target Leakage Prevention

### Target Leakage Detection & Resolution
In early iterations, the machine learning models (XGBoost and Random Forest) exhibited perfect scores (`Accuracy = 1.0000`, `ROC-AUC = 1.0000`). Further validation revealed a **target leakage** where concurrent change metrics (`change_pct`) and subscriber count changes were leaking the contemporaneous target label (churn in month $t$).

To resolve this, we restructured the dataset for **time-aware prediction**:
1. **Target Shift:** The target variable was shifted by `-1` (`churn_next`), making it represent whether the subscriber base declines in the *next month* ($t+1$).
2. **Current Features:** All input features (subscriber volume, volatility, change percentage, rolling average) represent the *current month* ($t$) and past history.
3. **Data Partitioning:** The last month of records for each segment was dropped during training since the future target is unknown.

This time-shift eliminated target leakage, resulting in robust and realistic metrics suitable for production:
* **XGBoost ROC-AUC:** `0.8731`
* **Random Forest ROC-AUC:** `0.8609`
* **Logistic Regression ROC-AUC:** `0.6279`

### Cross-Validation Strategy
A stratified 5-fold cross-validation was implemented on the training set. Stratification ensures that each fold contains roughly the same percentage of churn events (approx. 36.6% base churn rate), providing a highly stable estimate of model performance across folds (XGBoost CV AUC standard deviation: `0.0058`).
