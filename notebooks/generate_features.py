import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("Generating Telecom_Features.csv from Cleaned_Telecom_Subscriptions.csv...")
    
    # Path setup
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "Cleaned_Telecom_Subscriptions.csv"
    
    # Load dataset
    df = pd.read_csv(data_path, low_memory=False)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Clean value column
    df['value'] = pd.to_numeric(df['value'].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
    df = df.dropna(subset=['value'])
    df = df[df['value'] > 0]
    
    # Standardize circle names
    circle_mapping = {
        "All india": "All India",
        "Andaman and Nicobar Islands": "Andaman and Nicobar",
        "Chattisgarh": "Chhattisgarh",
        "North East1": "North East",
        "North East2": "North East",
        "North East 1": "North East",
        "North East 2": "North East",
        "Uttaranchal": "Uttarakhand",
        "Tamil Nadu (including Chennai)": "Tamil Nadu",
        "Chennai": "Tamil Nadu",
    }
    df["circle"] = df["circle"].str.strip().replace(circle_mapping)
    
    # Filter for wireless and circle level
    df['type_of_connection'] = df['type_of_connection'].str.strip().str.lower()
    df = df[df['type_of_connection'] == 'wireless']
    df = df[df['circle'] != 'All India']
    
    # Month order mapping
    MONTH_ORDER = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    df["month_num"] = df["month"].map({m: i + 1 for i, m in enumerate(MONTH_ORDER)})
    df = df.dropna(subset=["month_num"])
    df["month_num"] = df["month_num"].astype(int)
    
    # Date column
    df["date"] = df["year"].astype(str) + "-" + df["month_num"].apply(lambda x: f"{x:02d}") + "-01"
    
    # Sort for lag calculations
    df = df.sort_values(by=['circle', 'service_provider', 'year', 'month_num'])
    
    # Grouped lag calculations
    group = df.groupby(['circle', 'service_provider'])
    df['subscribers_lag_1'] = group['value'].shift(1).fillna(0)
    df['subscribers_lag_3'] = group['value'].shift(3).fillna(0)
    df['subscribers_lag_6'] = group['value'].shift(6).fillna(0)
    df['subscribers_lag_12'] = group['value'].shift(12).fillna(0)
    
    # Growth rates
    df['mom_growth'] = ((df['value'] - df['subscribers_lag_1']) / df['subscribers_lag_1']).replace([np.inf, -np.inf], np.nan).fillna(0)
    df['yoy_growth'] = ((df['value'] - df['subscribers_lag_12']) / df['subscribers_lag_12']).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Volatility calculations (rolling std of MoM growth)
    df['growth_volatility_3'] = group['mom_growth'].transform(lambda x: x.rolling(3, min_periods=1).std()).fillna(0)
    df['growth_volatility_6'] = group['mom_growth'].transform(lambda x: x.rolling(6, min_periods=1).std()).fillna(0)
    df['growth_volatility_12'] = group['mom_growth'].transform(lambda x: x.rolling(12, min_periods=1).std()).fillna(0)
    
    # Trend over 12 months
    df['trend_12m'] = df['value'] - df['subscribers_lag_12']
    
    # Market shares and ranks
    df['total_circle_subscribers'] = df.groupby(['circle', 'year', 'month_num'])['value'].transform('sum')
    df['market_share'] = df['value'] / df['total_circle_subscribers']
    df['market_rank'] = df.groupby(['circle', 'year', 'month_num'])['value'].rank(ascending=False, method='first')
    
    # Share gap to leader
    max_share = df.groupby(['circle', 'year', 'month_num'])['market_share'].transform('max')
    df['share_gap_leader'] = max_share - df['market_share']
    
    # Relative performance
    mean_share = df.groupby(['circle', 'year', 'month_num'])['market_share'].transform('mean')
    df['relative_performance'] = df['market_share'] / mean_share
    
    # Circle type (Metro vs Non-Metro)
    df['circle_type'] = np.where(df['circle'].isin(['Delhi', 'Mumbai', 'Kolkata']), 'Metro', 'Non-Metro')
    
    # Is wireless flag
    df['is_wireless'] = 1.0
    
    # Market size category (sum of circle subscribers over time)
    df['market_size_category'] = df.groupby('circle')['total_circle_subscribers'].transform('sum')
    
    # Geographic diversity (number of circles the operator operates in)
    df['operator_geographic_diversity'] = df.groupby('service_provider')['circle'].transform('nunique').astype(float)
    
    # Save the output to root and notebooks directory
    df.to_csv(project_root / "Telecom_Features.csv", index=False)
    df.to_csv(project_root / "notebooks" / "Telecom_Features.csv", index=False)
    
    print(f"Generated successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    print("Saved as Telecom_Features.csv in root and notebooks/ folders.")

if __name__ == '__main__':
    main()
