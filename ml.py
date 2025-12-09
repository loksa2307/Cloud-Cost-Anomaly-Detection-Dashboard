import pandas as pd
from sklearn.ensemble import IsolationForest

# ---------------------------------------
# STEP 1: Load your dataset
# ---------------------------------------
df = pd.read_csv("cloud_cost_data.csv")   # <-- replace with your file name
df['Date'] = pd.to_datetime(df['Date'])

# Expected columns: Date, Cost
# If your file has more columns (Service, Region, etc.) it's fine.

# ---------------------------------------
# STEP 2: Create daily aggregated cost
# ---------------------------------------
daily_cost = df.groupby('Date')['Cost'].sum()

# Convert to DataFrame for ML
daily_cost_df = daily_cost.reset_index()

# ---------------------------------------
# STEP 3: Fit Isolation Forest model
# ---------------------------------------
model = IsolationForest(contamination=0.05, random_state=42)

# Model trains on only the Cost column
daily_cost_df['anomaly'] = model.fit_predict(daily_cost_df[['Cost']])

# ---------------------------------------
# STEP 4: Print anomalies
# ---------------------------------------
anomalies = daily_cost_df[daily_cost_df['anomaly'] == -1]
print("Anomaly Days:")
print(anomalies)
