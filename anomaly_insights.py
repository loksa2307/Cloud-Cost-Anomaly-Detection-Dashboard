import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# ---------------------------------------------
# STEP 1: Load the CSV
# ---------------------------------------------
df = pd.read_csv("cloud_cost_data.csv")   # <-- change name if needed
df['Date'] = pd.to_datetime(df['Date'])

# ---------------------------------------------
# STEP 2: Create daily aggregated cost
# ---------------------------------------------
daily_cost = df.groupby('Date')['Cost'].sum().reset_index()

# daily_cost_df is needed for ML + plotting
daily_cost_df = daily_cost.copy()

# ---------------------------------------------
# STEP 3: Fit IsolationForest model
# ---------------------------------------------
model = IsolationForest(contamination=0.05, random_state=42)
daily_cost_df["anomaly"] = model.fit_predict(daily_cost_df[['Cost']])

# ---------------------------------------------
# STEP 4: Plot data + anomalies
# ---------------------------------------------
plt.figure(figsize=(12, 5))

# Main line
plt.plot(
    daily_cost_df['Date'],
    daily_cost_df['Cost'],
    label="Daily Cost",
    color="blue"
)

# Red dots for anomalies
plt.scatter(
    daily_cost_df[daily_cost_df['anomaly'] == -1]['Date'],
    daily_cost_df[daily_cost_df['anomaly'] == -1]['Cost'],
    color='red',
    label="Anomaly",
    s=50
)

plt.xlabel("Date")
plt.ylabel("Cost")
plt.title("Cloud Cost Anomaly Detection")
plt.legend()
plt.tight_layout()
plt.show()
