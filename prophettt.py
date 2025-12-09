# prophettt.py

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Step 1: Read or create your dataset
df = pd.read_csv("cloud_cost_data.csv")   # Replace with your actual file name
df['Date'] = pd.to_datetime(df['Date'])   # Ensure date column is datetime

# Step 2: Prepare daily cost (aggregate by date)
daily_cost = df.groupby('Date')['Cost'].sum().reset_index()

# Step 3: Rename columns for Prophet
df_prophet = daily_cost.rename(columns={'Date': 'ds', 'Cost': 'y'})

# Step 4: Create and train model
model = Prophet()
model.fit(df_prophet)

# Step 5: Predict next 30 days
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Step 6: Plot forecast
model.plot(forecast)
plt.title("Cloud Cost Forecast")
plt.show()
