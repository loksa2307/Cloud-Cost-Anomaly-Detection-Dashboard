import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cloud_cost_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

daily_cost = df.groupby('Date')['Cost'].sum()
daily_cost.plot(figsize=(10,5), title='Daily Cloud Cost')
plt.show()
