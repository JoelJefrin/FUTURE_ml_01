
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Step 2: Load Data
df = pd.read_csv('Sample - Superstore.csv', encoding='ISO-8859-1')
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Step 3: Aggregate Monthly Sales
monthly_sales = df.groupby(pd.Grouper(key='Order Date', freq='MS'))['Sales'].sum().reset_index()
monthly_sales.columns = ['ds', 'y']  # Prophet format

# Step 4: Train Prophet Model
model = Prophet()
model.fit(monthly_sales)

# Step 5: Forecast Next 12 Months
future = model.make_future_dataframe(periods=12, freq='MS')
forecast = model.predict(future)

# Optional: Plot
fig = model.plot(forecast)
plt.title("Monthly Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

# Step 6: Export Forecast for Power BI
forecast_export = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast_export = forecast_export.rename(columns={
    'ds': 'Date', 'yhat': 'Predicted_Sales', 
    'yhat_lower': 'Lower_Bound', 'yhat_upper': 'Upper_Bound'
})
forecast_export.to_csv("forecast_output.csv", index=False)

# Optional: Merge actuals for Power BI comparison
merged = pd.merge(monthly_sales, forecast_export, left_on='ds', right_on='Date', how='outer')
merged.to_csv("sales_forecast_merged.csv", index=False)
