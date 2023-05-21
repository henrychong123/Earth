import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Read the dataset from a CSV file
data = pd.read_csv('temperature_data.csv')

# Convert the 'datetime' column to datetime format
data['datetime'] = pd.to_datetime(data['datetime'])

# Sort the data by datetime
data = data.sort_values('datetime')

# Create a plot of average temperature over time
plt.figure(figsize=(12, 6))
plt.plot(data['datetime'], data['average_temperature'], color='blue')
plt.xlabel('Datetime')
plt.ylabel('Average Temperature')
plt.title('Average Temperature over Time')
plt.xticks(rotation=45)
plt.show()

# Convert datetime to numerical values for regression
data['datetime_numeric'] = data['datetime'].apply(lambda x: x.timestamp())

# Extract the features (datetime_numeric) and target (average_temperature)
X = data[['datetime_numeric']]
y = data['average_temperature']

# Create a linear regression model for temperature prediction
model_temperature = LinearRegression()
model_temperature.fit(X, y)

# Generate future datetime values for prediction
last_datetime = data['datetime'].max()
future_datetimes = pd.date_range(last_datetime, periods=10, freq='M') + timedelta(days=1)
future_datetimes_numeric = future_datetimes.map(lambda x: x.timestamp()).values.reshape(-1, 1)

# Predict future temperatures
future_temperatures = model_temperature.predict(future_datetimes_numeric)

# Create a plot of future temperature predictions
plt.figure(figsize=(12, 6))
plt.plot(data['datetime'], data['average_temperature'], color='blue', label='Historical Data')
plt.plot(future_datetimes, future_temperatures, color='red', linestyle='--', label='Future Predictions')
plt.xlabel('Datetime')
plt.ylabel('Average Temperature')
plt.title('Future Temperature Predictions')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Create a linear regression model for temperature uncertainty prediction
model_uncertainty = LinearRegression()
model_uncertainty.fit(X, data['average_temperature_uncertainty'])

# Predict future temperature uncertainties
future_uncertainties = model_uncertainty.predict(future_datetimes_numeric)

# Create a plot of future temperature uncertainty predictions
plt.figure(figsize=(12, 6))
plt.plot(data['datetime'], data['average_temperature_uncertainty'], color='blue', label='Historical Data')
plt.plot(future_datetimes, future_uncertainties, color='red', linestyle='--', label='Future Predictions')
plt.xlabel('Datetime')
plt.ylabel('Temperature Uncertainty')
plt.title('Future Temperature Uncertainty Predictions')
plt.xticks(rotation=45)
plt.legend()
plt.show()
