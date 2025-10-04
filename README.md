# Ex.No: 6               HOLT WINTERS METHOD
### Date: 04-10-25



### AIM:
 To implement the Holt Winters Method Model using Python
### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
data = pd.read_csv("/content/car_price_prediction.csv")

# Group by Production Year and take average price (time series)
data_yearly = data.groupby("Prod. year")["Price"].mean()

# Convert to datetime index (treat year as Jan 1st of that year)
ts_data = pd.Series(data_yearly.values, 
                    index=pd.to_datetime(data_yearly.index, format='%Y'))

print(ts_data.head())

# Plot original time series
ts_data.plot(title="Average Car Price per Production Year")
plt.xlabel("Year")
plt.ylabel("Average Price")
plt.show()

# Scale the data
scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(ts_data.values.reshape(-1, 1)).flatten(),
                        index=ts_data.index)

scaled_data = scaled_data + 1   # ensure positive values for multiplicative model
scaled_data.plot(title="Scaled Data")
plt.show()

# Decompose to check trend/seasonality
decomposition = seasonal_decompose(ts_data, model="additive", period=1)
decomposition.plot()
plt.show()

# Train-Test Split
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Holt-Winters Model
model = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()

# Forecast on test set
test_predictions = model.forecast(len(test_data))

ax = train_data.plot(label="Train")
test_data.plot(ax=ax, label="Test")
test_predictions.plot(ax=ax, label="Forecast")
plt.legend()
plt.title("Holt-Winters Forecast on Car Prices (by Year)")
plt.show()

# Evaluate performance
rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
mae = mean_absolute_error(test_data, test_predictions)
print("RMSE:", rmse)
print("MAE:", mae)

# Final model on full data
final_model = ExponentialSmoothing(scaled_data, trend='add', seasonal='mul', seasonal_periods=3).fit()
future_forecast = final_model.forecast(5)  # predict next 5 years

ax = scaled_data.plot(label="Historical")
future_forecast.plot(ax=ax, label="Future Forecast")
plt.legend()
plt.title("Future Car Price Forecast (Holt-Winters)")
plt.show()

```
### OUTPUT:

<img width="973" height="657" alt="image" src="https://github.com/user-attachments/assets/4ffc492d-0fb6-4e9b-923e-d40bb092bec9" />
<img width="903" height="535" alt="image" src="https://github.com/user-attachments/assets/2a1a8362-f607-4268-98e5-26a741413d4a" />
<img width="906" height="582" alt="image" src="https://github.com/user-attachments/assets/727a6889-e3be-440f-b23b-99868a23d421" />


TEST_PREDICTION

<img width="839" height="552" alt="image" src="https://github.com/user-attachments/assets/509ffcb9-e061-445b-8805-ea02c77b43a6" />

FINAL_PREDICTION
<img width="801" height="513" alt="image" src="https://github.com/user-attachments/assets/436dd220-0757-4a49-aa1a-5589429978b4" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
