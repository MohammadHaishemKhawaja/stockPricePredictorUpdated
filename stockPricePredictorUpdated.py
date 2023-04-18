import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

symbol = input("Enter a stock symbol (e.g. AAPL): ")

df = yf.download(symbol, start="2022-04-17", end="2023-04-17")

train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

X_train = train_data.drop(columns=["Close"])
y_train = train_data["Close"]
X_test = test_data.drop(columns=["Close"])
y_test = test_data["Close"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

plt.plot(test_data.index, y_test, label="Actual")
plt.plot(test_data.index, y_pred, label="Predicted")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

actual_price = df.iloc[-1]['Close']
print(f"Actual price for {symbol} on {df.index[-1].strftime('%Y-%m-%d')}: {actual_price:.2f}")
