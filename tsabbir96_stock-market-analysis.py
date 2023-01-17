import numpy as np
import pandas as pd
import os
os.listdir("../input/Data/Stocks/")

import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from sklearn.model_selection import train_test_split

from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

df = pd.read_csv("../input/Data/Stocks/tsla.us.txt")
df
df.shape

df[["Close"]].plot()
plt.title("Tesla")
plt.show()
dr = df.cumsum()
dr.plot()
plt.title("Tesla Cumulative Sum/ Cumulative return")
plt.show()
lag_plot(df["Open"], lag =5)
plt.title("Tesla Autocorrelation plot")
train_data, test_data = df[0:int(len(df)*0.80)], df[int(len(df)*0.80):]


plt.figure(figsize = (12,7))
plt.title("Tesla")
plt.xlabel("Dates")
plt.ylabel("Prices")

plt.plot(df["Open"],"blue", label = "Training Data")
plt.plot(test_data["Open"],"green", label = "Testing Data")

plt.xticks(np.arange(0,1857,300), df["Date"][0:1857:300])
plt.legend()
def smape(y_true,y_pred):
    smape = np.mean((np.abs(y_pred - y_true)*100)/((np.abs(y_pred + y_true))/2))
    return smape
train_ar = train_data["Open"].values
test_ar = test_data["Open"].values

history = [x for x in train_ar]


predictions = list()

for t in range(len(test_ar)):

    model = ARIMA(history, order = (5,1,0))
    model_fit = model.fit(disp = 0)
    output = model_fit.forecast()
    predicted_value = output[0]
    predictions.append(predicted_value)
    true_value = test_ar[t]
    history.append(true_value)

        
mse = mean_squared_error(test_ar, predictions)
print("MSE = ", mse)
smape = smape(test_ar,predictions)
print("SMAPE = ", smape)
plt.figure(figsize=(12,7))
plt.plot(df["Open"], "green", color = "blue", label = "Training Data")
plt.plot(test_data.index, predictions, color = "red", marker = "o", linestyle = "dashed", label = "Predicted Price")
plt.plot(test_data.index, test_data["Open"], color = "green", label = "Actual Price")
plt.title("Tesla Prices Predictions")
plt.xlabel("Dates")
plt.ylabel("Prices")
plt.xticks(np.arange(0,1857,300), df["Date"][0:1857:300])
plt.legend()
plt.figure(figsize = (12,7))
plt.plot(test_data.index, predictions, color = "green", marker = "o", linestyle = "dashed", label= 'Predicted Price ')
plt.plot(test_data.index, test_data['Open'],color = 'red', label = "Actual Price")
plt.xticks(np.arange(1486, 1856, 60), df["Date"][1486:1856:60])
plt.title("Title Prices Prediction")
plt.xlabel("Dates")
plt.ylabel("Prices")
plt.legend()
# def forecast_price(data, 60):
#     #preprocessed
#     #chose best k
#     #trained tested with arima
#     #predicted the results
#     #created a graph
#     return graph
# datalists = ["tesla.csv","apple.csv", "google.csv"]

# for data in datalists:
#     forecast_price(data,60)
