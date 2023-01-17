# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
%matplotlib inline
import plotly.graph_objects as go
import matplotlib as mpl
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
TSLA = pd.read_csv("../input/tesla-stock-data-from-2010-to-2020/TSLA.csv")
TSLA['Date'] = pd.to_datetime(TSLA['Date'])
TSLA.index = TSLA['Date']
TSLA.head(3)
Tesla=TSLA.rename(columns={'Adj Close': 'AdjClose'})
Tesla.head(3)
sns.pairplot(Tesla[["Open", "High", "Close", "Volume"]], diag_kind="kde")

fig = px.line(Tesla, x='Date', y='Volume')
fig.show()
fig = go.Figure(data=[go.Candlestick(
    x=Tesla['Date'],
    open=Tesla['Open'], high=Tesla['High'],
    low=Tesla['Low'], close=Tesla['Close'],
    increasing_line_color= 'cyan', decreasing_line_color= 'green'
)])

fig.show()
#Close price Autocorrelation visualization
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(16,6), dpi= 80)
plot_acf(Tesla.Close.tolist(), ax=ax1, lags=50)
plot_pacf(Tesla.Close.tolist(), ax=ax2, lags=20)


ax1.spines["top"].set_alpha(.3); ax2.spines["top"].set_alpha(.3)
ax1.spines["bottom"].set_alpha(.3); ax2.spines["bottom"].set_alpha(.3)
ax1.spines["right"].set_alpha(.3); ax2.spines["right"].set_alpha(.3)
ax1.spines["left"].set_alpha(.3); ax2.spines["left"].set_alpha(.3)

ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='both', labelsize=12)
Tesla_HPY = pd.DataFrame({'Date':Tesla['Date'], 'HPY':Tesla['Close'] / Tesla['Open']-1})
Tesla_HPY.head(3)
# Telsa HPY static statement 
Tesla_HPY.describe()
# Tesla visualization 
fig = go.Figure([go.Scatter(x=Tesla_HPY['Date'], y=Tesla_HPY['HPY'])])
fig.show()
from statsmodels.tsa.arima_model import ARIMA
Tesla['Time'] = Tesla.index-Tesla.index.mean()
Tesla["Time"]=Tesla.index-Tesla.index[0]
Tesla["Time"]=Tesla["Time"].dt.days
train_ml=Tesla.iloc[:int(Tesla.shape[0]*0.95)]
valid_ml=Tesla.iloc[int(Tesla.shape[0]*0.95):]
log_series=np.log(train_ml["Close"])
y_pred=valid_ml.copy()
model_arima=ARIMA(log_series,(5,2,3))
model_arima_fit=model_arima.fit()
prediction_arima=model_arima_fit.forecast(len(valid_ml))[0]
y_pred["ARIMA Prediction"]=list(np.exp(prediction_arima))
plt.figure(figsize=(10,5))
plt.plot(train_ml.index,train_ml["Close"],label="Train Set",marker='o')
plt.plot(valid_ml.index,valid_ml["Close"],label="Validation Set",marker='*')
plt.plot(y_pred["ARIMA Prediction"],label="ARIMA Model Prediction Set",marker='^')
plt.legend()
plt.xlabel("Date Time")
plt.ylabel('Close Price')
plt.title("Close Price ARIMA Model Forecasting")
plt.xticks(rotation=90)
from sklearn import preprocessing
import lightgbm as lgb
from sklearn.model_selection import KFold, GridSearchCV
train_ml.head(3)
X = np.array(train_ml["Time"]).reshape(-1,1)
y = np.array(train_ml["Close"]).reshape(-1,1)
kfold = KFold(n_splits=5, random_state = 2020, shuffle = True)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb.fit(X,y)
prediction_valid_bgm=model_lgb.predict(np.array(valid_ml["Time"]).reshape(-1,1))
print("Validation LightBGM prediction:",prediction_valid_bgm)
plt.figure(figsize=(11,6))
prediction_bgm=model_lgb.predict(np.array(train_ml["Time"]).reshape(-1,1))
plt.plot(train_ml["Close"],label="Actual Confirmed Cases")
plt.plot(train_ml.index,prediction_bgm, linestyle='--',label="Predicted Close Price using LightBGM",color='black')
plt.xlabel('Time')
plt.ylabel('Close')
plt.title("Close Price Linear Regression Prediction")
plt.xticks(rotation=90)
plt.legend()