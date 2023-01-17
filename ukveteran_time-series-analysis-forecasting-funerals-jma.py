import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/cusersmarildownloadsfuneralscsv/funerals.csv', delimiter=';',encoding = "ISO-8859-1") 
df.dataframeName = 'funerals.csv'
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')
df.head()
df1=df.drop(["postcode_area_of_last_known_address", "cost_recovered", "date_referred_to_treasury_solicitor","gender"], axis = 1)
df1
df2=df1.rename(columns={"date_of_death": "ds", "cost_of_funeral": "y"})
df2
df2['y'] = df2['y'].str.replace('\£','')
df2
print(df2.dtypes)
df2['y'] = df2['y'].str.replace(',', '').astype(float)
df2
df2.dropna()
# Sort the Order Date 
df2 = df2.sort_values('ds')

#print the sorted values
print(df2.head(1))

#check any missing values
df2.isnull().sum()
# grouping sales according to Order Date
df2.groupby('ds')['y'].sum().reset_index()

# min and max values of Order Date
print(df2['ds'].min())
print(df2['ds'].max())
#set 'Order Date' as index
df2 = df2.set_index('ds')
df2.index
df2.index = pd.to_datetime(df2.index)
z= df2['y'].resample('MS').mean()
z['2019':]
z.plot(figsize = (15, 6))
plt.show()
p = d = q = range(0, 2)

#take all possible combination for p, d and q
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(z, order = param, seasonal_order = param_seasonal, enforce_stationary = False,enforce_invertibility=False) 
            result = mod.fit()   
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, result.aic))
        except:
            continue
prediction = result.get_prediction(start = pd.to_datetime('2019-01-01'), dynamic = False)
prediction_ci = prediction.conf_int()
prediction_ci
ax = z['2014':].plot(label = 'observed')
prediction.predicted_mean.plot(ax = ax, label = 'One-step ahead Forecast', alpha = 0.7, figsize = (14, 7))
ax.fill_between(prediction_ci.index, prediction_ci.iloc[:, 0], prediction_ci.iloc[:, 1], color = 'k', alpha = 0.2)
ax.set_xlabel("Date")
ax.set_ylabel('Total Volume')
plt.legend()
plt.show()
ax = z['2019':].plot(label = 'observed')
prediction.predicted_mean.plot(ax = ax, label = 'One-step ahead Forecast', alpha = 0.7, figsize = (14, 7))
ax.fill_between(prediction_ci.index, prediction_ci.iloc[:, 0], prediction_ci.iloc[:, 1], color = 'k', alpha = 0.2)
ax.set_xlabel("Date")
ax.set_ylabel('Total Volume')
plt.legend()
plt.show()
z_hat = prediction.predicted_mean
z_truth = z['2019-01-01':]

mse = ((z_hat - z_truth) ** 2).mean()
rmse = np.sqrt(mse)
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(rmse, 2)))
pred_uc = result.get_forecast(steps = 100)
pred_ci = pred_uc.conf_int()

ax = z.plot(label = 'observed', figsize = (14, 7))
pred_uc.predicted_mean.plot(ax = ax, label = 'forecast')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color = 'k', alpha = 0.25)
ax.set_xlabel('Date')
ax.set_ylabel('')

plt.legend()
plt.show()