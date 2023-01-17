# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.stattools import adfuller
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
%matplotlib inline
import itertools
data_matrix = pd.read_csv("../input/LaturRains_1965_2002.csv",sep="\t")
data_matrix.head()
data_matrix.set_index('Year', inplace=True)
data_matrix.head()
data_matrix = data_matrix.transpose()
data_matrix
dates = pd.date_range(start='1965-01', freq='MS', periods=len(data_matrix.columns)*12)
dates
plt.figure(figsize=(13,7))
plt.plot(data_matrix)
plt.xlabel('Year')
plt.ylabel('Precipitation(mm)')
plt.title('Month vs Precipitation across all years')
plt.figure(figsize=(10,5))
# type(data_matrix)
plt.boxplot(data_matrix)
plt.xlabel('Month')
plt.ylabel('Precipitation(mm)')
plt.title('Month vs Precipitation across all years')
print("Original: ",data_matrix.shape)
rainfall_data_matrix_np = data_matrix.transpose().as_matrix()
shape = rainfall_data_matrix_np.shape
rainfall_data_matrix_np = rainfall_data_matrix_np.reshape((shape[0] * shape[1], 1))
print("After Transformation: ",rainfall_data_matrix_np.shape)
rainfall_data = pd.DataFrame({'Precipitation': rainfall_data_matrix_np[:,0]})
rainfall_data.set_index(dates, inplace=True)

test_data = rainfall_data.ix['1995': '2002']
train_data = rainfall_data.ix[: '1994']
train_data.tail() # 1965-1994
test_data.tail() # 1995-2002
plt.figure(figsize=(20,5))
plt.plot(rainfall_data, color='blue')
plt.xlabel('Year')
plt.ylabel('Precipitation(mm)')
plt.title('Precipitation in mm')
fig, axes = plt.subplots(2, 2, sharey=False, sharex=False)
fig.set_figwidth(14)
fig.set_figheight(8)
axes[0][0].plot(rainfall_data.index, rainfall_data, label='Original')
axes[0][0].plot(rainfall_data.index, rainfall_data.rolling(window=4).mean(), label='4-Months Rolling Mean')
axes[0][0].set_xlabel("Years")
axes[0][0].set_ylabel("Precipitation in mm")
axes[0][0].set_title("4-Months Moving Average")
axes[0][0].legend(loc='best')
############
axes[0][1].plot(rainfall_data.index, rainfall_data, label='Original')
axes[0][1].plot(rainfall_data.index, rainfall_data.rolling(window=8).mean(), label='8-Months Rolling Mean')
axes[0][1].set_xlabel("Years")
axes[0][1].set_ylabel("Precipitation in mm")
axes[0][1].set_title("8-Months Moving Average")
axes[0][1].legend(loc='best')
############
axes[1][0].plot(rainfall_data.index, rainfall_data, label='Original')
axes[1][0].plot(rainfall_data.index, rainfall_data.rolling(window=12).mean(), label='12-Months Rolling Mean')
axes[1][0].set_xlabel("Years")
axes[1][0].set_ylabel("Precipitation in mm")
axes[1][0].set_title("12-Months Moving Average")
axes[1][0].legend(loc='best')
############
axes[1][1].plot(rainfall_data.index, rainfall_data, label='Original')
axes[1][1].plot(rainfall_data.index, rainfall_data.rolling(window=16).mean(), label='16-Months Rolling Mean')
axes[1][1].set_xlabel("Years")
axes[1][1].set_ylabel("Precipitation in mm")
axes[1][1].set_title("16-Months Moving Average")
axes[1][1].legend(loc='best')
# ############
# axes[0][1].plot(rainfall_data.index, rainfall_data, label='Original')
# axes[0][1].plot(rainfall_data.index, rainfall_data.rolling(window=14).mean(), label='4-Months Rolling Mean')
# axes[0][1].set_xlabel("Years")
# axes[0][1].set_ylabel("Number of Tractor's Sold")
# axes[0][1].set_title("14-Months Moving Average")
# axes[0][1].legend(loc='best')
plt.tight_layout()
plt.show()
#Determing rolling statistics
rolmean = rainfall_data.rolling(window=12).mean()
rolstd = rainfall_data.rolling(window=12).std()

#Plot rolling statistics:
orig = plt.plot(rainfall_data, label='Original')
mean = plt.plot(rolmean, label='Rolling Mean')
std = plt.plot(rolstd, label = 'Rolling Std',color='green')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)

#dickey-fuller test
def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
adf_test(rainfall_data.Precipitation) 
decomposition = sm.tsa.seasonal_decompose(rainfall_data, model='additive')
fig = decomposition.plot()
fig.set_figwidth(12)
fig.set_figheight(8)
fig.suptitle('Decomposition of time series')
plt.show()
fig, axes = plt.subplots(1, 2, sharey=False, sharex=False)
fig.set_figwidth(12)
fig.set_figheight(4)
smt.graphics.plot_acf(rainfall_data, lags=30, ax=axes[0], alpha=0.5)
smt.graphics.plot_pacf(rainfall_data, lags=30, ax=axes[1], alpha=0.5)
plt.tight_layout()
#differencing with a factor of 12
diff_12_data = rainfall_data.diff(periods=12)
diff_12_data.dropna(inplace=True)

plt.plot(diff_12_data)
# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, d and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
pdq
seasonal_pdq 
best_aic = np.inf
best_pdq = None
best_seasonal_pdq = None
temp_model = None

for param in pdq:
    for param_seasonal in seasonal_pdq:        
        try:
            temp_model = sm.tsa.statespace.SARIMAX(train_data,
                                             order = param,
                                             seasonal_order = param_seasonal,
                                             enforce_stationarity=True,
                                             enforce_invertibility=True)
            results = temp_model.fit()
#             print("AIC for SARIMA{}x{}12 model - AIC:{}".format(param, param_seasonal, results.aic))
            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
        except:
            continue

print("")
print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))

best_pdq = (0, 0, 0)
best_seasonal_pdq = (1, 1, 1, 12)
#building the model with the best set of parameters obtained above
best_model = sm.tsa.statespace.SARIMAX(train_data,
                                      order=best_pdq,
                                      seasonal_order=best_seasonal_pdq,
                                      enforce_stationarity=True,
                                      enforce_invertibility=True)
best_results = best_model.fit()
print(best_results.summary().tables[0])
pred_dynamic = best_results.get_prediction(start=pd.to_datetime('1985-01-01'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()
rainfall_predicted = pred_dynamic.predicted_mean
rainfall_truth = rainfall_data['1985':'1994'].Precipitation
rainfall_predicted.shape
# print(rainfall_predicted
 #Plot the actual values.
axis_plt = train_data['1985':'1999'].plot(label='Observed', figsize=(10, 6))

# Plot the predicted values.
pred_dynamic.predicted_mean[:'1999'].plot(ax=axis_plt, label='Dynamic Forecast')

# Plot confidence values and fill it with some colour.
#axis_plt.fill_between(pred_dynamic_ci.index, pred_dynamic_ci.iloc[:, 0], pred_dynamic_ci.iloc[:, 1], color='k', alpha=0.1)
#axis_plt.fill_betweenx(axis_plt.get_ylim(), pd.to_datetime('1990'), pd.to_datetime('2000'), alpha=0.1, zorder=-1)

# Set labels.
axis_plt.set_xlabel('Years')
axis_plt.set_ylabel('Precipitation')

# Put legend on the plot at the best place it fits.
plt.legend(loc='best')
# Get forecast 96 steps (8 years) ahead in future
n_steps = 96
pred_uc_95 = best_results.get_forecast(steps=n_steps, alpha=0.05) # alpha=0.05 95% CI
index = pd.date_range(train_data.index[-1] + 1, periods=n_steps, freq='MS')
forecast_data = pd.DataFrame(np.column_stack([pred_uc_95.predicted_mean]), 
                     index=index, columns=['forecast'])

forecast_data.head()
# Create the plot.
plt.figure(figsize = (15, 5))
plt.plot(rainfall_data['1985':], label = "True value")
plt.plot(pred_dynamic.predicted_mean[:'1999'], color='green',label = "Training set prediction")
plt.plot(forecast_data['forecast'],color='red', label = "Test set prediction")
plt.xlabel("Months")
plt.ylabel("Precipitation in mm")
plt.title("Latur Rainfall Data Prediction - ARIMA")
plt.legend()
plt.show()
import math
mse_train = math.sqrt(((rainfall_predicted - rainfall_truth) ** 2).mean())
print('Train RMSE {}'.format(round(mse_train, 4)))
mse_test = math.sqrt(((forecast_data['forecast'] - rainfall_data['1995':'2003'].Precipitation) ** 2).mean())
print('Test RMSE {}'.format(round(mse_test, 4)))
plot_df = pd.DataFrame({'':['train','test'], 'RMSE':[mse_train,mse_test]})
ax = plot_df.plot.bar(x='', y='RMSE', rot=0)
