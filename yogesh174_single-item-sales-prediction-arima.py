import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
mpl.rcParams["figure.figsize"] = [15,7]
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf,pacf
os.chdir("/kaggle/input/wallmart")
df = pd.read_csv("kaggle_subset.csv")

# Extracting sales of 'FOODS_1_001' item to df1
df1 = pd.DataFrame(np.zeros((1913,1)))
df1.columns = ['FOODS_1_001'] 
for i in range(1,30491):
    i = df.columns[i]
    item = i.split('_')[0] + '_' + i.split('_')[1] + '_' + i.split('_')[2]
    if item == 'FOODS_1_001':
        df1[item] += df[i]
df1.head()

df1['date'] = df['date']
df1 = df1.set_index('date')
df1.head()
# Droping zeros from dataset 
# Should do imputing later on
df1 = df1[df1['FOODS_1_001'] != 0]
# Log transformation
df1_log = np.log(df1)
df1_log.head()
# Finding moving averages and standard deviation
moving_avg_log = df1_log.rolling(window=365).mean()
moving_std_log = df1_log.rolling(window=365).std()
# Calculating difference b/w log of dataset values and movig average
logscaleMinusMA = df1_log - moving_avg_log
logscaleMinusMA.dropna(inplace=True)
# Calculating exponential weighted averages of log of dataset values
expDecayWeightAvg = df1_log.ewm(halflife=12, min_periods=0, adjust = True).mean()
dflogMinusExpDecay = df1_log - expDecayWeightAvg
plt.figure(figsize = [19, 5])
plt.plot()
plt.plot(dflogMinusExpDecay)
plt.plot(dflogMinusExpDecay.rolling(window=12).mean())
dftest  =  adfuller(dflogMinusExpDecay['FOODS_1_001'],autolag='AIC')
dfoutput = pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags used','#Observations'])
for key,value in dftest[4].items() :
    dfoutput["Critical values(%s)"%key] = value
dfoutput
# Integration part of ARIMA
# Difference is taken as 1
df1LogDiffShift = df1_log - df1_log.shift(1)
plt.plot(df1LogDiffShift)
# Droping NaN values and plotting
df1LogDiffShift.dropna(inplace = True)
plt.plot(df1LogDiffShift)
# Performing Dickey fuller test to check whether data is stationary or not
dftest  =  adfuller(df1LogDiffShift['FOODS_1_001'],autolag='AIC')
dfoutput = pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags used','#Observations'])
for key,value in dftest[4].items() :
    dfoutput["Critical values(%s)"%key] = value
dfoutput
# Determining P and Q values which are respectively AR and MA parts of ARIMA model 
# Plotting pacf and acf graph to get p and q values
lag_acf = acf(df1LogDiffShift, nlags=20)
lag_pacf = pacf(df1LogDiffShift, nlags=20, method ='ols')

plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df1LogDiffShift)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df1LogDiffShift)),linestyle='--',color='gray')
plt.title('ACF')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df1LogDiffShift)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df1LogDiffShift)),linestyle='--',color='gray')
plt.title('PACF')
plt.tight_layout()
# Fitting the ARIMA model
model = ARIMA(df1_log,order=(1,1,1))
results_AR = model.fit(disp=-1)
results_AR.fittedvalues
plt.plot(df1LogDiffShift['FOODS_1_001'])
plt.plot(results_AR.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues - df1LogDiffShift['FOODS_1_001'])**2))
predictions_AR_diff = pd.Series(results_AR.fittedvalues , copy= True)
predictions_AR_diff_cumsum = predictions_AR_diff.cumsum()
predictions_AR_log = pd.Series(df1_log['FOODS_1_001'].iloc[0], index = df1_log.index)
predictions_AR_log = predictions_AR_log.add(predictions_AR_diff_cumsum,fill_value=0)
predictions_AR_log.head()
predictions_AR = np.exp(predictions_AR_log)
plt.plot(df1,color='black')
#plt.plot(predictions_AR, color='red')
predictions_AR
plt.plot(predictions_AR, color='red')
results_AR.plot_predict(1,1709+60)
x=results_AR.forecast(steps=60)