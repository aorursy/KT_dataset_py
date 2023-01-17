import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
!pip install pmdarima -U
from pmdarima.arima import auto_arima
!pip install arch -U
from arch import arch_model
!pip install yfinance -U
import yfinance
import warnings
warnings.filterwarnings("ignore")
sns.set()
raw_data = yfinance.download (tickers = "^GSPC ^FTSE ^N225 ^GDAXI", start = "1994-01-07", 
                              end = "2019-09-01", interval = "1d", group_by = 'ticker', auto_adjust = True, treads = True)
df_comp = raw_data.copy()
df_comp['spx'] = df_comp['^GSPC'].Close[:]
df_comp['dax'] = df_comp['^GDAXI'].Close[:]
df_comp['ftse'] = df_comp['^FTSE'].Close[:]
df_comp['nikkei'] = df_comp['^N225'].Close[:]
df_comp = df_comp.iloc[1:]
del df_comp['^N225']
del df_comp['^GSPC']
del df_comp['^GDAXI']
del df_comp['^FTSE']
df_comp=df_comp.asfreq('b')
df_comp=df_comp.fillna(method='ffill')
df_comp['ret_spx'] = df_comp.spx.pct_change(1).mul(100)
df_comp['ret_ftse'] = df_comp.ftse.pct_change(1).mul(100)
df_comp['ret_dax'] = df_comp.dax.pct_change(1).mul(100)
df_comp['ret_nikkei'] = df_comp.nikkei.pct_change(1).mul(100)
df_comp['norm_ret_spx'] = df_comp.ret_spx.div(df_comp.ret_spx[1])*100
df_comp['norm_ret_ftse'] = df_comp.ret_ftse.div(df_comp.ret_ftse[1])*100
df_comp['norm_ret_dax'] = df_comp.ret_dax.div(df_comp.ret_dax[1])*100
df_comp['norm_ret_nikkei'] = df_comp.ret_nikkei.div(df_comp.ret_nikkei[1])*100
size = int(len(df_comp)*0.8)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]
model_ar = ARIMA(df.ftse, order = (1,0,0))
results_ar = model_ar.fit()
df.tail()
start_date = "2014-07-16"
end_date = "2015-01-01"
df_pred['predictions'] = results_ar.predict(start = start_date, end = end_date)
df_pred.predictions[start_date:end_date].plot(figsize = (20,5), color = "red")
df_test.ftse[start_date:end_date].plot(color = "blue")
plt.title("Predictions v/s Actuals", size = 24)
plt.legend()
plt.show()
model_ret_ar = ARIMA(df.ret_ftse[1:], order = (5,0,0))
results_ret_ar = model_ret_ar.fit()
df_pred_ret_ar = results_ret_ar.predict(start = start_date, end = end_date)
df_pred_ret_ar[start_date:end_date].plot(figsize = (20,5), color = "red")
df_test.ret_ftse[start_date:end_date].plot(color = "blue")
plt.title("Predictions vs Actuals(Returns) | AR", size = 24)
plt.show()
model_ret_ma = ARIMA(df.ret_ftse[1:], order = (0,0,5))
results_ret_ma = model_ret_ma.fit()
df_pred_ret_ma = results_ret_ma.predict(start = start_date, end = end_date)
df_pred_ret_ma[start_date:end_date].plot(figsize = (20,5), color = "red")
df_test.ret_ftse[start_date:end_date].plot(color = "blue")
plt.title("Predictions vs Actuals(Returns) | MA", size = 24)
plt.show()
model_ret_arma = ARIMA(df.ret_ftse[1:], order = (4,0,5))
results_ret_arma = model_ret_arma.fit()

df_pred_ret_arma = results_ret_arma.predict(start = start_date, end = end_date)

df_pred_ret_arma[start_date:end_date].plot(figsize = (20,5), color = "red")
df_test.ret_ftse[start_date:end_date].plot(color = "blue")
plt.title("Predictions vs Actuals(Returns) | ARMA", size = 24)
plt.show()
model_ret_armax = ARIMA(df.ret_ftse[1:], exog = df[['ret_spx', 'ret_dax', 'ret_nikkei']][1:], order = (1,0,1))
results_ret_armax = model_ret_armax.fit()

df_pred_ret_armax = results_ret_armax.predict(start = start_date, end = end_date,
                                             exog = df_test[['ret_spx', 'ret_dax', 'ret_nikkei']][start_date:end_date])

df_pred_ret_armax[start_date:end_date].plot(figsize = (20,5), color = "red")
df_test.ret_ftse[start_date:end_date].plot(color = "blue")
plt.title("Predictions vs Actuals(Returns)|ARMAX", size = 24)
plt.show()
df_test['int_ftse_ret'] = df_test.ftse.diff(1) 
model_arima = ARIMA(df.ret_ftse[1:], order = (1,1,1))
results_arima = model_arima.fit()

df_pred_arima = results_arima.predict(start = start_date, end = end_date)

df_pred_arima[start_date:end_date].plot(figsize = (20,5), color = "red")
df_test.ret_ftse[start_date:end_date].plot(color = "blue")
plt.title("Predictions vs Actuals(Returns) | ARIMA", size = 24)
plt.show()
model_arimax = ARIMA(df.ret_ftse[1:], exog = df[['ret_spx', 'ret_dax', 'ret_nikkei']][1:], order = (1,1,1))
results_arimax = model_arimax.fit()

df_pred_arimax = results_arimax.predict(start = start_date, end = end_date,
                                             exog = df_test[['ret_spx', 'ret_dax', 'ret_nikkei']][start_date:end_date])

df_pred_arimax[start_date:end_date].plot(figsize = (20,5), color = "red")
df_test.ret_ftse[start_date:end_date].plot(color = "blue")
plt.title("Predictions vs Actuals(Returns)|ARIMAX", size = 24)
plt.show()
model_ret_sarma = SARIMAX(df.ret_ftse[1:], order = (3,0,4), seasonal_order = (3,0,2,5))
results_ret_sarma = model_ret_sarma.fit()

df_pred_ret_sarma = results_ret_sarma.predict(start = start_date, end = end_date)

df_pred_ret_sarma[start_date:end_date].plot(figsize = (20,5), color = "red")
df_test.ret_ftse[start_date:end_date].plot(color = "blue")
plt.title("Predictions vs Actuals(Returns) | SARMA", size = 24)
plt.show()
model_ret_sarimax = SARIMAX(df.ret_ftse[1:], exog = df[['ret_spx', 'ret_dax', 'ret_nikkei']][1:], order = (3,0,4),
                           seasonal_order = (3,0,2,5))
results_ret_sarimax = model_ret_sarimax.fit()

df_pred_ret_sarimax = results_ret_sarimax.predict(start = start_date, end = end_date, exog =  df_test[['ret_spx', 'ret_dax', 'ret_nikkei']][start_date:end_date])

df_pred_ret_sarimax[start_date:end_date].plot(figsize = (20,5), color = "red")
df_test.ret_ftse[start_date:end_date].plot(color = "blue")
plt.title("Predictions vs Actuals(Returns) | SARIMAX", size = 24)
plt.show()
model_auto = auto_arima(df.ret_ftse[1:], exogenous = df[['ret_spx', 'ret_dax', 'ret_nikkei']][1:], m = 5, max_p = 5, max_q = 5, max_P = 5, max_Q = 5)
df_auto_pred = pd.DataFrame(model_auto.predict(n_periods = len(df_test[start_date:end_date]),
                           exogenous = df_test[['ret_spx', 'ret_dax', 'ret_nikkei']][start_date:end_date]), index = df_test[start_date:end_date].index)
df_auto_pred[start_date:end_date].plot(figsize = (20,5), color = "red")
df_test.ret_ftse[start_date:end_date].plot(color = "blue")
plt.title("Predictions vs Actuals(Returns) | AUTO ARIMA", size = 24)
plt.show()
mod_garch = arch_model(df_comp.ret_ftse[1:], vol = "GARCH", p = 1, q = 1, mean = "Constant", dist = "Normal" )
res_garch = mod_garch.fit(last_obs = start_date, update_freq = 10)
pred_garch = res_garch.forecast(horizon = 1, align = 'target')
pred_garch.residual_variance[start_date:].plot(figsize = (20,5), color = "red", zorder = 2)
df_test.ret_ftse.abs().plot(color = "blue", zorder = 1)
plt.title("Volatitliy Predictions| FTSE100 | (GARCH)", size = 24 )
plt.show()

from statsmodels.tsa.api import VAR
df_ret = df[['ret_spx', 'ret_dax', 'ret_ftse', 'ret_nikkei']][1:]
model_var_ret = VAR(df_ret)
model_var_ret.select_order(20)
results_var_ret = model_var_ret.fit(ic = 'aic')
results_var_ret.summary()
lag_order_ret = results_var_ret.k_ar
var_pred_ret = results_var_ret.forecast(df_ret.values[-lag_order_ret:], len(df_test[start_date:end_date]))

df_ret_pred = pd.DataFrame(data = var_pred_ret, index = df_test[start_date:end_date].index,
                                columns = df_test[start_date:end_date].columns[4:8])

df_ret_pred.ret_nikkei[start_date:end_date].plot(figsize = (20,5), color = "red")

df_test.ret_nikkei[start_date:end_date].plot(color = "blue")
plt.title("Real vs Prediction", size = 24)
plt.show()
results_var_ret.plot_forecast(1000)
plt.show()