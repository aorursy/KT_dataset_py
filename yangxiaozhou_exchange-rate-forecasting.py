from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Function to read data into Kernel
def read_data(file):
    return pd.read_csv("../input/" + file + ".csv")
# import the packages and libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import seaborn as sns
%matplotlib inline

# Set global random seed
np.random.seed(7012)

# Set global plotting style
# sns.set_style("whitegrid", {"xtick.major.size": 8, "ytick.major.size": 8, 'legend.frameon': True})
# sns.set_palette("dark")

# sns.set_context("notebook", font_scale=1.2)

sns.set_palette('Set2', 10)
sns.palplot(sns.color_palette())
sns.set_context('paper', font_scale = 1.2)

# Read and set up dataframe
data = read_data('P2training')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True, drop=True)
data[:'1980-1-10']
jpy_usd = data['JPY/USD']
log_jpy_usd = np.log(jpy_usd)
jpy_usd_100 = jpy_usd.head(100)
jpy_usd_1000 = jpy_usd.head(1000)
fig = plt.figure(figsize=(10,6))
plt.plot('JPY/USD', data=data)
plt.legend()
plt.tight_layout()
# Visualise data in three different time periods
fig, ax = plt.subplots(3,1,figsize=(12,12))
ax[0].plot('JPY/USD', data=data[:'1984'])
ax[1].plot('JPY/USD', data=data['1984':'1988'])
ax[2].plot('JPY/USD', data=data['1989':])
plt.tight_layout()
# plt.savefig('jpy_usd_three_periods.pdf')
# Visualise data with other currencies
plt.figure(figsize=(12,6))
data['DEM/USD'].plot()
data['CHF/USD'].plot()
#data['FRF/USD'].plot()
#data['AUD/USD'].plot()
data['JPY/USD'].plot(secondary_y=True, c='b')
plt.legend(loc='best')
plt.tight_layout()
# SACF and SPACF
alpha=0.05
lags=100

fig, ax = plt.subplots(1,2,figsize=(12,6))
fig = sgt.plot_acf(jpy_usd, ax=ax[0], lags=lags, alpha=alpha, unbiased=True)
fig = sgt.plot_pacf(jpy_usd, ax=ax[1], lags=lags, alpha=alpha, method='ols')
# plt.savefig('sacf_n_spacf_plot.pdf')
# Lag plots of three lag level: 1, 10, 100
fig, ax = plt.subplots(1,3,figsize=(12,6))
pd.plotting.lag_plot(jpy_usd, lag=1, ax = ax[0])
pd.plotting.lag_plot(jpy_usd, lag=10, ax = ax[1])
pd.plotting.lag_plot(jpy_usd, lag=100, ax = ax[2])
plt.tight_layout()
# plt.savefig('three_lag_plots.pdf')
al = .81;
ewma = jpy_usd.ewm(alpha=al, min_periods=0, freq=None, adjust=False)
plt.figure(figsize=(12,6))
plt.plot(jpy_usd_100, label='Original')
plt.plot(ewma.mean(), color='red', label='Exponential Smoothing')
plt.legend(loc='best')
plt.tight_layout()
from bokeh.plotting import figure, show, output_notebook
output_notebook()
TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

# reduce data size
data = jpy_usd

p = figure(x_axis_type="datetime", tools=TOOLS, title="JPY/USD Exchange Rate")
p.xgrid.grid_line_color=None
p.ygrid.grid_line_alpha=0.5
p.xaxis.axis_label = 'Time'
p.yaxis.axis_label = 'Rate'

p.line(data.index, data, line_color="blue")
p.line(data.index,  ewma.mean(), line_color='red')

show(p)
# Find optimal alpha
alpha = np.linspace(0.01,1,num=10)
err = [];
for al in alpha:
    ewma = jpy_usd.ewm(alpha=al, min_periods=0, freq=None)
    pred = ewma.mean();
    diff = jpy_usd - pred.shift(1);
    err.append((diff ** 2).mean())
    
plt.plot(alpha, err)
optal = alpha[np.argmin(err)]
plt.axvline(x=optal, color='red')
print(optal, min(err))
plt.xlabel('Alpha')
plt.ylabel('MSE')
plt.title('Optimal Alpha Value for EWMA')
# plt.savefig('optimal_alpha_ewma.pdf')
pass
# given a series and alpha, return series of smoothed points
def double_exponential_smoothing(series, alpha, beta, L0, B0):
    result = []
    # No prediction for the first value
    result.append(np.nan)
    for n in range(0, len(series)):
        val = series[n]
        if n==0:
            level = alpha*val + (1-alpha)*(L0+B0);
            trend = beta*(level-L0) + (1-beta)*B0;
            last_level = level;
        else:
            level = alpha*val + (1-alpha)*(last_level+trend)
            trend = beta*(level-last_level) + (1-beta)*trend
            last_level = level;
            
        result.append(level+trend)
        
    # No prediction for the (n+1)th value
    return result[:-1]
a = 1;
b = .01;
series = jpy_usd.values
holt = double_exponential_smoothing(series, a, b, series[0], series[1]-series[0])
holt = pd.DataFrame({"holt": holt}, index=jpy_usd.index)

SSE = np.sum(np.power(jpy_usd.values[1:] - holt.dropna().holt.values, 2))
print("SSE: %.3f" % SSE)

fig = plt.figure(figsize=(12,6))
plt.plot(jpy_usd)
plt.plot(holt, label='Holt')
plt.legend(loc='best')
plt.tight_layout()
# Find the optimal alpha and beta
alpha = np.linspace(0.01,1,num=10)
beta = np.linspace(0.01,1,num=10)

series = jpy_usd_100.values
err = [];
min_err = 10000;
for al in alpha:
    for be in beta:
        holt = double_exponential_smoothing(series, al, be, series[0], series[1]-series[0])
        holt = pd.DataFrame({"holt": holt}, index=jpy_usd_100.index)
        pred = holt.dropna().holt.values;
        diff = jpy_usd_100.values[1:] - pred;
        error = np.sqrt((diff ** 2).mean())
        if min_err > error:
            min_err = error
            alpha_hat = al
            beta_hat = be

        err.append(error)
plt.plot(err, label='RMSE')
plt.legend()
plt.tight_layout()
print(alpha_hat, beta_hat)
plt.ylabel('RMSE')
plt.tight_layout()
def cal_cv_rmse(X, orig_data, diff_order, fold, p, d, q, log):
    """Calculate root mean square 1-step-ahead forecasting error
       based on timeseries split cross validation
       
       params:
       X: data after order differencing
       orig_data: original data (could be log-transformed)
       diff_order: order of differencing in the trans_data
       fold: cross validation fold
       p, d, q: int, params for ARIMA
       log: boolean, True is X is log-transformed data
       
       return:
       RMSE: list, list of RMSE for all folds
    """
    tscv = TimeSeriesSplit(n_splits=fold)

    RMSE = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        #y_train, y_test = y[train_index], y[test_index]
        model = ARIMA(X_train, order=(p, d, q))  
        results_ns = model.fit(disp=-1) 
        
        # forcast
        forecasts = results_ns.forecast(X_test.size)[0]

        # errors 
        errors = []
        
        # get last two values from the original space
        second_last = orig_data.loc[X_train.index][-2]
        last = orig_data.loc[X_train.index][-1]
        
        if diff_order == 1:
            # first prediction
            forecasts[0] = forecasts[0] + last
            if log:
                errors.append(np.exp(forecasts[0]) - np.exp(orig_data.loc[X_test.index][0]))
            else:
                errors.append(forecasts[0] - orig_data.loc[X_test.index][0])
            
            for i in range(diff_order, X_test.size):
                # to correct for first order differencing
                forecasts[i] = forecasts[i] + orig_data.loc[X_test.index][i-1]
                if log:
                    errors.append(np.exp(forecasts[i]) - np.exp(orig_data.loc[X_test.index][i])) 
                else:
                    errors.append(forecasts[i] - orig_data.loc[X_test.index][i])

        
        if diff_order == 2:
            # first two predictions
            pred_1 = forecasts[0] + 2*last - second_last
            pred_2 = forecasts[1] + 2*pred_1 - last
            forecasts[0] = pred_1
            forecasts[1] = pred_2
            if log:
                errors.append(np.exp(pred_1) - np.exp(orig_data.loc[X_test.index][0]))
                errors.append(np.exp(pred_2) - np.exp(orig_data.loc[X_test.index][1]))
            else:
                errors.append(pred_1 - orig_data.loc[X_test.index][0])
                errors.append(pred_2 - orig_data.loc[X_test.index][1])
            for i in range(diff_order, X_test.size):
                # to correct for second order differencing
                forecasts[i] = forecasts[i] + 2*orig_data.loc[X_test.index][i-1] - orig_data.loc[X_test.index][i-2]
                if log:
                    errors.append(np.exp(forecasts[i]) - np.exp(orig_data.loc[X_test.index][i]))    
                else:
                    errors.append(forecasts[i] - orig_data.loc[X_test.index][i])    
        
        RMSE.append(np.sqrt(np.mean(np.power(errors, 2))))              
        
    return RMSE
# 1st and 2nd Order Differencing
data_1diff = jpy_usd - jpy_usd.shift(1)
data_1diff.dropna(inplace=True)
data_2diff = jpy_usd - 2*jpy_usd.shift(1) + jpy_usd.shift(2)
data_2diff.dropna(inplace=True)

Kfold = 10
log = False

# Calculate Kfold cross validation RMSE of various ARIMA models
# d_1: 1st order differencing data is used
# 011: (p, d, q) order of ARIMA models
RMSE_d_1_011 = cal_cv_rmse(data_1diff, jpy_usd, 1, Kfold, 0, 1, 1, log)
RMSE_d_2_011 = cal_cv_rmse(data_2diff, jpy_usd, 2, Kfold, 0, 1, 1, log)
RMSE_d_1_001 = cal_cv_rmse(data_1diff, jpy_usd, 1, Kfold, 0, 0, 1, log)
RMSE_d_2_001 = cal_cv_rmse(data_2diff, jpy_usd, 2, Kfold, 0, 0, 1, log)
RMSE_d_1_110 = cal_cv_rmse(data_1diff, jpy_usd, 1, Kfold, 1, 1, 0, log)
RMSE_d_2_110 = cal_cv_rmse(data_2diff, jpy_usd, 2, Kfold, 1, 1, 0, log)
RMSE_d_1_100 = cal_cv_rmse(data_1diff, jpy_usd, 1, Kfold, 1, 0, 0, log)
RMSE_d_2_100 = cal_cv_rmse(data_2diff, jpy_usd, 2, Kfold, 1, 0, 0, log)
# Visualise errors of each model
plt.figure(figsize=(12,8))
plt.plot(RMSE_d_1_011, label='RMSE D1 011')
plt.plot(RMSE_d_2_011, label='RMSE D2 011')
plt.plot(RMSE_d_1_001, label='RMSE D1 001', c='r')
plt.plot(RMSE_d_2_001, label='RMSE D2 001')
plt.plot(RMSE_d_1_110, label='RMSE D1 110')
plt.plot(RMSE_d_2_110, label='RMSE D2 110')
plt.plot(RMSE_d_1_100, label='RMSE D1 100', c='r')
plt.plot(RMSE_d_2_100, label='RMSE D2 100')

plt.axhline(np.mean(RMSE_d_1_011), label='Mean RMSE D1 011 (%.3f)' %np.mean(RMSE_d_1_011), linestyle='-.')
plt.axhline(np.mean(RMSE_d_2_011), label='Mean RMSE D2 011 (%.3f)' %np.mean(RMSE_d_2_011), linestyle='-.')
plt.axhline(np.mean(RMSE_d_1_001), label='Mean RMSE D1 001 (%.3f)' %np.mean(RMSE_d_1_001), c='r', linestyle='-.')
plt.axhline(np.mean(RMSE_d_2_001), label='Mean RMSE D2 001 (%.3f)' %np.mean(RMSE_d_2_001), linestyle='-.')
plt.axhline(np.mean(RMSE_d_1_110), label='Mean RMSE D1 110 (%.3f)' %np.mean(RMSE_d_1_110), linestyle='-.')
plt.axhline(np.mean(RMSE_d_2_110), label='Mean RMSE D2 110 (%.3f)' %np.mean(RMSE_d_2_110), linestyle='-.')
plt.axhline(np.mean(RMSE_d_1_100), label='Mean RMSE D1 100 (%.3f)' %np.mean(RMSE_d_1_100), c='r', linestyle='-.')
plt.axhline(np.mean(RMSE_d_2_100), label='Mean RMSE D2 100 (%.3f)' %np.mean(RMSE_d_2_100), linestyle='-.')

plt.xlabel('CV Fold')
plt.ylabel('RMSE')
plt.title('Error Estimation of Various ARIMA Models')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('error_estimation_1.pdf')
# 1st and 2nd order differencing of log-transformed data
log_data_1diff = log_jpy_usd - log_jpy_usd.shift(1)
log_data_1diff.dropna(inplace=True)
log_data_2diff = log_jpy_usd - 2*log_jpy_usd.shift(1) + log_jpy_usd.shift(2)
log_data_2diff.dropna(inplace=True)

log = True

# Calculate Kfold cross validation RMSE of various ARIMA models
# d_1: 1st order differencing data is used
# 011: (p, d, q) order of ARIMA models
log_RMSE_d_1_011 = cal_cv_rmse(log_data_1diff, log_jpy_usd, 1, Kfold, 0, 1, 1, log)
log_RMSE_d_2_011 = cal_cv_rmse(log_data_2diff, log_jpy_usd, 2, Kfold, 0, 1, 1, log)
log_RMSE_d_1_001 = cal_cv_rmse(log_data_1diff, log_jpy_usd, 1, Kfold, 0, 0, 1, log)
log_RMSE_d_2_001 = cal_cv_rmse(log_data_2diff, log_jpy_usd, 2, Kfold, 0, 0, 1, log)
log_RMSE_d_1_110 = cal_cv_rmse(log_data_1diff, log_jpy_usd, 1, Kfold, 1, 1, 0, log)
log_RMSE_d_2_110 = cal_cv_rmse(log_data_2diff, log_jpy_usd, 2, Kfold, 1, 1, 0, log)
log_RMSE_d_1_100 = cal_cv_rmse(log_data_1diff, log_jpy_usd, 1, Kfold, 1, 0, 0, log)
log_RMSE_d_2_100 = cal_cv_rmse(log_data_2diff, log_jpy_usd, 2, Kfold, 1, 0, 0, log)
# Visualise errors of all models
plt.figure(figsize=(12,8))
plt.plot(log_RMSE_d_1_011, label='RMSE D1 011')
plt.plot(log_RMSE_d_2_011, label='RMSE D2 011')
plt.plot(log_RMSE_d_1_001, label='RMSE D1 001', c='r')
plt.plot(log_RMSE_d_2_001, label='RMSE D2 001')
plt.plot(log_RMSE_d_1_110, label='RMSE D1 110')
plt.plot(log_RMSE_d_2_110, label='RMSE D2 110')
plt.plot(log_RMSE_d_1_100, label='RMSE D1 100', c='r')
plt.plot(log_RMSE_d_2_100, label='RMSE D2 100')

plt.axhline(np.mean(log_RMSE_d_1_011), label='Mean RMSE D1 011 (%.3f)' %np.mean(log_RMSE_d_1_011), linestyle='-.')
plt.axhline(np.mean(log_RMSE_d_2_011), label='Mean RMSE D2 011 (%.3f)' %np.mean(log_RMSE_d_2_011), linestyle='-.')
plt.axhline(np.mean(log_RMSE_d_1_001), label='Mean RMSE D1 001 (%.3f)' %np.mean(log_RMSE_d_1_001), c='r', linestyle='-.')
plt.axhline(np.mean(log_RMSE_d_2_001), label='Mean RMSE D2 001 (%.3f)' %np.mean(log_RMSE_d_2_001), linestyle='-.')
plt.axhline(np.mean(log_RMSE_d_1_110), label='Mean RMSE D1 110 (%.3f)' %np.mean(log_RMSE_d_1_110), linestyle='-.')
plt.axhline(np.mean(log_RMSE_d_2_110), label='Mean RMSE D2 110 (%.3f)' %np.mean(log_RMSE_d_2_110), linestyle='-.')
plt.axhline(np.mean(log_RMSE_d_1_100), label='Mean RMSE D1 100 (%.3f)' %np.mean(log_RMSE_d_1_100), c='r', linestyle='-.')
plt.axhline(np.mean(log_RMSE_d_2_100), label='Mean RMSE D2 100 (%.3f)' %np.mean(log_RMSE_d_2_100), linestyle='-.')

plt.xlabel('CV Fold')
plt.ylabel('RMSE')
plt.title('Error Estimation for Various ARIMA Models - Log-transformed Data')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('error_estimation_2.pdf')
# Data after 1st order differencing
plt.plot(log_jpy_usd)
plt.legend()
plt.tight_layout()
log_data_1diff = log_jpy_usd - log_jpy_usd.shift(1)
log_data_1diff.dropna(inplace=True)
# Visualise data after differencing
plt.plot(log_data_1diff, label='1st Order Difference')
plt.legend()
plt.tight_layout()
plt.title('Data after 1st Order Differencing')
# plt.savefig('data_1_d_diff.pdf')
# SACF and SPACF
alpha=0.05
lags=50

fig, ax = plt.subplots(1,2,figsize=(15,6))
fig = sgt.plot_acf(log_data_1diff, ax=ax[0], lags=lags, alpha=alpha, unbiased=True, 
                   title='Autocorrelation (1st Order)')
fig = sgt.plot_pacf(log_data_1diff, ax=ax[1], lags=lags, alpha=alpha, method='ols', 
                    title='Partial Autocorrelation (1st Order)')
from statsmodels.tsa.arima_model import ARIMA
# Fit the data to ARIMA model
data_to_fit = log_data_1diff

model = ARIMA(data_to_fit, order=(0, 0, 1))  
results_ns = model.fit(disp=-1) 
print(results_ns.summary())
fig = plt.figure(figsize=(12,5))
plt.plot(data_to_fit, label='Original')
plt.plot(results_ns.fittedvalues, color='red', label='fitted')
plt.legend(loc='best')
plt.title('RSS: %.4f'% sum((results_ns.fittedvalues-data_to_fit[:])**2))
plt.tight_layout()
# map the prediction in the original space
pred = log_jpy_usd.copy();
pred.loc[data_to_fit.index] = results_ns.fittedvalues
# the first few samples are used as initial values
allday = pred.index;
for day in data_to_fit.index:
    # get the index of the day
    idx = allday.get_loc(day);
    pred.iloc[idx] = pred.iloc[idx] + jpy_usd.iloc[idx-1];
    #pred.iloc[idx] = pred.iloc[idx] + 2*jpy_usd.iloc[idx-1] - jpy_usd.iloc[idx-2];
    
fig = plt.figure(figsize=(12,6))
plt.plot(jpy_usd[1:], label='Original')
plt.plot(pred[1:], color='red', label='fitted')
plt.legend(loc='best')
import scipy.stats as stats
# Diagnostic plots for residuals from ARIMA model
res = results_ns.resid
fig, ax = plt.subplots(1,2,figsize=(12,6))
ax[0].plot(res)
fig = sm.qqplot(res, stats.distributions.norm, line='r', ax=ax[1]) 
# plt.savefig('res_after_arma.pdf')
fig, ax = plt.subplots(1,2,figsize=(12,6))
fig = sgt.plot_acf(res, ax=ax[0], lags=lags, alpha=alpha, unbiased=True)
fig = sgt.plot_pacf(res, ax=ax[1], lags=lags, alpha=alpha, method='ols')
# plt.savefig('sacf_n_spacf_after_arma.pdf')
# Calculate cross-validation RMSE of the best model
RMSE = cal_cv_rmse(log_data_1diff, log_jpy_usd, 1, Kfold, 0, 0, 1, True)
plt.figure(figsize=(8,6))
plt.plot(RMSE, label='RMSE D1 001')
plt.axhline(np.mean(RMSE), label='Mean RMSE D1 001 (%.3f)' %np.mean(RMSE), c='r', linestyle='-.')
plt.xlabel('CV Fold')
plt.ylabel('RMSE')
plt.title('Error Estimation')
plt.legend(loc='best')
plt.tight_layout()
data = read_data('P2training')
data['Date'] = pd.to_datetime(data['Date'])

df = pd.DataFrame(data[['Date']])
df['JPY_USD'] = data['JPY/USD']
#df['JPY_USD'] = np.log(data['JPY/USD'])
month = []
year = []
day = []
dow = []
for item in df['Date']:
    month.append(item.month)
    year.append(item.year)
    day.append(item.day)
    dow.append(item.dayofweek+1)
df['Year'] = year
df['Month'] = month
df['Day'] = day
df['DoW'] = dow
df['bef_1986'] = (df['Year'] < 1986)
df['jpy_usd_lag_1'] = df['JPY_USD'].shift(1)
df['jpy_usd_lag_5'] = df['JPY_USD'].shift(5)
df['jpy_usd_lag_10'] = df['JPY_USD'].shift(10)
df['diff_lag_1_lag_2'] = df['JPY_USD'].shift(1) - df['JPY_USD'].shift(2)

df.head()
def cal_cv_rmse_reg(X, formula, fold):
    """Calculate root mean square forecasting error
       based on timeseries split cross validation, regression model
       on data is used.
       
       params:
       X: dataframe, data for fitting
       formula: string, formula for ols regression
       fold: int, cross validation fold
       
       return:
       RMSE: list, list of RMSE for all folds
    """ 
    
    RMSE = []
    tscv = TimeSeriesSplit(n_splits=fold)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.loc[train_index], X.loc[test_index]

        lmfit = smf.ols(formula, data = X_train).fit()
        
        forecasts = lmfit.predict(X_test.drop('JPY_USD', 1))
        errors = forecasts - X_test['JPY_USD']
        RMSE.append(np.sqrt(np.mean(np.power(errors, 2))))
        
    return RMSE 
X = df
formula1 = 'JPY_USD~C(Month)+Year+C(Day)+C(bef_1986)'
formula_lag_1 = 'JPY_USD~C(Month)+Year+C(Day)+jpy_usd_lag_1+C(bef_1986)'
formula_lag_5 = 'JPY_USD~C(Month)+Year+C(Day)+jpy_usd_lag_5+C(bef_1986)'
formula_lag_10 = 'JPY_USD~C(Month)+Year+C(Day)+jpy_usd_lag_10+C(bef_1986)'
formula_w_lag_diff = 'JPY_USD~C(Month)+Year+C(Day)+C(bef_1986)+diff_lag_1_lag_2+jpy_usd_lag_1'
formula_lag_n_cat = 'JPY_USD~C(bef_1986)+diff_lag_1_lag_2+jpy_usd_lag_1'

RMSE_wo_lag = cal_cv_rmse_reg(X, formula1, Kfold)
RMSE_w_lag_1 = cal_cv_rmse_reg(X, formula_lag_1, Kfold)
RMSE_w_lag_5 = cal_cv_rmse_reg(X, formula_lag_5, Kfold)
RMSE_w_lag_10 = cal_cv_rmse_reg(X, formula_lag_10, Kfold)
RMSE_w_lag_diff = cal_cv_rmse_reg(X, formula_w_lag_diff, Kfold)
RMSE_lag_n_cat = cal_cv_rmse_reg(X, formula_lag_n_cat, Kfold)
plt.figure(figsize=(12,8))
# plt.plot(RMSE_wo_lag, label='RMSE w/o Lag')
plt.plot(RMSE_w_lag_1, label='RMSE Lag 1')
plt.plot(RMSE_w_lag_5, label='RMSE Lag 5')
# plt.plot(RMSE_w_lag_10, label='RMSE Lag 10')
plt.plot(RMSE_w_lag_diff, label='RMSE w Lag Diff')
plt.plot(RMSE_lag_n_cat, label='RMSE Lag & Cat', c='r')


# plt.axhline(np.mean(RMSE_wo_lag), label='Mean RMSE w/o Lag (%.5f)' %np.mean(RMSE_wo_lag), linestyle='-.')
plt.axhline(np.mean(RMSE_w_lag_1), label='Mean RMSE Lag 1 (%.5f)' %np.mean(RMSE_w_lag_1), linestyle='-.')
plt.axhline(np.mean(RMSE_w_lag_5), label='Mean RMSE Lag 5 (%.5f)' %np.mean(RMSE_w_lag_5), linestyle='-.')
# plt.axhline(np.mean(RMSE_w_lag_10), label='Mean RMSE Lag 10 (%.5f)' %np.mean(RMSE_w_lag_10), linestyle='-.')
plt.axhline(np.mean(RMSE_w_lag_diff), label='Mean RMSE w Lag Diff (%.5f)' %np.mean(RMSE_w_lag_diff), linestyle='-.')
plt.axhline(np.mean(RMSE_lag_n_cat), label='Mean RMSE Lag & Cat (%.5f)' %np.mean(RMSE_lag_n_cat), linestyle='-.', c='r')


plt.legend(loc='best')
plt.title('Error Estimation of Various Regression Models')
plt.xlabel('CV Fold')
plt.ylabel('RMSE')
plt.tight_layout()
plt.savefig('error_estimation_3.pdf')
# Regression using the best model found in the previous section
lmfit = smf.ols(formula_lag_n_cat, data = df).fit()
lmfit.summary()
fig, ax = plt.subplots(1, 2, figsize=(12,6))
fig = ax[0].plot(lmfit.resid)
fig = sm.qqplot(lmfit.resid, line='s', ax=ax[1])
plt.tight_layout()
plt.savefig('res_after_reg.pdf')
fig, ax = plt.subplots(1, 2, figsize=(12,6))
fig = sgt.plot_acf(lmfit.resid, lags=50, ax=ax[0])
fig = sgt.plot_pacf(lmfit.resid, lags=50, ax=ax[1])
plt.tight_layout()
plt.savefig('sacf_n_spacf_after_reg.pdf')
newcol = []
for item in data.columns:
    newcol.append(item.replace('/', '_'))
data.columns = newcol
collist = list(data.columns);
collist.remove('Date')
collist.remove('JPY_USD')
df = data[collist].shift(1) - data[collist].shift(2)
df.head()
df = df.merge(data, how='left')
df['JPY_USD'] = data['JPY_USD']
df['Date'] = data['Date']
month = []
year = []
day = []
dow = []
for item in data['Date']:
    month.append(item.month)
    year.append(item.year)
    day.append(item.day)
    dow.append(item.dayofweek+1)
#df = data.copy()
df['Year'] = year
df['Month'] = month
df['Day'] = day
df['DoW'] = dow
df['bef_1986'] = (df['Year'] < 1986)
df['jpy_usd_lag_1'] = df['JPY_USD'].shift(1)
df['diff_lag_1_lag_2'] = df['JPY_USD'].shift(1) - df['JPY_USD'].shift(2)

df.head()
collist = list(df.columns);
collist.remove('Date')
collist.remove('JPY_USD')
form = 'JPY_USD' + '~' + '+'.join(collist);
form
form = 'JPY_USD~AUD_USD+GBP_USD+CAD_USD+NLG_USD+FRF_USD+DEM_USD+CHF_USD+C(bef_1986)+jpy_usd_lag_1'
RMSE = cal_cv_rmse_reg(df, form, Kfold)
#plt.figure(figsize=(6,5))
plt.plot(RMSE, label='RMSE')
plt.axhline(np.mean(RMSE), label='Mean RMSE(%.5f)' %np.mean(RMSE), linestyle='-.', c='r')
plt.legend(loc='best')
plt.tight_layout()