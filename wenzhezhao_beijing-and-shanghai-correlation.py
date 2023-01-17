import os

import numpy as np 

import pandas as pd 

import datetime as dt

import statsmodels.api as sm

from statsmodels.tsa.api import VAR

from statsmodels.tsa.stattools import adfuller

from statsmodels.tools.eval_measures import rmse, aic

from statsmodels.tsa.stattools import grangercausalitytests

from statsmodels.tsa.vector_ar.vecm import coint_johansen

from statsmodels.stats.stattools import durbin_watson

from statsmodels.tsa.stattools import acf



import matplotlib.pyplot as plt

%matplotlib inline

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot



import cufflinks as cf

cf.go_offline()
os.listdir('../input/anjuke')
df = pd.DataFrame(index=pd.date_range(start = dt.datetime(2011,1,1), end = dt.datetime(2021,1,1), freq='M'))

monthRange = df.index.to_series().apply(lambda x: dt.datetime.strftime(x, '%b %Y'))
bj_data = pd.read_csv('../input/anjuke/beijing')
bj_data
bj_series = pd.concat([bj_data['2011'],bj_data['2012'],bj_data['2013'],bj_data['2014'],bj_data['2015'],

                       bj_data['2016'],bj_data['2017'],bj_data['2018'],bj_data['2019'],bj_data['2020']])
bj_series.index =  monthRange
bj_series
sh_data = pd.read_csv('../input/anjuke/shanghai')
sh_data
sh_series = pd.concat([sh_data['2011'],sh_data['2012'],sh_data['2013'],sh_data['2014'],sh_data['2015'],

                       sh_data['2016'],sh_data['2017'],sh_data['2018'],sh_data['2019'],sh_data['2020']])
sh_series.index =  monthRange
sh_series
bj_series.dropna(inplace = True)

sh_series.dropna(inplace = True)
fig = go.Figure()

fig.add_trace(go.Scatter(x=bj_series.index, y=bj_series.values, name = 'Beijing'))

fig.add_trace(go.Scatter(x=sh_series.index, y=sh_series.values, name = 'Shanghai'))





fig.layout.title='Beijing and Shanghai house price monthly variation from Year 2011 to 2020'

fig.layout.yaxis.title="Price"

fig.layout.xaxis.title="Time"

fig.show()
np.corrcoef(bj_series.values,sh_series.values)

### 相关系数有：0.96320018
two_df = pd.DataFrame({'Beijing':bj_series.values,'Shanghai':sh_series.values})

two_df.index = monthRange[:-5]
two_df
fig=go.Figure()



fig.add_scatter(x=two_df['Shanghai'],y=two_df['Beijing'],mode='markers',marker=dict(size=5))



fig.layout.title='Relation between Shanghai house price and Beijing house price'

fig.layout.xaxis.title="Shanghai house price"

fig.layout.yaxis.title="Beijing house price"



iplot(fig)
lm = sm.OLS(two_df['Beijing'],sm.add_constant(two_df['Shanghai']))

res = lm.fit()

res.summary()### strong linear relationship
# train_df = two_df.iloc[:-10,:]

# test_df = two_df.iloc[-10:,:]

# model = sm.tsa.statespace.SARIMAX(train_df["Beijing"], exog=train_df["Shanghai"]).fit(disp=-1)

# model.summary()
sm.graphics.tsa.plot_acf(bj_series.values)## , lags=100

plt.show()
sm.graphics.tsa.plot_acf(sh_series.values)## , lags=100

plt.show()
plt.plot(sm.tsa.stattools.ccf(bj_series.values,sh_series.values, unbiased=False)[:22])
plt.plot(sm.tsa.stattools.ccf(sh_series.values,bj_series.values, unbiased=False)[:22])
maxlag=12

test = 'ssr_chi2test'

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    

    """Check Granger Causality of all possible combinations of the Time series.

    The rows are the response variable, columns are predictors. The values in the table 

    are the P-Values. P-Values lesser than the significance level (0.05), implies 

    the Null Hypothesis that the coefficients of the corresponding past values is 

    zero, that is, the X does not cause Y can be rejected.



    data      : pandas dataframe containing the time series variables

    variables : list containing names of the time series variables.

    """

    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)

    for c in df.columns:

        for r in df.index:

            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)

            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]

            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')

            min_p_value = np.min(p_values)

            df.loc[r, c] = min_p_value

    df.columns = [var + '_x' for var in variables]

    df.index = [var + '_y' for var in variables]

    return df



grangers_causation_matrix(two_df, variables = two_df.columns) 
def cointegration_test(df, alpha=0.05): 

    """Perform Johanson's Cointegration Test and Report Summary"""

    out = coint_johansen(df,-1,5)

    d = {'0.90':0, '0.95':1, '0.99':2}

    traces = out.lr1

    cvts = out.cvt[:, d[str(1-alpha)]]

    def adjust(val, length= 6): return str(val).ljust(length)



    # Summary

    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)

    for col, trace, cvt in zip(df.columns, traces, cvts):

        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)



cointegration_test(two_df)
nobs = 4

df_train, df_test = two_df[0:-nobs], two_df[-nobs:]



# Check size

print(df_train.shape)  # (111, 2)

print(df_test.shape)  # (4, 2)
def adfuller_test(series, signif=0.05, name='', verbose=False):

    """Perform ADFuller to test for Stationarity of given series and print report"""

    r = adfuller(series, autolag='AIC')

    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}

    p_value = output['pvalue'] 

    def adjust(val, length= 6): return str(val).ljust(length)



    # Print Summary

    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)

    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')

    print(f' Significance Level    = {signif}')

    print(f' Test Statistic        = {output["test_statistic"]}')

    print(f' No. Lags Chosen       = {output["n_lags"]}')



    for key,val in r[4].items():

        print(f' Critical value {adjust(key)} = {round(val, 3)}')



    if p_value <= signif:

        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")

        print(f" => Series is Stationary.")

    else:

        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")

        print(f" => Series is Non-Stationary.")    
# ADF Test on each column

for name, column in df_train.iteritems():

    adfuller_test(column, name=column.name)

    print('\n')
# 1st difference

df_differenced = df_train.diff().dropna()
# ADF Test on each column of 1st Differences Dataframe

for name, column in df_differenced.iteritems():

    adfuller_test(column, name=column.name)

    print('\n')
model = VAR(df_differenced)

for i in [1,2,3,4,5,6,7,8,9]:

    result = model.fit(i)

    print('Lag Order =', i)

    print('AIC : ', result.aic)

    print('BIC : ', result.bic)

    print('FPE : ', result.fpe)

    print('HQIC: ', result.hqic, '\n')
x = model.select_order(maxlags=12)

x.summary()

## Since the explicitly computed AIC is the lowest at lag 1, I choose the selected order as 1.
model_fitted = model.fit(1)

model_fitted.summary()
out = durbin_watson(model_fitted.resid)



for col, val in zip(two_df.columns, out):

    print(col, ':', round(val, 2))
# Get the lag order

lag_order = model_fitted.k_ar

print(lag_order)  #> 4



# Input data for forecasting

forecast_input = df_differenced.values[-lag_order:]

forecast_input
# Forecast

fc = model_fitted.forecast(y=forecast_input, steps=nobs)

df_forecast = pd.DataFrame(fc, index=two_df.index[-nobs:], columns=two_df.columns + '_2d')

df_forecast
def invert_transformation(df_train, df_forecast, second_diff=False):

    """Revert back the differencing to get the forecast to original scale."""

    df_fc = df_forecast.copy()

    columns = df_train.columns

    for col in columns:        

        # Roll back 2nd Diff

        if second_diff:

            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()

        # Roll back 1st Diff

        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()

    return df_fc
df_results = invert_transformation(df_train, df_forecast, second_diff=True)        

df_results.loc[:, ['Beijing_forecast', 'Shanghai_forecast']]
fig, axes = plt.subplots(nrows=int(len(two_df.columns)/2), ncols=2, dpi=150, figsize=(15,6))

for i, (col,ax) in enumerate(zip(two_df.columns, axes.flatten())):

    df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)

    df_test[col][-nobs:].plot(legend=True, ax=ax);

    ax.set_title(col + ": Forecast vs Actuals")

    ax.xaxis.set_ticks_position('none')

    ax.yaxis.set_ticks_position('none')

    ax.spines["top"].set_alpha(0)

    ax.tick_params(labelsize=6)



plt.tight_layout();
def forecast_accuracy(forecast, actual):

    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE

    me = np.mean(forecast - actual)             # ME

    mae = np.mean(np.abs(forecast - actual))    # MAE

    mpe = np.mean((forecast - actual)/actual)   # MPE

    rmse = np.mean((forecast - actual)**2)**.5  # RMSE

    corr = np.corrcoef(forecast, actual)[0,1]   # corr

    mins = np.amin(np.hstack([forecast[:,None], 

                              actual[:,None]]), axis=1)

    maxs = np.amax(np.hstack([forecast[:,None], 

                              actual[:,None]]), axis=1)

    minmax = 1 - np.mean(mins/maxs)             # minmax

    return({'mape':mape, 'me':me, 'mae': mae, 

            'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})



print('Forecast Accuracy of: Beijing')

accuracy_prod = forecast_accuracy(df_results['Beijing_forecast'].values, df_test['Beijing'])

for k, v in accuracy_prod.items():

    print(k, ': ', round(v,4))



print('\nForecast Accuracy of: Shanghai')

accuracy_prod = forecast_accuracy(df_results['Shanghai_forecast'].values, df_test['Shanghai'])

for k, v in accuracy_prod.items():

    print(k, ': ', round(v,4))