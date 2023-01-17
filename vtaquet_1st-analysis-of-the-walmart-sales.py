# import the usual modules

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualisation

import matplotlib.pyplot as plt # plot

import os

import warnings

#import funcs

from datetime import datetime

from matplotlib.pyplot import cm
# Some commands

warnings.filterwarnings('ignore')

# color map
import os

import zipfile

#

dir_zip = '../input/walmart-recruiting-store-sales-forecasting/'

zip_features = dir_zip+'features.csv.zip'

zip_train = dir_zip+'train.csv.zip'

zip_test = dir_zip+'test.csv.zip'

zip_sample = dir_zip+'sampleSubmission.csv.zip'

#

zip_ref = zipfile.ZipFile(zip_features, 'r')

zip_ref.extractall()

zip_ref.close()

#

zip_ref = zipfile.ZipFile(zip_train, 'r')

zip_ref.extractall()

zip_ref.close()

#

zip_ref = zipfile.ZipFile(zip_test, 'r')

zip_ref.extractall()

zip_ref.close()

#

zip_ref = zipfile.ZipFile(zip_sample, 'r')

zip_ref.extractall()

zip_ref.close()

#

# read csv files and create dataframes

df_train = pd.read_csv('train.csv', parse_dates=['Date']).set_index(keys=['Store','Dept','Date'], drop=False)

df_test = pd.read_csv('test.csv', parse_dates=['Date']).set_index(keys=['Store','Dept','Date'], drop=False)

df_stores = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/stores.csv').set_index(keys=['Store'], drop=False)

df_features = pd.read_csv('features.csv', parse_dates=['Date']).set_index(keys=['Store','Date'], drop=False)

df_train.index.names = ['Store_ind', 'Dept_ind', 'Date_ind']

df_test.index.names = ['Store_ind', 'Dept_ind', 'Date_ind']
# create new features 

df_train['Weekofyear'] = df_train['Date'].dt.weekofyear

df_test['Weekofyear'] = df_test['Date'].dt.weekofyear

df_train['Month'] = df_train['Date'].dt.month

df_test['Month'] = df_test['Date'].dt.month

df_train['Year'] = df_train['Date'].dt.year

df_test['Year'] = df_test['Date'].dt.year

df_features['Temperature'] = (df_features['Temperature']-32.)/1.8 # to Celsius degree

df_features['Fuel_Price'] = df_features['Fuel_Price']/3.785411784 # to liters

df_train['pctchange_Weekly_Sales'] = df_train['Weekly_Sales'].pct_change().fillna(0.)
# join dataframes

df_train_all = df_train.join(df_features[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 

                                          'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']],

                             on=['Store', 'Date'])

df_train_all = df_train_all.join(df_stores[['Type','Size']], on=['Store'])

df_test_all = df_test.join(df_features[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 

                                        'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']],

                           on=['Store', 'Date'])

df_test_all = df_test_all.join(df_stores[['Type','Size']], on=['Store'])
# perform some data cleaning

list_dummies = ['IsHoliday', 'Type']

for dummy in list_dummies:

    df_train_all[dummy] = pd.get_dummies(df_train_all[dummy],drop_first=True)

    df_test_all[dummy] = pd.get_dummies(df_train_all[dummy],drop_first=True)

list_markdown = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']

df_train_all[list_markdown] = df_train_all[list_markdown].fillna(0.)

df_test_all[list_markdown] = df_test_all[list_markdown].fillna(0.) 
list_inpfeat = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday', 'Type', 'Size']

list_inpdate = ['Weekofyear', 'Month', 'Year']

list_store_dept = [(1,1),(2,2),(3,3),(4,4),(5,5)]
df_train_all.head(10) 
df_train_all2 = df_train_all.loc[(slice(None), slice(None), slice('2010-02-05','2012-02-03')),:] #

df_valid_all = df_train_all.loc[(slice(None), slice(None), slice('2012-02-10','2015-02-05')),:]

#print(len(df_valid_all.loc[(1,1)]))

#df_valid_all.head()

#df_train_all.head()
# first statistics

Nstore = df_train.index.max()[0] ; Ndept = df_train.index.max()[1]

df_stats = pd.concat([df_train.groupby(['Store','Dept']).median().Weekly_Sales,

                      df_train.groupby(['Store','Dept']).mean().Weekly_Sales,

                      df_train.groupby(['Store','Dept']).std().Weekly_Sales,

                      df_train.groupby(['Store','Dept']).max().Weekly_Sales, 

                      df_train.groupby(['Store','Dept']).min().Weekly_Sales],

                     axis=1)

df_stats.columns = ['Median', 'Mean', 'Std', 'Max', 'Min']

df_stats['Norm_std'] = df_stats['Std']/(df_stats['Mean']) #np.abs

df_stats['Norm_maxmin'] = (df_stats['Max']-df_stats['Min'])/df_stats['Median']

df_stats = df_stats.sort_values(by=['Norm_std'], axis=0, ascending=False)

df_stats.to_csv('summary_stores_stats.csv')

df_stats.head()
# Distribution of averaged weekly sales and normalised standard deviations

from scipy.stats import norm, skew, probplot #for some statistics

plt.figure(0,figsize=[15,5])

plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.subplot(1,2,1)

sns.distplot(df_stats.Mean)#, fit=norm

plt.subplot(1,2,2)

plt.xlim([0,5])

sns.distplot(df_stats.Norm_std.replace([np.inf, -np.inf], np.nan).dropna(),bins=1000)#, fit=norm

plt.show()

plt.close(0)
# plot target feature with time

def plot_target(df, storedept, target, plotname, dpi=100):

    plt.figure(0,figsize=(9,3), dpi=dpi)

    for storedept_tupple in storedept:

        namemag = str(storedept_tupple[0])+' '+str(storedept_tupple[1])

        try:

            df2 = df.loc[storedept_tupple,target]#.drop('Date',axis=1)

            plt.plot(df2.index.to_pydatetime(),df2,label=namemag)# #, color='tab:red')

        except:

            pass

    plt.gca().set(xlabel='Date', ylabel=target, title='Store '+str(storedept_tupple[0])+' '+str(storedept_tupple[1]))

    plt.legend()

    #plt.savefig('figures/'+plotname+'.pdf',bbox_inches='tight',transparent=True)

    plt.show()

    plt.close(0)
# plot weekly sale for all departments in individual stores with time

plot_target(df_train_all,list_store_dept,'Weekly_Sales','sale_store')
# plot target feature with time

def plot_target_diff(df, storedept, list_diff, target, plotname, dpi=100):

    plt.figure(0,figsize=(9,3), dpi=dpi)

    for storedept_tupple in storedept:

        namemag = str(storedept_tupple[0])+' '+str(storedept_tupple[1])

        try:

            df2 = df.loc[storedept_tupple,target]#.drop('Date',axis=1)

            for diff in list_diff:

                df2 = df2.diff(diff)

            plt.plot(df2.index.to_pydatetime(),df2,label=namemag)# #, color='tab:red')

        except:

            pass

    plt.gca().set(xlabel='Date', ylabel=target, title='Store '+str(storedept_tupple[0])+' '+str(storedept_tupple[1]))

    plt.legend()

    #plt.savefig('figures/'+plotname+'.pdf',bbox_inches='tight',transparent=True)

    plt.show()

    plt.close(0)
# plot weekly sale for all departments in individual stores with time

plot_target_diff(df_train_all,list_store_dept,[1,52],'Weekly_Sales','sale_store')
# Distribution of target feature

def plot_targetdistrib(y,plottitle,plotname):

	from scipy.stats import norm, skew, probplot #for some statistics

	plt.figure(0,figsize=[15,5])

	plt.subplots_adjust(wspace=0.2, hspace=0.5)

	plt.subplot(1,2,1)

	(mu, sigma) = norm.fit(y)

	#print( 'mu = {:.2f} and sigma = {:.2f}'.format(mu, sigma))

	sns.distplot(y, fit=norm)

	plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

	            loc='best')

	plt.ylabel('Frequency')

	plt.title(plottitle)

	# QQ-plot wrt normal distribution

	plt.subplot(1,2,2)

	res = probplot(y, plot=plt)

	#plt.savefig('figures/'+plotname+'.pdf',bbox_inches='tight',transparent=True)

	plt.show()

	plt.close(0)

# plot weekly sale distribution for a given department

df2 = df_train_all.loc[(1,1)]

plot_targetdistrib(df2.Weekly_Sales,'Weekly_Sales','saledistrib1')
# plot individual feature with time

def plot_features(df, list_feature, storedept, plotname, dpi=100):

    Nstore = df.index.max()[0]

    plt.figure(0,figsize=(16,5), dpi=dpi)

    plt.subplots_adjust(wspace=0.25, hspace=0.5)

    #

    for ifeat, feat in enumerate(list_feature):

        plt.subplot(1,len(list_feature),ifeat+1)

        plt.gca().set(xlabel='Date', ylabel=feat)

        plt.xticks(rotation=45)

        for storedept_tupple in storedept:

            try:

                df2 = df.loc[storedept_tupple].drop('Date',axis=1)

                plt.plot(df2.index.to_pydatetime(), df2[feat])# #, color='tab:red')

            except:

                pass

    #plt.savefig('figures/'+plotname+'.pdf',bbox_inches='tight',transparent=True)

    plt.show()

    plt.close(0)
# plot evolution of input features with time

plot_features(df_train_all,['Temperature', 'Fuel_Price', 'CPI', 'Unemployment'],

              list_store_dept, 'temp_time')
# compare features vs weekly sale

def comp_features(df, list_x, y, storedept, plotname, dpi=100):

    Nstore = df.index.max()[0]

    plt.figure(0,figsize=(16,5), dpi=dpi)

    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    #

    for ix, x in enumerate(list_x):

        plt.subplot(1,len(list_x),ix+1)

        plt.gca().set(xlabel=x, ylabel=y)

        plt.xticks(rotation=45)

        for storedept_tupple in storedept:

            try:

                df2 = df.loc[storedept_tupple].drop('Date',axis=1)

                plt.scatter(df2[x], df2[y])# #, color='tab:red')

            except:

                pass

    #plt.savefig('figures/'+plotname+'.pdf',bbox_inches='tight',transparent=True)

    plt.show()

    plt.close(0)
comp_features(df_train_all,['Temperature', 'Fuel_Price', 'CPI', 'Unemployment'], 'Weekly_Sales',

                   list_store_dept, 'features_vs_weeklysale')
# compare features vs weekly sale

def corr_depts(df, store, list_dept, target, plotname, dpi=100):

    Nstore = df.index.max()[0]

    plt.figure(0,figsize=(16,5), dpi=dpi)

    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    #

    iplot = 1

    for x in (list_dept[1:]):

        plt.subplot(1,len(list_dept)-1,iplot)

        plt.gca().set(xlabel=x, ylabel=list_dept[0])

        plt.xticks(rotation=45)

        try:

            dfx = df.loc[(store,x)].drop('Date',axis=1)[target]

            dfy = df.loc[(store,list_dept[0])].drop('Date',axis=1)[target]

            plt.scatter(dfx, dfy, label=('%.2f' % (dfx.corr(dfy))))# #, color='tab:red')

        except:

            pass

        iplot += 1

        plt.legend()

    #plt.savefig('figures/'+plotname+'.pdf',bbox_inches='tight',transparent=True)

    plt.show()

    plt.close(0)
corr_depts(df_train_all,1, [1,2,3,4,5],'Weekly_Sales', 'features_vs_weeklysale')
corr_depts(df_train_all,1, [1,2,3,4,5],'pctchange_Weekly_Sales', 'features_vs_weeklysale')
# ADF Test

from statsmodels.tsa.stattools import adfuller, kpss

def adf_test(df):

    if len(df.Weekly_Sales) > 10:

        result = adfuller(df.Weekly_Sales, autolag='AIC')

    else:

        result = [np.nan, np.nan, np.nan, np.nan, np.nan]

    return result[1] #result[0], result[1], result[4]

#df_stats['ADFstats'] = df_train_all.groupby(['Store','Dept']).apply(adf_test)

df_stats['ADF_pvalue'] = df_train_all.groupby(['Store','Dept']).apply(adf_test)

df_stats.sort_values(['Store','Dept'], axis=0, ascending=True).head()
# Seasonal Decomposition

from statsmodels.tsa.seasonal import seasonal_decompose

def plot_season(df, list_store_dept):

    plt.rcParams.update({'figure.figsize': (8,5)})

    #

    # Multiplicative Decomposition 

    # Additive Decomposition

    plt.figure(figsize=[15,10])

    plt.subplots_adjust(wspace=0.33, hspace=0.5)

    for ist, storedept in enumerate(list_store_dept):

        result_add = seasonal_decompose(df.loc[storedept,'Weekly_Sales'], 

                                        model='additive', extrapolate_trend='freq', freq=52)

        # Plot

        plt.subplot(len(list_store_dept),4,ist*4+1) ; plt.xticks(rotation=30)

        plt.plot(df.loc[storedept,'Weekly_Sales'],label=storedept) ; plt.ylabel('Weekly Sales') ; plt.legend()

        plt.subplot(len(list_store_dept),4,ist*4+2) ; plt.xticks(rotation=30)

        plt.plot(result_add.trend,label=storedept) ; plt.ylabel('Trend') ; plt.legend()

        plt.subplot(len(list_store_dept),4,ist*4+3) ; plt.xticks(rotation=30)

        plt.plot(result_add.seasonal) ; plt.ylabel('Seasonal')

        plt.subplot(len(list_store_dept),4,ist*4+4) ; plt.xticks(rotation=30)

        plt.plot(result_add.resid) ; plt.ylabel('Residual')#.plot()#.suptitle('Additive Decompose', fontsize=16)
plot_season(df_train_all, list_store_dept)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



def plot_diff(df,list_diff,nlags):

    plt.rcParams.update({'figure.figsize':(9,5), 'figure.dpi':120})

    #

    df2 = df.Weekly_Sales.reset_index()#

    #

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(15, 5))

    for diff in list_diff:

        if diff > 0:

            df2 = df2.diff(diff)

    ax1.plot(df2.Weekly_Sales) ; ax1.set_title('Order Differencing '+str(list_diff))

    plot_acf(df2.Weekly_Sales.dropna(), ax=ax2,lags=nlags); ax2.set_xlim(0,nlags) ; ax2.set_ylim(-1,1) 

    plot_pacf(df2.Weekly_Sales.dropna(), ax=ax3,lags=nlags) ; ax3.set_xlim(0,nlags) ; ax3.set_ylim(-1,1) 
plot_diff(df_train_all.loc[(2,2)],[0],53)
plot_diff(df_train_all.loc[(2,2)],[52,1],20)
# create a function that plots the rolling mean and std, and performs the ADF test

from statsmodels.tsa.stattools import adfuller

def test_stationarity(ts,window):

    #

    #Determing rolling statistics

    rolmean = ts.rolling(window=window).mean() #pd.rolling_mean(ts, window=12)

    rolstd = ts.rolling(window=window).std() #pd.rolling_std(ts, window=12)



    #Plot rolling statistics:

    plt.figure(figsize=(8,3))

    orig = plt.plot(ts, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(ts, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)
# Accuracy metrics

from statsmodels.tsa.stattools import acf

def forecast_accuracy(forecast, actual):

    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # Mean Absolute Percentage Error

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

    #acf1 = acf(fc-test)[1]                      # ACF1

    return({'mape':mape, 'me':me, 'mae': mae, 

            'mpe': mpe, 'rmse':rmse, #'acf1':acf1, 

            'corr':corr, 'minmax':minmax})
# Averaging the yearly evolution

mean_sale = df_train_all2.groupby(['Store','Dept','Weekofyear'])['Weekly_Sales'].mean()

df_valid_pred = df_valid_all[['Store', 'Dept', 'Weekofyear']].join(mean_sale, on=['Store', 'Dept', 'Weekofyear'])
list_index = [] ; list_mape = []

for storedept in list_store_dept:

    dftrain = df_train_all2.loc[storedept] ; dfpred = df_valid_pred.loc[storedept] ; dfactual = df_valid_all.loc[storedept]

    list_index.append(storedept) ; list_mape.append(forecast_accuracy(dfpred.Weekly_Sales, dfactual.Weekly_Sales)['mape'])

df_mape = pd.DataFrame({'MAPE_simplemodel' : list_mape}, index = list_index)

df_mape.head()
# plot the prediction 

def plot_predtarget(list_storedept, df_train, df_test, df_actual, target, plotname, dpi=100):

    color=cm.tab10(np.linspace(0,1,len(list_storedept)*2))

    plt.figure(0,figsize=(16,5), dpi=dpi)

    for ist, storedept in enumerate(list_storedept):

        dftrain = df_train.loc[storedept] ; dfpred = df_test.loc[storedept] ; dfactual = df_actual.loc[storedept]

        plt.plot(dftrain.index.to_pydatetime(),dftrain[target],color=color[ist],label=storedept)# #, color='tab:red')

        plt.plot(dfpred.index.to_pydatetime(),dfpred[target],color=color[ist],ls='--')# #, color='tab:red')

        plt.plot(dfactual.index.to_pydatetime(),dfactual[target],color=color[ist],ls='-')# #, color='tab:red')

    plt.gca().set(xlabel='Date', ylabel=target, title='Store ')

    plt.legend()

    #plt.savefig('figures/'+plotname+'.pdf',bbox_inches='tight',transparent=True)

    plt.show() ; plt.close(0)
plot_predtarget(list_store_dept,df_train_all2,df_valid_pred,df_valid_all,'Weekly_Sales','predsale_store1')
train = df_train_all2.loc[list_store_dept[1],'Weekly_Sales']

valid = df_valid_all.loc[list_store_dept[1],'Weekly_Sales']
#Perform a grid search by: varying (p,d,q) and (P,D,Q), using the AIC as information criterion

from statsmodels.tsa.arima_model import ARIMA

import pmdarima as pm



smodel = pm.auto_arima(train, 

                       start_p=1, start_q=1, start_d=1, # start p, q, d 

                       max_p=2, max_q=2,     # maximum p and q

                       max_d=2, d=None,        # let model determine 'd'

                       seasonal=True,        # Seasonality

                       m=52,                 # frequency of series

                       start_P=1, start_Q=1, # 

                       max_P=1, max_Q=1,     # maximum P and Q

                       D=1,                  # one "yearly" differentiation

                       test='adf',           # use adftest to find optimal 'd'

                       information_criterion='aic', # used to select best model

                       trace=True,          # print results whilst training

                       error_action='ignore',   # ignore orders that don't work

                       suppress_warnings=True, 

                       stepwise=True,       # apply intelligent order search

                      )





print(smodel.summary())
smodel.plot_diagnostics()

plt.show()
# Forecast

n_periods = len(valid.index) 

fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)

in_sample_preds, in_sample_confint = smodel.predict_in_sample(return_conf_int=True)

index_of_fc = valid.index #pd.date_range(valid.index[0], periods = Nteststeps, freq='7D')

#

# make forecast series

train_fc = pd.Series(in_sample_preds, index=train.index)

valid_fc = pd.Series(fitted, index=index_of_fc)

valid_fc_lower = pd.Series(confint[:, 0], index=index_of_fc)

valid_fc_upper = pd.Series(confint[:, 1], index=index_of_fc)
# Compare predicted and actual test TS

fig, axes = plt.subplots(1, 1, figsize=(10,5), dpi=100, sharex=True)

plt.plot(train, label='Training set')

plt.plot(train_fc, label='Fit training set')

plt.plot(valid_fc, color='darkgreen', label='Forecast')

plt.fill_between(valid_fc_lower.index, 

                 valid_fc_lower, 

                 valid_fc_upper, 

                 color='k', alpha=.15)

plt.plot(valid, label='Validation set')

plt.legend()

plt.title("SARIMA - Forecast on validation set")

plt.show()
forecast_accuracy(valid_fc, valid)
#Perform a grid search by: varying (p,d,q) and (P,D,Q), using the AIC as information criterion

from statsmodels.tsa.arima_model import ARIMA

import pmdarima as pm



sarima_index = [] ; sarima_mape = []

sarima_train_fc = [] ; sarima_valid_fc = []

sarima_valid_fc_lower = [] ; sarima_valid_fc_upper = []

#

for storedept in list_store_dept: #[:2]:

    train = df_train_all2.loc[storedept,'Weekly_Sales'] ; valid = df_valid_all.loc[storedept,'Weekly_Sales']

    #

    # run the grid search for given timeseries

    smodel = pm.auto_arima(train, 

                           start_p=1, start_q=1, 

                           max_p=2, max_q=2,     # maximum p and q

                           d=None,               # let model determine 'd'

                           seasonal=True,        # Seasonality

                           m=52,                 # frequency of series

                           start_P=1, start_Q=1, # 

                           max_P=1, max_Q=1,     # maximum P and Q

                           D=1,                  # one "yearly" differentiation

                           test='adf',           # use adftest to find optimal 'd'

                           information_criterion='aic', # used to select best model

                           trace=True,          # print results whilst training

                           error_action='ignore',   # ignore orders that don't work

                           suppress_warnings=True, 

                           stepwise=True,       # apply intelligent order search

                          )

    #

    # get the forecast TS

    n_periods = len(valid.index)

    fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)

    in_sample_preds, in_sample_confint = smodel.predict_in_sample(return_conf_int=True)

    index_of_fc = valid.index #pd.date_range(holdout.index[0], periods = Nteststeps, freq='7D')

    #

    # make forecast series

    train_fc = pd.Series(in_sample_preds, index=train.index)

    valid_fc = pd.Series(fitted, index=index_of_fc)

    valid_fc_lower = pd.Series(confint[:, 0], index=index_of_fc)

    valid_fc_upper = pd.Series(confint[:, 1], index=index_of_fc)

    #

    #print(smodel.summary())

    # save the forecast 

    sarima_train_fc.append(train_fc) ; sarima_valid_fc.append(valid_fc)

    sarima_valid_fc_lower.append(valid_fc_lower) ; sarima_valid_fc_upper.append(valid_fc_upper)

    #

    # evaluate the error

    sarima_mape.append(forecast_accuracy(valid_fc, valid)['mape'])

    sarima_index.append(storedept)

df_mape['MAPE_SARIMA'] = pd.DataFrame({'MAPE_simeplemodel' : sarima_mape}, index = sarima_index)

df_mape.head()
# Compare predicted and actual test TS

def plot_forecast(stdep, train, trainfc, valid, validfc, validfclower, validfcupper, plottitle):

    fig, axes = plt.subplots(1, 1, figsize=(10,3), dpi=100, sharex=True)

    plt.plot(train, label='Training set')

    plt.plot(trainfc, label='Fit training set')

    plt.plot(validfc, color='darkgreen', label='Forecast')

    try:

        plt.fill_between(validfclower.index, 

                         validfclower, 

                         validfcupper, 

                         color='k', alpha=.15)

    except:

        pass

    plt.plot(valid, label='Validation set')

    plt.legend()

    plt.title(plottitle+" - Forecast on validation set of "+str(stdep))

    plt.show()
for isd, storedept in enumerate(list_store_dept): #[:2]:

    train = df_train_all2.loc[storedept,'Weekly_Sales'] ; valid = df_valid_pred.loc[storedept,'Weekly_Sales']

    train_fc = sarima_train_fc[isd]

    valid_fc = sarima_valid_fc[isd] ; valid_fc_lower = sarima_valid_fc_lower[isd] ; valid_fc_upper = sarima_valid_fc_upper[isd] ; 

    plot_forecast(storedept, train, train_fc, valid, valid_fc, valid_fc_lower, valid_fc_upper, 'SARIMA')
#Perform a grid search by: varying (p,d,q) and (P,D,Q), using the AIC as information criterion

from statsmodels.tsa.arima_model import ARIMA

import pmdarima as pm



sarimax_index = [] ; sarimax_mape = [] ; sarimax_train_fc = [] ; sarimax_valid_fc = []

sarimax_valid_fc_lower = [] ; sarimax_valid_fc_upper = []

#

for storedept in list_store_dept: # [:2]

    train = df_train_all2.loc[storedept,'Weekly_Sales'] ; valid = df_valid_all.loc[storedept,'Weekly_Sales']

    feattrain = df_train_all2.loc[storedept, list_inpfeat] ; featvalid = df_valid_all.loc[storedept, list_inpfeat]

    #

    # run the grid search for given timeseries

    smodel = pm.auto_arima(train, 

                           start_p=1, start_q=1, 

                           max_p=2, max_q=2,     # maximum p and q

                           d=None,               # let model determine 'd'

                           seasonal=True,        # Seasonality

                           m=52,                 # frequency of series

                           start_P=1, start_Q=1, # 

                           max_P=1, max_Q=1,     # maximum P and Q

                           D=1,                  # one "yearly" differentiation

                           exogenous=feattrain,  # exogeneous variables

                           test='adf',           # use adftest to find optimal 'd'

                           information_criterion='aic', # used to select best model

                           trace=True,           # print results whilst training

                           error_action='ignore',# ignore orders that don't work

                           suppress_warnings=True, 

                           stepwise=True,       # apply intelligent order search

                          )

    #

    # get the forecast TS

    n_periods = len(valid.index)

    fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True, exogenous=featvalid)

    in_sample_preds, in_sample_confint = smodel.predict_in_sample(return_conf_int=True, exogenous=feattrain)

    index_of_fc = valid.index 

    #

    # make forecast series

    train_fc = pd.Series(in_sample_preds, index=train.index)

    valid_fc = pd.Series(fitted, index=index_of_fc)

    valid_fc_lower = pd.Series(confint[:, 0], index=index_of_fc)

    valid_fc_upper = pd.Series(confint[:, 1], index=index_of_fc)

    #

    #print(smodel.summary())

    # save the forecast 

    sarimax_train_fc.append(train_fc) ; sarimax_valid_fc.append(valid_fc)

    sarimax_valid_fc_lower.append(valid_fc_lower) ; sarimax_valid_fc_upper.append(valid_fc_upper)

    #

    # evaluate the error

    sarimax_mape.append(forecast_accuracy(valid_fc, valid)['mape'])

    sarimax_index.append(storedept)

df_mape['MAPE_SARIMAX'] = pd.DataFrame({'MAPE_simeplemodel' : sarimax_mape}, index = sarimax_index)

df_mape.head()
for isd, storedept in enumerate(list_store_dept): #[:2]:

    train = df_train_all2.loc[storedept,'Weekly_Sales'] ; valid = df_valid_pred.loc[storedept,'Weekly_Sales']

    train_fc = sarimax_train_fc[isd]

    valid_fc = sarimax_valid_fc[isd] ; valid_fc_lower = sarimax_valid_fc_lower[isd] ; valid_fc_upper = sarimax_valid_fc_upper[isd] ; 

    #valid_fc.head()

    plot_forecast(storedept, train, train_fc, valid, valid_fc, valid_fc_lower, valid_fc_upper, 'SARIMAX')
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = check_arrays(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) #* 100
def create_lags(df, maxshift):

    # data is a pandas Series containing time series data

    data = pd.Series(...)

    # Shifts

    shifts = np.arange(1,maxshift+1)

    # Create a dictionary of time-shifted data

    many_shifts = {'lag_{}'.format(ii): df.shift(ii) for ii in shifts}

    # Convert them into a dataframe

    many_shifts = pd.DataFrame(many_shifts).fillna(0.)

    return many_shifts
import xgboost as xgb

from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV



Nlags = 10

inpfeature = list_inpdate  + ['lag_'+str(i+1) for i in range(Nlags)] # + list_inpfeat+ list_markdown['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']

list_rescv = ['params', 'mean_test_score', 'std_test_score', 'mean_test_score', 'std_test_score', 'rank_test_score']

#

# define list of models and parameters

list_models = [('RF', RandomForestRegressor()),

               ('XGB', xgb.XGBRegressor())

              ]



list_params = [#{'Ridge__alpha' : np.logspace(0,4,10)},

               {'max_depth' : [2, 4, 6, 8, 10, 12], #

                #'n_estimators' : [10, 20, 50, 100, 200, 500, 1000]

                },

               {'max_depth' : [2, 4, 6, 8, 10, 12], #

                #'n_estimators' : [5, 10, 20, 50, 100, 200]

                }

              ]

#

list_train_fc = [] ; list_valid_fc = [] ; 

list_bestparams = [] ; list_bestest = [] ; list_cvres = []

#

for im, model in enumerate(list_models): 

    list_bestparams2 = [] ; list_bestest2 = [] ; list_cvres2 = []

    list_train_fc2 = [] ; list_valid_fc2 = [] 

    list_mape = [] ; list_index = []

    for storedept in list_store_dept:

        print(storedept)

        #

        # introduce time-shifted features

        dftrain2 = df_train_all2.loc[storedept]

        dfvalid2 = df_valid_all.loc[storedept]

        dftrainlag = create_lags(dftrain2.Weekly_Sales, Nlags)

        dfvalidlag = create_lags(dfvalid2.Weekly_Sales, Nlags)

        dftrain2 = dftrain2.join(dftrainlag,on='Date')

        dfvalid2 = dfvalid2.join(dfvalidlag,on='Date')

        #

        # define training and validation sets

        X = dftrain2[inpfeature] #df_train_all2.loc[storedept,inpfeature]

        y = df_train_all2.loc[storedept,'Weekly_Sales']

        X_valid = dfvalid2[inpfeature] #df_valid_all.loc[storedept,inpfeature]

        y_valid = df_valid_all.loc[storedept,'Weekly_Sales']

        #

        # run grid search

        param_search = list_params[im]

        tscv = TimeSeriesSplit(n_splits=5)

        gsearch = GridSearchCV(estimator=model[1],        # choice of model

                               cv=tscv,                   # choice of splitting

                               param_grid=param_search,   # grid of parameters

                               verbose=1,                 # print messages

                               return_train_score=True,   # return train score in CV grid result

                               n_jobs=-1,                 # number of CPUs to be used

                               scoring='neg_mean_absolute_error' # metrics to be used

                              )

        #

        # save useful output

        gsearch.fit(X, y)

        list_bestparams2.append(gsearch.best_params_)

        list_bestest2.append(gsearch.best_estimator_)

        cvres = pd.DataFrame(gsearch.cv_results_)[list_rescv].sort_values(by='rank_test_score',ascending=True)

        list_cvres2.append(cvres)

        print('Best-fit parameters: ',gsearch.best_params_)

        #

        # get the forecast TS

        #n_periods = len(valid.index)

        valid_fc = pd.Series(gsearch.predict(X_valid), index=y_valid.index) #gsearch.predict(X_valid) 

        train_fc = pd.Series(gsearch.predict(X), index=X.index) #gsearch.predict(X)

        #

        # save the forecast 

        list_train_fc2.append(train_fc) ; list_valid_fc2.append(valid_fc)

        #

        # evaluate the error

        list_mape.append(forecast_accuracy(valid_fc, y_valid)['mape'])

        list_index.append(storedept)

    #

    list_train_fc.append(list_train_fc2) ; list_valid_fc.append(list_valid_fc2)

    list_bestparams.append(list_bestparams2) ; list_bestest.append(list_bestest2) ; list_cvres.append(list_cvres2)

    df_mape['MAPE_'+model[0]] = pd.DataFrame({'MAPE_'+model[0] : list_mape}, index = list_index)
df_mape.head()
fig, axes = plt.subplots(len(list_store_dept), 1, figsize=(8,8), dpi=100, sharex=False)

fig.tight_layout()

num_xgb = 1

for isd, storedept in enumerate(list_store_dept):

    #print(list_bestest[num_xgb][isd])

    xgb.plot_importance(list_bestest[num_xgb][isd], show_values=False, xlim=None, height=0.8, max_num_features=8, ax=axes[isd]) ; axes[isd].set_title(str(storedept))

    if isd != len(list_store_dept)-1:

        axes[isd].set_xlabel(' ')
# Compare predicted and actual test TS

def plot_forecast(stdep, train, trainfc, valid, validfc, validfclower, validfcupper, plottitle):

    fig, axes = plt.subplots(1, 1, figsize=(10,3), dpi=100, sharex=True)

    plt.plot(train, label='Training set')

    plt.plot(trainfc, label='Fit training set')

    plt.plot(validfc, color='darkgreen', label='Forecast')

    try:

        plt.fill_between(validfclower.index, 

                         validfclower, 

                         validfcupper, 

                         color='k', alpha=.15)

    except:

        pass

    plt.plot(valid, label='Validation set')

    plt.legend()

    plt.title(plottitle+" - Forecast on validation set of "+str(stdep))

    plt.show()
for im, model in enumerate(list_models): 

    for isd, storedept in enumerate(list_store_dept): #[:2]:

        train = df_train_all2.loc[storedept,'Weekly_Sales'] ; valid = df_valid_pred.loc[storedept,'Weekly_Sales']

        train_fc = list_train_fc[im][isd]

        valid_fc = list_valid_fc[im][isd]

        #valid_fc.head()

        plot_forecast(storedept, train, train_fc, valid, valid_fc, 0., 0., model[0])

        

        