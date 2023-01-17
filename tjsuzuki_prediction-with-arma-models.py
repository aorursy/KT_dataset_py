import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
%matplotlib inline

train = pd.read_csv('../input/sales_train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')
items = pd.read_csv('../input/items.csv')
item_cats = pd.read_csv('../input/item_categories.csv')
shops = pd.read_csv('../input/shops.csv')
print("train:", train.shape)
print(train.head())
print("test:", test.shape)
print(test.head())
print("submission:",submission.shape)
print(submission.head())
print("items:",items.shape)
#print(items.head)
print("item_cats:",item_cats.shape)
#print(item_cats)
print("shops:",shops.shape)
#print(shops)

### Sales for each shops ###

SaleEachShop = train.groupby(["shop_id"],as_index=False)["item_cnt_day"].sum()
SaleEachShop = SaleEachShop.sort_values(by='item_cnt_day', ascending = False)
print(SaleEachShop.head())
print(SaleEachShop.tail())
ax = SaleEachShop.plot(y=["item_cnt_day"], bins=10, alpha=0.5, figsize=(16,4), kind='hist')
### Sales for each item ###

SaleEachItem = train.groupby(["item_id"],as_index=False)["item_cnt_day"].sum()
SaleEachItem = SaleEachItem.sort_values(by='item_cnt_day', ascending = False)
print(SaleEachItem["item_cnt_day"].describe())
print(SaleEachItem.head())
MaxSale = SaleEachItem["item_cnt_day"].max()
Q95 = SaleEachItem["item_cnt_day"].quantile(.95)
print("quantile 0.95 = {0}".format(Q95))
SaleEachItem[SaleEachItem["item_cnt_day"]< Q95].plot(y=["item_cnt_day"], bins=30, alpha=0.5, figsize=(16,4), title="~95%",kind='hist')

Monthly_sale_shop = train.groupby(["date_block_num","shop_id"],as_index=False)["item_cnt_day"].sum()
MaxSpan = Monthly_sale_shop["date_block_num"].max()
for i in range(len(shops)):
    Monthly_sale_shop[Monthly_sale_shop["shop_id"] == i].plot(x="date_block_num",y="item_cnt_day",xlim=[0, MaxSpan],
                                                              title="shop {0}".format(i))

Monthly_sale_total = train.groupby(["date_block_num"],as_index=False)["item_cnt_day"].sum()
Monthly_sale_total.plot(x="date_block_num",y="item_cnt_day")

import statsmodels.api as sm

### The index should be given in Datetime format to use seasonal_decompose
Monthly_sale_total["item_cnt_day"].index = pd.DatetimeIndex(freq="m",start='2013-1',periods=len(Monthly_sale_total))

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(Monthly_sale_total["item_cnt_day"], lags=20, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(Monthly_sale_total["item_cnt_day"], lags=20, ax=ax2)

res = sm.tsa.seasonal_decompose(Monthly_sale_total["item_cnt_day"], freq=3)
res.plot()

from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):

    #Plot rolling statistics:
    NWindow = 5
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(pd.rolling_mean(timeseries, window=NWindow), color='red', label='Rolling Mean')
    plt.plot(pd.rolling_std(timeseries,  window=NWindow), color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.show(block=False)
    
    #Dickey-Fuller test:
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(Monthly_sale_total["item_cnt_day"])

ModData = Monthly_sale_total - Monthly_sale_total.shift(periods=12)
ModData = ModData.dropna()
test_stationarity(ModData["item_cnt_day"])

import statsmodels.api as sm

### The index should be given in Datetime format to use seasonal_decompose
ModData.index = pd.DatetimeIndex(freq="m",start='2014-1',periods=len(ModData))

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(ModData["item_cnt_day"], lags=20, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(ModData["item_cnt_day"], lags=20, ax=ax2)

res = sm.tsa.seasonal_decompose(ModData["item_cnt_day"], freq=3)
res.plot()

res = sm.tsa.arma_order_select_ic(ModData["item_cnt_day"], ic='aic', trend='nc')
res
ARMA_3_1 = sm.tsa.ARMA(ModData["item_cnt_day"], (3,1)).fit()
print(ARMA_3_1.summary())
### Autocorrelation function of the residual ###

resid = ARMA_3_1.resid
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=20, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=20, ax=ax2)
### prediction with ARMA model

pred = ARMA_3_1.predict('2015-1-31', '2016-010-31')
 
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(211)
ax.set_xlim([datetime.date(2013, 1, 31), datetime.date(2016, 10, 31)])
Shifted = Monthly_sale_total["item_cnt_day"].copy()
Shifted.index = pd.date_range('2014-01-01', periods = len(Shifted), freq = 'M')
ax.plot(Monthly_sale_total["item_cnt_day"],label="orig")
ax.plot(pred+Shifted, "r",label="pred")
plt.legend()
