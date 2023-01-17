import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from math import *

sns.set()
raw_csv_data = pd.read_csv('../input/Index2018.csv')

raw_csv_data
df_comp = raw_csv_data.copy()
df_comp.date = pd.to_datetime(df_comp.date, dayfirst= True)

df_comp.set_index('date', inplace = True)

df_comp = df_comp.drop(['spx', 'dax', 'nikkei'], axis = 1)

df_comp
df_comp['market_value'] = df_comp.ftse

del df_comp['ftse']
df_comp = df_comp.asfreq('b')

df_comp = df_comp.fillna(method = 'ffill')

df_comp
size = int(len(df_comp)*0.8)

df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]
df.tail()
df_test.head()
from scipy.stats.distributions import chi2
def LLR_test(mod_1, mod_2, DF=1):

    L1 = mod_1.fit().llf

    L2 = mod_2.fit().llf

    LR = (2*(L2-L1))

    p = chi2.sf(LR,DF).round(3)

    return p
import statsmodels.tsa.stattools as sts
sts.adfuller(df.market_value)
df['returns'] = df.market_value.pct_change(1).mul(100)
df['sq_returns'] = df.returns.mul(df.returns)
df.returns.plot(figsize = (20,5))

plt.title("Returns", size = 24)

plt.show()
df.sq_returns.plot(figsize = (20,5))

plt.title("Volatility", size = 22)

plt.show()
import statsmodels.graphics.tsaplots as sgt
sgt.plot_pacf(df.returns[1:], lags = 40, zero = False, method = ('ols'))

plt.title('PACF Returns', size = 22)

plt.show()
sgt.plot_pacf(df.sq_returns[1:], lags = 40, zero = False, method = ('ols'))

plt.title("PACF Volatility", size = 22)

plt.show()
!pip install arch -U
from arch import arch_model
model_arch_1 = arch_model(df.returns[1:])

results_arch_1 = model_arch_1.fit(update_freq = 5)

results_arch_1.summary()
model_arch_1 = arch_model(df.returns[1:], 

                          mean = "Constant",

                          vol = "ARCH",

                          p = 1)

results_arch_1 = model_arch_1.fit(update_freq = 5)

results_arch_1.summary()

model_arch_2 = arch_model(df.returns[1:], mean = "Constant", vol = "ARCH", p = 2)

results_arch_2 = model_arch_2.fit(update_freq = 5)

results_arch_2.summary()
model_arch_3 = arch_model(df.returns[1:], mean = "Constant", vol = "ARCH", p = 3)

results_arch_3 = model_arch_3.fit(update_freq = 5)

results_arch_3.summary()
from arch import arch_model
model_garch_1_1 = arch_model(df.returns[1:], mean = "Constant", vol = "GARCH", p = 1, q = 1)

results_garch_1_1 = model_garch_1_1.fit(update_freq = 5)

results_garch_1_1.summary()