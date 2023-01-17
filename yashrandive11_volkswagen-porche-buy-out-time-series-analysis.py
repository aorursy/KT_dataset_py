!pip install pmdarima -U
!pip install arch -U
!pip install yfinance -U
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
from pmdarima.arima import auto_arima
from pmdarima.arima import OCSBTest 
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model
import seaborn as sns
import yfinance
import warnings
warnings.filterwarnings("ignore")
sns.set()
raw_data = yfinance.download(tickers = "VOW3.DE, PAH3.DE, BMW.DE", interval = "1d", group_by = "ticker", auto_adjust = True, treads = True)
df = raw_data.copy()
# Starting Date
start_date = '2009-04-05'

# First Official Announcement (On this day VW announced they owned 49.9% of Porche)
ann_1 = '2009-12-09'

# Second Official Announcement(On this day VW announced Full Ownership of Porche at 50.01% )
ann_2 = '2012-07-05'

# Ending Date
end_date = '2014-01-01'

# Dieselgate Scandal
d_gate = '2015-09-20'
# Extracting closing prices
df['vol'] = df['VOW3.DE'].Close
df['por'] = df['PAH3.DE'].Close
df['bmw'] = df['BMW.DE'].Close

# Creating Returns
df['ret_vol'] = df['vol'].pct_change(1).mul(100)
df['ret_por'] = df['por'].pct_change(1).mul(100)
df['ret_bmw'] = df['bmw'].pct_change(1).mul(100)

# Creating Squared Returns
df['sq_vol'] = df.ret_vol.mul(df.ret_vol)
df['sq_por'] = df.ret_por.mul(df.ret_por)
df['sq_bmw'] = df.ret_bmw.mul(df.ret_bmw)

# Extracting Volume (Number of Purchases and Sales each day)
df['q_vol'] = df['VOW3.DE'].Volume
df['q_por'] = df['PAH3.DE'].Volume
df['q_bmw'] = df['BMW.DE'].Volume

df = df.asfreq('b')
df = df.fillna(method = 'bfill')
df = df.drop(['VOW3.DE','PAH3.DE', 'BMW.DE'], axis = 1)
df.vol[start_date:end_date].plot(figsize = (20,7), color = 'blue')
df.por[start_date:end_date].plot(color = 'green')
df.bmw[start_date:end_date].plot(color = 'gold')
plt.legend()
plt.show()
df.vol[start_date:ann_1].plot(figsize = (20,7), color = '#33B8FF')
df.por[start_date:ann_1].plot(color = '#49FF3A')
df.bmw[start_date:ann_1].plot(color = '#FEB628')

df.vol[ann_1:ann_2].plot(color = '#1E7EB2')
df.por[ann_1:ann_2].plot(color = '#2FAB25')
df.bmw[ann_1:ann_2].plot(color = '#BA861F')

df.vol[ann_2:end_date].plot(color = '#0E3A52')
df.por[ann_2:end_date].plot(color = '#225414')
df.bmw[ann_2:end_date].plot(color = '#7C5913')

plt.legend(['Volkswagen', 'Porche', 'BMW'])
plt.show()
print('Correlation between the companies from ' + str(start_date) + ' to ' + str(end_date) + '\n')
print('Volkswagen and Porche Correlation : \t' + str(df['vol'][start_date:end_date].corr(df['por'][start_date:end_date])))
print('Volkswagen and BMW Correlation : \t' + str(df['vol'][start_date:end_date].corr(df['bmw'][start_date:end_date])))
print('Porche and BMW Correlation : \t\t' + str(df['por'][start_date:end_date].corr(df['bmw'][start_date:end_date])))
print('Correlation between the companies from ' + str(start_date) + ' to ' + str(ann_1) + '\n')
print('Volkswagen and Porche Correlation : \t' + str(df['vol'][start_date:ann_1].corr(df['por'][start_date:ann_1])))
print('Volkswagen and BMW Correlation : \t' + str(df['vol'][start_date:ann_1].corr(df['bmw'][start_date:ann_1])))
print('Porche and BMW Correlation : \t\t' + str(df['por'][start_date:ann_1].corr(df['bmw'][start_date:ann_1])))
print('Correlation between the companies from ' + str(ann_1) + ' to ' + str(ann_2) + '\n')
print('Volkswagen and Porche Correlation : \t' + str(df['vol'][ann_1:ann_2].corr(df['por'][ann_1:ann_2])))
print('Volkswagen and BMW Correlation : \t' + str(df['vol'][ann_1:ann_2].corr(df['bmw'][ann_1:ann_2])))
print('Porche and BMW Correlation : \t\t' + str(df['por'][ann_1:ann_2].corr(df['bmw'][ann_1:ann_2])))
print('Correlation between the companies from ' + str(ann_2) + ' to ' + str(end_date) + '\n')
print('Volkswagen and Porche Correlation : \t' + str(df['vol'][ann_2:end_date].corr(df['por'][ann_2:end_date])))
print('Volkswagen and BMW Correlation : \t' + str(df['vol'][ann_2:end_date].corr(df['bmw'][ann_2:end_date])))
print('Porche and BMW Correlation : \t\t' + str(df['por'][ann_2:end_date].corr(df['bmw'][ann_2:end_date])))
print('Correlation between the companies from ' + str(end_date) + ' to ' + str(df.index[-1]) + '\n')
print('Volkswagen and Porche Correlation : \t' + str(df['vol'][end_date:].corr(df['por'][end_date:])))
print('Volkswagen and BMW Correlation : \t' + str(df['vol'][end_date:].corr(df['bmw'][end_date:])))
print('Porche and BMW Correlation : \t\t' + str(df['por'][end_date:].corr(df['bmw'][end_date:])))
print('Correlation between the companies in the Dieselgate Scandle from ' + str(end_date) + ' to ' + str(d_gate) + '\n')
print('Volkswagen and Porche Correlation : \t' + str(df['vol'][end_date:d_gate].corr(df['por'][end_date:d_gate])))
print('Volkswagen and BMW Correlation : \t' + str(df['vol'][end_date:d_gate].corr(df['bmw'][end_date:d_gate])))
print('Porche and BMW Correlation : \t\t' + str(df['por'][end_date:d_gate].corr(df['bmw'][end_date:d_gate])))
mod_pr_pre_vol = auto_arima(df.vol[start_date:ann_1], exogenous = df[['por','bmw']][start_date:ann_1], m= 5, max_p = 5, max_q = 5)

mod_pr_btn_vol = auto_arima(df.vol[ann_1:ann_2], exogenous = df[['por','bmw']][ann_1:ann_2], m= 5, max_p = 5, max_q = 5)

mod_pr_post_vol = auto_arima(df.vol[ann_2:end_date], exogenous = df[['por','bmw']][ann_2:end_date], m= 5, max_p = 5, max_q = 5)
mod_pr_pre_vol.summary()
mod_pr_btn_vol.summary()
mod_pr_post_vol.summary()
mod_pr_pre_por = auto_arima(df.por[start_date:ann_1], exogenous = df[['vol','bmw']][start_date:ann_1], m= 5, max_p = 5, max_q = 5)

mod_pr_btn_por = auto_arima(df.por[ann_1:ann_2], exogenous = df[['vol','bmw']][ann_1:ann_2], m= 5, max_p = 5, max_q = 5)

mod_pr_post_por = auto_arima(df.por[ann_2:end_date], exogenous = df[['vol','bmw']][ann_2:end_date], m= 5, max_p = 5, max_q = 5)
mod_pr_pre_por.summary()
mod_pr_btn_por.summary()
mod_pr_post_por.summary()
model_auto_pred_pr = auto_arima(df.vol[start_date:ann_1], 
                                exogenous = df[['por','bmw']][start_date:ann_1],
                                m = 5,
                                max_p= 5,
                                max_q = 5,
                                max_P = 5,
                                max_Q = 5,
                                trend = 'ct')

df_auto_pred_pr = pd.DataFrame(model_auto_pred_pr.predict(n_periods = len(df[ann_1:ann_2]),exogenous = df[['por', 'bmw']][ann_1:ann_2]),index=df[ann_1:ann_2].index)
df_auto_pred_pr[ann_1:ann_2].plot(figsize = (20,5), color = "red" )
df.vol[ann_1:ann_2].plot(color = "blue")
plt.title("VW Predictions(Porsche and BMW as Exog) vs Predictions", size = 24)
plt.legend(['Predictions'], ['Actuals'])
plt.show()
df['sq_vol'][start_date:ann_1].plot(figsize = (20,5), color = '#33B8FF' )
df['sq_vol'][ann_1:ann_2].plot(color = '#1E7EB2')
df['sq_vol'][ann_2:end_date].plot(color = '#0E3A52')
plt.title("VW Volatility", size = 24)
plt.show()
model_garch_pre = arch_model(df.ret_vol[start_date:ann_1], mean = "Constant", vol = "GARCH", p = 1, q = 1)
results_garch_pre = model_garch_pre.fit(update_freq = 5)

model_garch_btn = arch_model(df.ret_vol[ann_1:ann_2],  mean = "Constant", vol = "GARCH", p = 1, q = 1)
results_garch_btn = model_garch_btn.fit(update_freq = 5)

model_garch_post = arch_model(df.ret_vol[ann_2:end_date],  mean = "Constant", vol = "GARCH", p = 1, q = 1)
results_garch_post = model_garch_post.fit(update_freq = 5)
results_garch_pre.summary()
results_garch_btn.summary()
results_garch_post.summary()