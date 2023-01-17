import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from pandas.tseries.offsets import DateOffset

import matplotlib.pyplot as plt
%matplotlib inline

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/avocado-prices-2020/avocado-updated-2020.csv')
df.head()
df.columns = ['Date', 'Val',"","","","","","","","","","",""]
df.head()
df.info()
df.isnull().sum()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index ('Date', inplace = True)
df.index
df_new = df['2015-01-04':]
df_new.tail()
df_new.describe().transpose()
time_series = df_new['Val']
type(time_series)
time_series.rolling(12).mean().plot(label = '12 Months Rolling Mean', figsize = (16,10))
time_series.rolling(12).std().plot(label = '12 Months Rolling Std')
time_series.plot()
plt.legend();