import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualisation Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Date & Time
from datetime import date, datetime, timedelta

warnings.simplefilter(action='ignore', category=FutureWarning)
%matplotlib inline

# plt.style.use('ggplot')
plt.style.use('seaborn-white')
font = {
    'family' : 'normal',
    'weight' : 'bold',
    'size'   : 13
}
plt.rc('font', **font)
# read the excel file
dat = pd.read_csv("../input/monthlymilkproductionpounds/monthly-milk-production-pounds.csv")
dat.head()
from fbprophet import Prophet
df=dat.rename(columns={"Month": "ds", "Monthly milk production: pounds per cow. Jan 62 ? Dec 75": "y"})
df
df.isnull().sum()
df1=df.dropna()  
df1
m = Prophet()
m.fit(df1)
future = m.make_future_dataframe(periods=365)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(m, forecast)  # This returns a plotly Figure
py.iplot(fig)