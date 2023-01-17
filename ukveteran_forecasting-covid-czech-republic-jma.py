# importing libreries and changing their name

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib

plt.style.use('fivethirtyeight')
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
# read the excel file
dat = pd.read_csv("../input/covid19-czech-republic/covid19-czech.csv")
dat.head()
# Remove columns which are not required in predictions

cols = ['age', 'sex', 'infected_abroad','region','sub_region', 'infected_in_country'
       ,'daily_infected','daily_cured','infected','cured', 'death'
       ,'daily_deaths', 'daily_cum_tested',
       'daily_cum_infected','daily_cum_cured', 'daily_cum_deaths', 'region_accessories_qty']
dat.drop(cols, axis = 1, inplace = True)
dat.head()
df=dat.rename(columns={"date": "ds"})
df
df1=df.rename(columns={"daily_tested": "y"})
df1
from fbprophet import Prophet
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