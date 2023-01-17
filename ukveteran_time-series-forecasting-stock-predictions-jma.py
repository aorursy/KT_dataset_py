import numpy as np

import pandas as pd

from fbprophet import Prophet
df=pd.read_csv("../input/stock-price-predictions/Banana.csv",parse_dates=['Price Date'])

df.head()
df1=df.drop(["States", "Commodity"], axis = 1)

df1
df1.columns=["ds","y"]
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