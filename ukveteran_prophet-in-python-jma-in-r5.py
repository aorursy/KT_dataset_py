import pandas as pd

from fbprophet import Prophet
df = pd.read_csv('../input/log-r-outliers/example_wp_log_R_outliers1.csv')

df.head()
m = Prophet()

m.fit(df)
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
from fbprophet.plot import add_changepoints_to_plot

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
df1 = pd.read_csv('../input/log-r-outliers/example_wp_log_R_outliers2.csv')

df1.head()
n = Prophet()

n.fit(df)
future = n.make_future_dataframe(periods=365)

future.tail()
forecast = n.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = n.plot(forecast)
fig2 = n.plot_components(forecast)
from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()



fign = plot_plotly(n, forecast)  # This returns a plotly Figure

py.iplot(fign)
from fbprophet.plot import add_changepoints_to_plot

fignn = n.plot(forecast)

a = add_changepoints_to_plot(fignn.gca(), n, forecast)