import pandas as pd

from fbprophet import Prophet
df = pd.read_csv('../input/example-wp-log-peyton-manningcsv/example_wp_log_peyton_manning.csv')

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