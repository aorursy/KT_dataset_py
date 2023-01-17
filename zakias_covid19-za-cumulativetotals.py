import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.offline as py
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/covid19zacumulativetotals/covid19-za-cumulativetotals.csv')
df.head()
df.tail()
plt.figure(figsize=(30,15), dpi=80, linewidth=1000)
x = df['DATE']
y = df['TOTAL']
plt.plot(x, y)
plt.xlabel('Date')
plt.ylabel('Total Cumulative Positive Cases')
plt.title('Total Cumulative Positive COVID-19 Cases in South Africa from 5 March 2020 to 2 April 2020')
plt.show()
df = df[['DATE','TOTAL']]
#df.columns = ['ds', 'y']
df = df.rename(columns={'DATE': 'ds', 'TOTAL': 'y'})
#df = pd.set_option('precision', 0)
df.head()
df.isnull().sum()
df.set_index('ds').y.plot()
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=28)
future.tail()
forecast = model.predict(future)
forecast.tail()
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
model.plot(forecast);
model.plot_components(forecast);
py.init_notebook_mode()
fig = plot_plotly(model, forecast)  # This returns a plotly Figure
py.iplot(fig);
metrics_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
metrics_df.tail()
metrics_df.dropna(inplace=True)
metrics_df.tail()
r2_score(metrics_df.y, metrics_df.yhat)
mean_squared_error(metrics_df.y, metrics_df.yhat)
mean_absolute_error(metrics_df.y, metrics_df.yhat)
