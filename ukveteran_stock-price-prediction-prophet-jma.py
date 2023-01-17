import pickle
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import pandas_datareader.data as web
from dateutil.relativedelta import relativedelta
import datetime
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation,performance_metrics
from fbprophet.plot import plot_cross_validation_metric, add_changepoints_to_plot
from sklearn.preprocessing import MinMaxScaler
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
def get_stock_data(symbol,res="yahoo"):
    start_date = datetime.date.today() - relativedelta(years=years)
    end_date = datetime.date.today()
    df = web.DataReader(symbol, res, start_date, end_date).dropna()   
    return df
def calculate_prediction_error(df, pred_day_count):
    df['e'] = df['y'] - df['yhat']
    df['p'] = 100 * df['e'] / df['y']
    prediction = df[-pred_day_count:]
    err = lambda err_code: np.mean(np.abs(prediction[err_code]))
    return {'MAPE': err('p'), 'MAE': err('e')}
train = True
symbol = 'ISCTR.IS'
years = 5 # We train with 5 years of data
ay = 3  # We predict 3 months of trend direction
np.random.seed(13)
symbol_data = get_stock_data(symbol)
orgDatLen = symbol_data.shape[0]
# Facebook research prophet library needs 'ds' date and 'y' value columns inorder to predict linear growth
symbol_data = symbol_data.reset_index().rename(columns={'Date':'ds'})
symbol_data['y'] = symbol_data['Adj Close']
symbol_data
# We could have used internal future = model.make_future_dataframe(periods=365) function but we need just bussiness days
future = pd.DataFrame()
future['ds'] = pd.bdate_range(symbol_data['ds'][0], pd.datetime.today()+pd.DateOffset(months=ay))
future
%%time
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
if train:
    model.fit(symbol_data)
    with open(symbol+".model", "wb") as f:
        pickle.dump(model, f)
else:
    with open(symbol+".model", "rb") as f:
        model = pickle.load(f)
%%time
forecast = model.predict(future)
forecast = forecast.merge(symbol_data,on=['ds'],how='left')
# Plot an interactive chart with plotly inorder to analyze over webbrowsers
figure = py.iplot({"data": [
    go.Scatter(x=symbol_data['ds'], y=symbol_data['Adj Close'], name='Price'),    
    go.Candlestick(x=symbol_data.ds,
                   open=symbol_data.Open,
                   high=symbol_data.High,
                   low=symbol_data.Low,
                   close=symbol_data.Close, visible='legendonly', name='OHLC'),
    go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prediction'),
    go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', name='Upper Band'),
    go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', name='Lower Band'),
    go.Scatter(x=forecast['ds'], y=forecast['trend'], name='Trend')]})
import pandas as pd
from fbprophet import Prophet
m = Prophet()
m.fit(symbol_data)
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