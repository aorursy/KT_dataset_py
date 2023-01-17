import os

import pandas as pd



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.io as pio

import plotly.graph_objs as go

from plotly.subplots import make_subplots



# Show charts when running kernel

init_notebook_mode(connected=True)



# Change default background color for all visualizations

layout=go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(250,250,250,0.8)')

fig = go.Figure(layout=layout)

templated_fig = pio.to_templated(fig)

pio.templates['my_template'] = templated_fig.layout.template

pio.templates.default = 'my_template'



def plot_scatter(x, y, title):

    fig = go.Figure(go.Scatter(x=x, y=y, name=title))

    fig.update_layout(title_text=title)

    fig.show()
ETF_NAME = 'SPY'

ETF_DIRECTORY = '/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Data/ETFs/'



df = pd.read_csv(os.path.join(ETF_DIRECTORY, ETF_NAME.lower() + '.us.txt'), sep=',')

df.head()
df.describe()
fig = go.Figure([go.Ohlc(x=df.Date,

                         open=df.Open,

                         high=df.High,

                         low=df.Low,

                         close=df.Close)])

fig.update(layout_xaxis_rangeslider_visible=False)

fig.show()
fig = go.Figure(go.Bar(x=df.Date, y=df.Volume, name='Volume', marker_color='red'))

fig.show()
df['EMA_9'] = df['Close'].ewm(5).mean().shift()

df['SMA_50'] = df['Close'].rolling(50).mean().shift()

df['SMA_100'] = df['Close'].rolling(100).mean().shift()

df['SMA_200'] = df['Close'].rolling(200).mean().shift()



fig = go.Figure()

fig.add_trace(go.Scatter(x=df.Date, y=df.EMA_9, name='EMA 9'))

fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_50, name='SMA 50'))

fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_100, name='SMA 100'))

fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_200, name='SMA 200'))

fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name='Close', line_color='dimgray', opacity=0.3))

fig.show()
def RSI(df, n=14):

    close = df['Close']

    delta = close.diff()

    delta = delta[1:]

    pricesUp = delta.copy()

    pricesDown = delta.copy()

    pricesUp[pricesUp < 0] = 0

    pricesDown[pricesDown > 0] = 0

    rollUp = pricesUp.rolling(n).mean()

    rollDown = pricesDown.abs().rolling(n).mean()

    rs = rollUp / rollDown

    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi



num_days = 365

df['RSI'] = RSI(df).fillna(0)

fig = go.Figure(go.Scatter(x=df.Date.tail(num_days), y=df.RSI.tail(num_days)))

fig.show()
EMA_12 = pd.Series(df['Close'].ewm(span=12, min_periods=12).mean())

EMA_26 = pd.Series(df['Close'].ewm(span=26, min_periods=26).mean())

MACD = pd.Series(EMA_12 - EMA_26)

MACD_signal = pd.Series(MACD.ewm(span=9, min_periods=9).mean())



fig = make_subplots(rows=2, cols=1)

fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name='Close'), row=1, col=1)

fig.add_trace(go.Scatter(x=df.Date, y=EMA_12, name='EMA 12'), row=1, col=1)

fig.add_trace(go.Scatter(x=df.Date, y=EMA_26, name='EMA 26'), row=1, col=1)

fig.add_trace(go.Scatter(x=df.Date, y=MACD, name='MACD'), row=2, col=1)

fig.add_trace(go.Scatter(x=df.Date, y=MACD_signal, name='Signal line'), row=2, col=1)

fig.show()
# https://stackoverflow.com/questions/30261541/slow-stochastic-implementation-in-python-pandas

def stochastic(df, k, d):

    df = df.copy()

    low_min  = df['Low'].rolling(window=k).min()

    high_max = df['High'].rolling( window=k).max()

    df['stoch_k'] = 100 * (df['Close'] - low_min)/(high_max - low_min)

    df['stoch_d'] = df['stoch_k'].rolling(window=d).mean()

    return df



stochs = stochastic(df, k=14, d=3)



fig = go.Figure()

fig.add_trace(go.Scatter(x=df.Date.tail(365), y=stochs.stoch_k.tail(365), name='K stochastic'))

fig.add_trace(go.Scatter(x=df.Date.tail(365), y=stochs.stoch_d.tail(365), name='D stochastic'))

fig.show()