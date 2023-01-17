import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import matplotlib.animation as animation

from IPython.display import HTML

import plotly.express as px

import plotly.graph_objects as go

import seaborn as sns

from plotly.subplots import make_subplots

%matplotlib inline





import plotly.tools as tls

import cufflinks as cf

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



init_notebook_mode(connected=True)



print(__version__) # requires version >= 1.9.0

cf.go_offline()
df_pull = pd.read_csv('https://github.com/nytimes/covid-19-data/raw/master/us-counties.csv')
df = df_pull

df['cases'] = df['cases'].astype(float)

df['log_cases'] = df.cases.apply(lambda x: np.log10(x))

df.deaths = df['deaths'].astype(float)

df['log_deaths'] = df.deaths.apply(lambda x: np.log(x))

df.tail()
df.drop(['fips'], axis=1)
df_reg=df.groupby(['county']).agg({'cases':'sum','deaths':'sum'}).sort_values(["cases"],ascending=False).reset_index()

df_reg.head(10)
fig = go.Figure(data=[go.Table(

    columnwidth = [50],

    header=dict(values=('county', 'cases', 'deaths'),

                fill_color='#104E8B',

                align='center',

                font_size=14,

                font_color='white',

                height=40),

    cells=dict(values=[df_reg['county'].head(10), df_reg['cases'].head(10), df_reg['deaths'].head(10)],

               fill=dict(color=['#509EEA', '#A4CEF8',]),

               align='right',

               font_size=12,

               height=30))

])



fig.show()
df_reg.iplot(kind='box')
fig = px.pie(df_reg.head(10),

             values="cases",

             names="county",

             title="cases",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo='value+label')

fig.show()
fig = px.pie(df_reg.head(10),

             values="deaths",

             names="county",

             title="deaths",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo='value+label')

fig.show()
df_county=df.groupby(['date','county']).agg({'cases':'sum','deaths':'sum'}).sort_values(["cases"],ascending=False)

df_county.head(10)
dfd = df_county.groupby('date').sum()

dfd.head()
dfd[['cases','deaths']].iplot(title = 'US Counties Situation Over Time')
dfd['Active'] = dfd['cases']-dfd['deaths']

dfd['Active'] 
from fbprophet import Prophet
df1=df.rename(columns={"date": "ds", "cases": "y"})

df1
m = Prophet()

m.fit(df1)
future = m.make_future_dataframe(periods=365)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
def prepare_county(c,state):

    county = df[(df.county == c) & (df.state == state)].sort_values('date')

    county.set_index('date',inplace=True)

    return county



def prepare_us(nyt_df):

    df_all = df.groupby('date').apply(lambda df: df.cases.sum())

    df_all = df_all.to_frame()

    df_all.columns = ['cases']

    df_all['log_cases'] = df_all.cases.apply(lambda x: np.log10(x))

    return df_all
import matplotlib.dates as mdates

import matplotlib.ticker as tkr

def plot_df(df,lower_bound=10):

    # plot time series cases, log cases and phase plot

    fig, (ax0,ax1,ax2) = plt.subplots(1,3,figsize = (12,4))

    fig.autofmt_xdate()

    plt.subplots_adjust(bottom=.2)

    df.cases.plot(ax = ax0)

    #ax0.set_title('Cases')

    ax0.set_ylabel('Cases')

    ax0.set_xlabel('')

    

    ax0.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=[7,14,21,28]))

    Y = df.log_cases

    Y.plot(ax=ax1)

    #ax1.set_title('Log10 Cases')

    ax1.set_ylabel('Log10 Cases')

    ax1.set_xlabel('')

    ax1.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=[7,14,21,28]))

    # plot change in cases against cases (x-axis)

    df1 = df[df.cases > lower_bound].copy()

    df1['dif'] = df1.loc[:,'cases'].diff()

    df1['dif_smooth'] = df1.dif.rolling(SMOOTHING_PERIOD).mean()

    df1['dif_smooth'] = df1['dif_smooth'].apply(lambda x: np.nan if x <=0 else x)

    ax2.scatter(np.log10(df1['cases']),np.log10(df1['dif_smooth']))

    ax2.set_xlabel('Log10 Total Cases')

    ax2.set_ylabel('Log10 Change in Cases')

    return fig,df
SMOOTHING_PERIOD = 10

fig,tu= plot_df(prepare_county('Pima','Arizona'))
fig,us = plot_df(prepare_us(df))