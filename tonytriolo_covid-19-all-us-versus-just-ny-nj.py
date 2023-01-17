# Import libraries and define useful functions



# import pandas as pd

from urllib.request import urlopen

import json

import plotly.express as px

import plotly.graph_objects as go

import plotly.offline as po

import datetime

import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression



# put plotly into notebook mode for graphics rendering

po.init_notebook_mode(connected=False)



# function to compute time to double for exponential growth

# This won't be valid when the growth curve starts to look logistic

def doubling_time(x,y):

    # pass in two numpy arrays 

    # The y values are assumed to be already in log

    reg = LinearRegression().fit(x,y)

    dbl_time = int(np.log(2)/reg.coef_.item())

    

    return dbl_time
# import state data from NYT

df = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv')

df.head()
nj=df['state']=='New Jersey'

ny=df['state']=='New York'

df_without_nynj=df[~nj & ~ny].copy()

all_us_data = df.groupby(['date']).sum()

all_without_nynj = df_without_nynj.groupby(['date']).sum()

g =all_us_data.groupby(['date'])

all_dates = [name for name,unused_df in g]

g =all_without_nynj.groupby(['date'])

all_dates_wo = [name for name,unused_df in g]
container = []

trace=go.Scatter(

    x=all_dates,

    y=all_us_data['cases'],

    name='all cases')

container.append(trace)

trace=go.Scatter(

    x=all_dates_wo,

    y=all_without_nynj['cases'],

    name='without NY and NJ')

container.append(trace)



tick_space = 86400000.0*2

fig = go.Figure(container)

ttext = "COVID # of cases as of "+datetime.datetime.today().strftime('%m/%d/%Y')

fig.update_layout(xaxis_type="date", yaxis_type="log", title=ttext, xaxis_title='date')

fig.update_layout(xaxis_dtick=tick_space,xaxis_tickformat="%m/%d")

fig.show()

all_us_data['d_cases']=all_us_data['cases'].shift(-1)-all_us_data['cases']

all_without_nynj['d_cases']=all_without_nynj['cases'].shift(-1)-all_without_nynj['cases']



# do a 7 day moving average to remove the "weekend effect" over time

all_us_data['ma_d_cases']=all_us_data.d_cases.rolling(7).mean()

all_without_nynj['ma_d_cases']=all_without_nynj.d_cases.rolling(7).mean()



start = 46

container = []

trace=go.Bar(

    x=all_dates[start:],

    y=all_us_data['d_cases'][start:],

    name='All US')

container.append(trace)



trace=go.Scatter(

    x=all_dates[start:],

    y=all_us_data['ma_d_cases'][start:],

    name='smoothed all US')

container.append(trace)



trace=go.Bar(

    x=all_dates_wo[start:],

    y=all_without_nynj['d_cases'][start:],

    name='w/o NY, NJ')

container.append(trace)



trace=go.Scatter(

    x=all_dates_wo[start:],

    y=all_without_nynj['ma_d_cases'][start:],

    name='smoothed w/o NY, NJ')

container.append(trace)



tick_space = 86400000.0*2

fig = go.Figure(container)

ttext = "US Cases Change Data as of "+datetime.datetime.today().strftime('%m/%d/%Y')

fig.update_layout(xaxis_type="date", yaxis_type="linear", title=ttext, xaxis_title='date',yaxis_title='new cases')

fig.update_layout(xaxis_dtick=tick_space,xaxis_tickformat="%m/%d")

fig.show()
all_us_data['d_deaths']=all_us_data['deaths'].shift(-1)-all_us_data['deaths']

all_without_nynj['d_deaths']=all_without_nynj['deaths'].shift(-1)-all_without_nynj['deaths']



all_us_data['ma_d_deaths']=all_us_data.d_deaths.rolling(7).mean()

all_without_nynj['ma_d_deaths']=all_without_nynj.d_deaths.rolling(7).mean()



start = 46

container = []

trace=go.Bar(

    x=all_dates[start:],

    y=all_us_data['d_deaths'][start:],

    name='All US')

container.append(trace)



trace=go.Scatter(

    x=all_dates[start:],

    y=all_us_data['ma_d_deaths'][start:],

    name='smoothed all US')

container.append(trace)



trace=go.Bar(

    x=all_dates_wo[start:],

    y=all_without_nynj['d_deaths'][start:],

    name='w/o NY, NJ')

container.append(trace)



trace=go.Scatter(

    x=all_dates_wo[start:],

    y=all_without_nynj['ma_d_deaths'][start:],

    name='smoothed w/o NY, NJ')

container.append(trace)



tick_space = 86400000.0*2

fig = go.Figure(container)

ttext = "US Deaths Change Data as of "+datetime.datetime.today().strftime('%m/%d/%Y')

fig.update_layout(xaxis_type="date", yaxis_type="linear", title=ttext, xaxis_title='date',yaxis_title='new deaths')

fig.update_layout(xaxis_dtick=tick_space,xaxis_tickformat="%m/%d")

fig.show()