import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 300) # specifies number of rows to show
pd.set_option('display.max_columns', 300)
pd.options.display.float_format = '{:40,.0f}'.format # specifies default number format to 4 decimal places
pd.options.display.max_colwidth
pd.options.display.max_colwidth = 1000
# This line tells the notebook to show plots inside of the notebook
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sb
twitter_insta = pd.read_csv('../input/frieze-eda.xlsx')

twitter_insta.sample(2)
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go
from plotly.graph_objs import *

#You can also plot your graphs offline inside a Jupyter Notebook Environment. 
#First you need to initiate the Plotly Notebook mode as below:
init_notebook_mode(connected=True)
twitter_insta.describe()
twitter_insta['DailyFreq'] = twitter_insta.groupby('Day')['Day'].transform('count')

data = [Bar(x=twitter_insta['Day'],  #change back to location_freq['Location']
            y=twitter_insta['DailyFreq'])] #change back to location_freq['Frequency']

layout = Layout(
    title="Number of Tweets by Day",
    xaxis=dict(title='Day in October'),
    yaxis=dict(title='Number of Tweets'),
    width = 700
)

fig = Figure(data=data, layout=layout)

iplot(fig, filename='jupyter/basic_bar')
twitter_insta['HourlyFreq'] = twitter_insta.groupby('Hour')['Hour'].transform('count')

data = [Bar(x=twitter_insta['Hour'],  #change back to location_freq['Location']
            y=twitter_insta['HourlyFreq'])] #change back to location_freq['Frequency']

layout = Layout(
    title="Number of Tweets by Hour",
    xaxis=dict(title='Hour of Day'),
    yaxis=dict(title='Number of Tweets'),
    width = 700
)

fig = Figure(data=data, layout=layout)

iplot(fig, filename='jupyter/basic_bar')