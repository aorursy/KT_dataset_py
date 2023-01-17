import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
%matplotlib inline
%config InlineBackend.figure_format='retina'

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.float_format = '{:,.2f}'.format
pd.options.display.max_rows = 300
pd.options.display.max_columns = 1000

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

d = {'covid19_':'','covid':'','19':'','.csv':'','covid_19':'','__':''}

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        vars()[replace_all(filename.lower(),d)] = pd.read_csv(os.path.join(dirname, filename))

alldfs = [var for var in dir() if isinstance(eval(var), pd.core.frame.DataFrame)]

print(alldfs)
from google.cloud import automl
from google.cloud import storage
from google.cloud import bigquery

PROJECT_ID = 'covid19-kaggle-vfx'

storage_client = storage.Client(project=PROJECT_ID)
bigquery_client = bigquery.Client(project=PROJECT_ID)
automl_client = automl.AutoMlClient()
from IPython.display import Markdown, display

def printmd(string):
    display(Markdown(string))
    
def inspect(x,y):
    [printmd(z) for z in ['\n','___','## Dataset: %s' % y]]
    display(x.dtypes,
            x.head(),
            x.describe())
for x in alldfs:
    display(inspect(vars()[x],x))
time_series_confirmed.head()
conf_ts = time_series_confirmed.copy()

## Reshape dataset so we have a single column for dates
conf_ts = conf_ts.melt(id_vars=conf_ts.columns[:4],var_name='date',value_name='confirmed')
conf_ts['date'] = pd.to_datetime(conf_ts['date'])

## Correct a few typos
conf_ts['Country/Region'] = conf_ts['Country/Region'].replace({"Mainland China":"China",
                                                               "United Kingdom":"UK",
                                                               "Viet Nam":"Vietnam"})

## Remove negative days
conf_ts = conf_ts[conf_ts['confirmed'] > 0]

ts_min = conf_ts.groupby(['Country/Region'])['date'].min()
conf_ts['d0'] = conf_ts['Country/Region'].map(ts_min)
conf_ts['d0'] = conf_ts['date'] - conf_ts['d0']
conf_ts
conf_ts[conf_ts['Country/Region'] == 'Brazil'].sort_values('date')
conf_ts_temp = conf_ts.groupby(['d0','Country/Region'])['confirmed'].sum().reset_index().sort_values(by=['d0','Country/Region'])
fig = px.line(conf_ts_temp, x=conf_ts_temp.d0.dt.days, y="confirmed", color='Country/Region', title='Cases Spread', height=600, log_y=True)
fig.update_layout(xaxis_type='category',xaxis_title='Days from first case')
fig.show()
conf_ts_temp = conf_ts.groupby(['d0','Country/Region'])['confirmed'].sum().reset_index().sort_values(by=['d0','Country/Region'])
fig = px.line(conf_ts_temp, x=conf_ts_temp.d0.dt.days, y="confirmed", color='Country/Region', title='Cases Spread', height=600)
fig.update_layout(xaxis_type='category')
fig.show()
conf_ts_temp.dtypes
conf_ts_temp['diff'] = conf_ts_temp.sort_values(by=['Country/Region','d0']).groupby(['Country/Region'])['confirmed'].diff(1)
conf_ts_temp['ratio'] = conf_ts_temp.sort_values(by=['Country/Region','d0']).groupby(['Country/Region'])['confirmed'].pct_change(1)+1
conf_ts_temp['ratio_diff'] = conf_ts_temp.sort_values(by=['Country/Region','d0']).groupby(['Country/Region'])['diff'].pct_change(1)+1
conf_ts_temp[conf_ts_temp['Country/Region'] == 'Brazil']
conf_ts_temp2 = conf_ts_temp.groupby(['Country/Region'])['ratio'].last()
conf_ts_temp2 = conf_ts_temp2.reset_index().sort_values(by='ratio',ascending=False)
conf_ts_temp2
conf_ts_temp[conf_ts_temp['Country/Region'] == 'Brazil']
conf_ts_br = conf_ts_temp[conf_ts_temp['Country/Region'] == 'Brazil']
conf_ts_br['d0'] = conf_ts_br['d0'].dt.days.astype('int')

fig = px.scatter(conf_ts_br, x="d0", y="ratio", title='Cases Spread', height=600, text='ratio', trendline="OLS")
# fig.update_layout(xaxis_type='category')
fig.data[0].update(mode='markers+lines')
fig.show()
fig = px.scatter(conf_ts_temp2, x="Country/Region", y="ratio", title='Cases Spread', height=600)
fig.update_layout(xaxis_type='category')
fig.show()
conf_ts_temp3 = conf_ts_temp.sort_values(by='d0',ascending=True).groupby(['Country/Region'])['ratio_diff','ratio'].last()
conf_ts_temp3 = conf_ts_temp3.reset_index().sort_values(by='ratio_diff',ascending=False)
conf_ts_temp3
fig = px.scatter(conf_ts_temp3, x="Country/Region", y="ratio_diff", title='Cases Spread', height=600)
fig.update_layout(xaxis_type='category')
fig.show()
fig = px.scatter(conf_ts_temp3.sort_values(by='Country/Region',ascending=True), x="ratio", y="ratio_diff",color='Country/Region', title='Cases Spread', height=600)
fig.show()