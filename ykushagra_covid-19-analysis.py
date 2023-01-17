# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go

import plotly.express as px

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
confirmed_df = pd.read_csv("../input/covid19-analysis/time_series_covid19_confirmed_global.csv")

print(confirmed_df.shape)

deaths_df = pd.read_csv("../input/covid19-analysis/time_series_covid19_deaths_global.csv")

print(deaths_df.shape)

recovered_df = pd.read_csv("../input/covid19-analysis/time_series_covid19_recovered_global.csv")

print(recovered_df.shape)
confirmed_df.columns
confirmed_df.head()
confirmed_df[confirmed_df['Country/Region']=='India']
#how many unique country are there

confirmed_df['Country/Region'].nunique()
#imputing data 

confirmed_df = confirmed_df.replace(np.nan,'',regex = True)

deaths_df = deaths_df.replace(np.nan,'' ,regex = True)

recovered_df = recovered_df.replace(np.nan,'',regex = True)
#total confirmed corona virus cases globally

confirmed_ts = confirmed_df.copy().drop(['Lat','Long','Country/Region','Province/State'],axis = 1)

confirmed_ts_summary = confirmed_ts.sum()

confirmed_ts_summary





#figure-1

fig_1 = go.Figure(data=go.Scatter(x = confirmed_ts_summary.index, y = confirmed_ts_summary.values ,mode = 'lines+markers'))

fig_1.update_layout(title="Total Coronavirus Confirmed Cases(Globally)",

                    yaxis_title='Confirmed Cases', xaxis_tickangle=315)

fig_1.show()
#total deaths due to corona virus cases globally

deaths_ts = deaths_df.copy().drop(['Lat','Long','Country/Region','Province/State'],axis = 1)

deaths_ts_summary = deaths_ts.sum()

deaths_ts_summary



#figure-2

fig_2 = go.Figure(data=go.Scatter(x = deaths_ts_summary.index, y = deaths_ts_summary.values ,mode = 'lines+markers'))

fig_2.update_layout(title="Total Coronavirus Deaths Cases(Globally)",

                    yaxis_title='Deaths Cases', xaxis_tickangle=315)

fig_2.show()
#total recovered corona virus cases globally

recovered_ts = recovered_df.copy().drop(['Lat','Long','Country/Region','Province/State'],axis = 1)

recovered_ts_summary = recovered_ts.sum()

recovered_ts_summary



#figure-3

fig_3 = go.Figure(data=go.Scatter(x = recovered_ts_summary.index, y = recovered_ts_summary.values ,mode = 'lines+markers'))

fig_3.update_layout(title="Total Coronavirus Recovered Cases(Globally)",

                    yaxis_title='Recovered Cases', xaxis_tickangle=315)

fig_3.show()
confirmed_agg_ts = confirmed_df.copy().drop(['Lat','Long','Country/Region','Province/State'],axis = 1).sum()

death_agg_ts = deaths_df.copy().drop(['Lat','Long','Country/Region','Province/State'],axis = 1).sum()

recovered_agg_ts = recovered_df.copy().drop(['Lat','Long','Country/Region','Province/State'],axis = 1).sum()



#timeseries data for activecases



active_agg_ts = pd.Series(

    data=np.array(

     [x1-x2-x3 for (x1,x2,x3) in zip(confirmed_agg_ts.values,death_agg_ts.values,recovered_agg_ts.values)]),

     index=confirmed_agg_ts.index)

active_agg_ts