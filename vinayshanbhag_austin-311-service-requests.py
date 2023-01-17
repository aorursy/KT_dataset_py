# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.offline as offline

offline.init_notebook_mode()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/austin_311_service_requests.csv')

df = df[(df['county'].isin(['TRAVIS','WILLIAMSON'])) & (df['city']=='AUSTIN') & (~df.created_date.isnull())]

df = df[~df['status'].isin(['CancelledTesting','Duplicate (Closed)','Duplicate (closed)','Duplicate (open)','TO BE DELETED',''])]

df.created_date = pd.to_datetime(df.created_date)

df['month'] = pd.DatetimeIndex(df['created_date']).month.astype(int)

df.last_update_date= pd.to_datetime(df.last_update_date)

df.close_date= pd.to_datetime(df.close_date)
incidents_by_month = df.month.value_counts()

incidents_by_month.sort_index(ascending=True, inplace=True)

trace = go.Scatter(

    x= ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],#county_stats[county_stats['county']=='TRAVIS'].month.values,

    y=incidents_by_month.values,

)



layout = go.Layout(

    title='Monthly volume',

    xaxis=dict(

        title='Month',

        titlefont=dict(

            family='Courier New, monospace',

            size=18,

            color='#7f7f7f'

        )

    ),

    yaxis=dict(

        title='# of incidents',

        titlefont=dict(

            family='Courier New, monospace',

            size=18,

            color='#7f7f7f'

        )

    )

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

#offline.plot(fig, filename='austin-311-monthly-volume.html')

iplot(fig)
monthly_stats_by_dept = df.groupby(['owning_department','month']).size().reset_index().rename(columns={0:'count'})

departments = monthly_stats_by_dept.owning_department.unique()

monthly_stats_by_dept = monthly_stats_by_dept.set_index(['owning_department','month'])
full_index = pd.MultiIndex.from_product([tuple(departments), range(1,13)],names=['owning_department', 'month'])

full = monthly_stats_by_dept.reindex(full_index).reset_index().fillna(0)
data = []

for department in departments:

    trace = go.Scatter(

        x= ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],

        y=full[full['owning_department']==department]['count'].values,

        fill='tozeroy',

        name=department

    )

    data.append(trace)



layout = go.Layout(

    title='Monthly volume by department',

    xaxis=dict(

        title='Month',

        titlefont=dict(

            family='Courier New, monospace',

            size=18,

            color='#7f7f7f'

        )

    ),

    yaxis=dict(

        title='# of incidents',

        titlefont=dict(

            family='Courier New, monospace',

            size=18,

            color='#7f7f7f'

        )

    )

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
incidents_by_dept = ((df.owning_department.value_counts()/df.owning_department.count())*100)

labels = incidents_by_dept.index

values = incidents_by_dept.values



trace = go.Pie(labels=labels, values=values)

fig = go.Figure(data=[trace])

iplot(fig)
source = ((df.source.value_counts()/df.source.count())*100)[:10]

labels = source.index

values = source.values



trace = go.Pie(labels=labels, values=values)

fig = go.Figure(data=[trace])

iplot(fig)
closed_requests = df[df.status=='Closed']

dd = (closed_requests.close_date - closed_requests.created_date).dt.days.value_counts()

dd.sort_index(ascending=True, inplace=True)

dd = dd[dd.index>=0]

trace = go.Scatter(

    x = dd.index[:30],

    y = dd.values[:30],

    mode = 'lines+markers',

    name = 'days to close'

)



layout = go.Layout(

    title='Days to close',

    xaxis=dict(

        title='# of days to close',

        titlefont=dict(

            family='Courier New, monospace',

            size=18,

            color='#7f7f7f'

        )

    ),

    yaxis=dict(

        title='# of incidents',

        titlefont=dict(

            family='Courier New, monospace',

            size=18,

            color='#7f7f7f'

        )

    )

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

iplot(fig)