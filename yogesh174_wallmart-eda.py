import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

import datetime as dt

import warnings

warnings.filterwarnings('ignore')
os.chdir("/kaggle/input/wallmart")
df2 = pd.read_csv("Total_sales.csv")

df2 = df2.iloc[:,1:]

df2 = df2.fillna(0)

df2.head()
df2['Total'] = 0

for i in range(30490):

    i = df2.columns[i]

    df2['Total'] += df2[i]

df2['Total'].head()
fig = px.line(df2, x='date', y='Total', title='Wallmart Sales 2011-2016/10 stores',width=1200)

fig.update_xaxes(rangeslider_visible=True)

fig.show()
for i in range(30490):

    i = df2.columns[i]

    state = i.split('_')[3]

    if state not in df2.columns:

        df2[state] = 0

for i in range(30490):

    i = df2.columns[i]

    state = i.split('_')[3]

    df2[state] += df2[i]

df2.head()
fig = go.Figure()

fig.add_trace(go.Scatter(x=df2['date'], y=df2['CA'].values,

                    mode='lines',

                    name='CA'))

fig.add_trace(go.Scatter(x=df2['date'], y=df2['TX'].values,

                    mode='lines',

                    name='TX'))

fig.add_trace(go.Scatter(x=df2['date'], y=df2['WI'].values,

                    mode='lines',

                    name='WI'))

fig.update_layout(

    autosize=False,

    width=1000,

    height=700,

    margin=dict(

        l=50,

        r=50,

        b=100,

        t=100,

        pad=4

    ),

    paper_bgcolor="LightSteelBlue",

    title="Walmart statewise sales",

    xaxis_title="Date",

    yaxis_title="Sales",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#042a30"

    )

)



fig.update_xaxes(rangeslider_visible=True,)

fig.show()
for i in range(30490):

    i = df2.columns[i]

    category = i.split('_')[0]

    if category not in df2.columns:

        df2[category] = 0

for i in range(30490):

    i = df2.columns[i]

    category = i.split('_')[0]

    df2[category] += df2[i]

df2.head()
import plotly.graph_objects as go



fig = go.Figure()



fig.add_trace(go.Scatter(x=df2['date'], y=df2['FOODS'].values,

                    mode='lines',

                    name='FOODS'))

fig.add_trace(go.Scatter(x=df2['date'], y=df2['HOBBIES'].values,

                    mode='lines',

                    name='HOBBIES'))

fig.add_trace(go.Scatter(x=df2['date'], y=df2['HOUSEHOLD'].values,

                    mode='lines',

                    name='HOUSEHOLD'))

fig.update_layout(

    autosize=False,

    width=1000,

    height=700,

    margin=dict(

        l=50,

        r=50,

        b=100,

        t=100,

        pad=4

    ),

    paper_bgcolor="LightSteelBlue",

    title="Walmart category wise sales",

    xaxis_title="Date",

    yaxis_title="Sales",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#042a30"

    )

)





fig.update_xaxes(rangeslider_visible=True)

fig.show()
df2.columns
for i in range(30490):

    i = df2.columns[i]

    store = i.split('_')[3] + '_' + i.split('_')[4]

    if store not in df2.columns:

        df2[store] = 0

for i in range(30490):

    i = df2.columns[i]

    store = i.split('_')[3] + '_' + i.split('_')[4]

    df2[store] += df2[i]

df2.head()
for i in range(30490):

    i = df2.columns[i]

    item = i.split('_')[3] + '_' + i.split('_')[4] + '_' + i.split('_')[0]

    if item not in df2.columns:

        df2[item] = 0

for i in range(30490):

    i = df2.columns[i]

    item = i.split('_')[3] + '_' + i.split('_')[4] + '_' + i.split('_')[0]

    df2[item] += df2[i]

df2.head()
import plotly.graph_objects as go



fig = go.Figure()



for i in range(30498,30502): 

    i = df2.columns[i]

    fig.add_trace(go.Scatter(x=df2['date'], y=df2[i].values,

                        mode='lines',

                        name=i.split('_')[0] + ' store ' +i.split('_')[1]))

    

fig.update_layout(

    autosize=False,

    width=1000,

    height=700,

    margin=dict(

        l=50,

        r=50,

        b=100,

        t=100,

        pad=4

    ),

    paper_bgcolor="LightSteelBlue",

    title="Walmart California store wise sales",

    xaxis_title="Date",

    yaxis_title="Sales",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#042a30"

    )

)



fig.update_xaxes(rangeslider_visible=True)

fig.show()
import plotly.graph_objects as go



fig = go.Figure()



for i in range(30510,30531,10): 

    i = df2.columns[i]

    fig.add_trace(go.Scatter(x=df2['date'], y=df2[i].values,

                        mode='lines',

                        name=i.split('_')[2]))

    

fig.update_layout(

    autosize=False,

    width=1000,

    height=700,

    margin=dict(

        l=50,

        r=50,

        b=100,

        t=100,

        pad=4

    ),

    paper_bgcolor="LightSteelBlue",

    title="Walmart California store 3 category wise sales",

    xaxis_title="Date",

    yaxis_title="Sales",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#042a30"

    )

)





fig.update_xaxes(rangeslider_visible=True)

fig.show()
cal = pd.read_csv('calendar.csv')

events = cal[['date','event_name_1','event_type_1','event_name_2','event_type_2']]

events = events.fillna(0)

events = events[(events['event_name_1'] != 0) | (events['event_name_2'] != 0)]

events.shape
l = []

c = 0

for x in events['date'].values:

    c +=1

    l.append(

    dict(

        type="line",

        yref='paper',

        y0=0,

        y1=1,

        xref='x1',

        x0=x,

        x1=x,

        line=dict(

            color="Red",

            width=2,

            dash="dashdot",

    )))

print(c)

fig = px.line(df2, x='date', y='CA_3')

fig.update_layout(shapes=l)

fig.show()


events[events['date'] == dt.datetime(2015, 4 , 12)]

events[(events['date'].apply(lambda a : dt.datetime.strptime(a, "%Y-%m-%d").month) == 5) | (events['date'].apply(lambda a : dt.datetime.strptime(a, "%Y-%m-%d").month) == 6)]
df2[['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2','snap_CA', 'snap_TX', 'snap_WI']] = cal[['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2','snap_CA', 'snap_TX', 'snap_WI']]

df2[['date','event_name_1', 'event_type_1', 'event_name_2', 'event_type_2','snap_CA', 'snap_TX', 'snap_WI']]

dis = pd.read_csv('us_disasters_m5.csv')

dis.head()
dis_ca = dis[dis['state'] == 'CA']

print(dis_ca.shape)

dis_ca.head()
dis_ca['declaration_date'] = pd.to_datetime(dis_ca['declaration_date'].apply(lambda x : x[:10]))

dis_ca['declaration_date'].head()
dis_ca_timeline = dis_ca[['incident_type','declaration_date']]

dis_ca_timeline['declaration_date'] = pd.to_datetime(dis_ca_timeline['declaration_date'])

dis_ca_timeline['declaration_date'] = dis_ca_timeline['declaration_date'].apply(lambda x : x.strftime("%Y-%m-%d"))


dis_ca_timeline = dis_ca_timeline.reset_index()

dis_ca_timeline = dis_ca_timeline.drop(columns='index') 

dis_ca_timeline.head()
dis_ca_timeline.head()
df2[['date','Total']].set_index('date').head()
l = []

for i in range(54):

    x = dis_ca_timeline['declaration_date'].iloc[i]

    l.append(

    dict(

        type="line",

        yref='paper',

        y0=0,

        y1=1,

        xref='x1',

        x0=x,

        x1=x,

        line=dict(

            color="Red",

            width=2,

            dash="dashdot",

    )))

fig = px.line(df2, x='date', y='Total')



i=40

x = dis_ca_timeline['declaration_date'].iloc[i]

fig.update_layout(shapes=l)    
dis_ca_timeline['declaration_date'].unique()
dis_ca_timeline['Threat level'] = 'Minor'



d = dis_ca_timeline['incident_type'] == 'Tsunami'



dis_ca_timeline.loc[d.values,'Threat level'] = 'Major'
dis_ca_timeline[dis_ca_timeline['incident_type'] != 'Fire']
dis[dis['state'] == 'CA']

dis.iloc[327:329,:]
dis_ca_timeline1 = dis_ca[['incident_type','incident_begin_date']]

dis_ca_timeline1['incident_begin_date'] = pd.to_datetime(dis_ca_timeline1['incident_begin_date'])

dis_ca_timeline1['incident_begin_date'] = dis_ca_timeline1['incident_begin_date'].apply(lambda x : x.strftime("%Y-%m-%d"))

dis_ca_timeline1 = dis_ca_timeline1.reset_index()

dis_ca_timeline1 = dis_ca_timeline1.drop(columns='index') 

dis_ca_timeline1.set_index('incident_begin_date')

#dis_ca_timeline1 = dis_ca_timeline1[dis_ca_timeline1['incident_type'] != 'Fire']

dis_ca_timeline1.head()
dis_ca_timeline2 = pd.read_excel('dis_ca_timeline1_modified.xlsx')

dis_ca_timeline2[dis_ca_timeline2['Threat level'] == 'Medium']['incident_begin_date'].values

dis_ca_timeline2.head()
l = []

for x in dis_ca_timeline2[dis_ca_timeline2['Threat level'] == 'Medium']['incident_begin_date'].values:

    l.append(

    dict(

        type="line",

        yref='paper',

        y0=0,

        y1=1,

        xref='x1',

        x0=x,

        x1=x,

        line=dict(

            color="Red",

            width=2,

            dash="dashdot",

    )))

    

fig = px.line(df2, x='date', y='CA_3')

fig.update_layout(shapes=l)

fig.show()