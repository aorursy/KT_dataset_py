import pandas as pd
import numpy as np
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import *
import matplotlib.pyplot as plt
import seaborn as sns   
from random import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
df=pd.read_csv('../input/DJ_Mag.csv')
df.drop(df.columns[0],axis=1,inplace=True)
df=df.dropna()
df=df.sort_values(by="Year" , ascending=True)
df=df.reset_index(inplace=False)
df.drop(df.columns[0],axis=1,inplace=True) # Data cleaning and sorting finished
## Line Plot###########################################################################
favdjs='Diplo','Skrillex','Major Lazer','Jack U','Calvin Harris','Avicii','Flume','Daft Punk'
names=[]
for i in range(len(favdjs)):
    name='trace'+str(i)
    #print(name)
    names.append(name)
    names[i]=go.Scatter(
        x = df['Year'][df['DJ']==favdjs[i]],
        y = df['Rank'][df['DJ']==favdjs[i]],
        name=favdjs[i]
        )
layout=dict(title='Ranking Changes of My Favorite Producers',
            xaxis=dict(title='Year',zeroline=True),
            yaxis=dict(title='Ranking',range=[100,0]),
            )
fig=dict(data=names,layout=layout)
plotly.offline.iplot(fig, filename='line_plot.html')
## Bar Charts
re=[] # Re-entry
ne=[] # New-entry
nc=[] # No change
names=[] # Clear the trace name list
for i in range(2010,2018):
    df0=list(df['Change'][df['Year']==i])
    re.append(df0.count('Re-entry'))
    ne.append(df0.count('New Entry'))
    nc.append(df0.count('No change'))
cates=['Re-entry','New Entry','No Change']
cnt=[re,ne,nc]
for i in range(0,3):    
    name='trace'+str(i)
    names.append(name)
    names[i]=go.Bar(
        x = ['2010','2011','2012','2013','2014','2015','2016','2017'],
        y = cnt[i],
        name=cates[i]
        )
layout=dict(title='Numbers of Re-entry, New Entry and No change',
            xaxis=dict(title='Year',zeroline=True),
            yaxis=dict(title='Number'),
            )
fig=dict(data=names,layout=layout)
plotly.offline.iplot(fig, filename='bar_charts.html')
## Bubble Plot###########################################################################
n=df['DJ'].unique()
msize=[]
x0=[]
y0=[]
txt=[]
msize0=[]
for i in n:
    t=list(df['DJ']).count(i)
    msize0.append(t)
    msize.append(t*t/3)
    x0.append(np.random.randint(1, 100))
    y0.append(np.random.randint(1, 100))
    txt0=str(i)+'<br>Times:'+str(t)
    txt.append(txt0)

df1=pd.DataFrame()
df1['Name']=n
df1['Times']=msize0
times=[]
for x in range(len(df['DJ'])): 
    m=df1.loc[df1['Name']== df['DJ'][x],'Times'].iloc[0]
    times.append(m)
df['Time']=times # New feature assed: times appeared on the chart

trace0 = go.Scatter(
    x=x0,
    y=y0,
    mode='markers',
    hoverinfo = 'text',
    text=txt,
    marker=dict(  
        size=msize,
        color=msize,
        showscale=True
    )
)
layout=dict(title='Number of Times Entering the Chart',
            xaxis=dict(autorange=True,
                       showgrid=False,
                       zeroline=False,
                       showline=False,
                       ticks='',
                       showticklabels=False),
            yaxis=dict(autorange=True,
                       showgrid=False,
                       zeroline=False,
                       showline=False,
                       ticks='',
                       showticklabels=False)
            )
fig=dict(data=[trace0],layout=layout)
plotly.offline.iplot(fig, filename='bubblechart-size')
## Pie Plot###########################################################################
rank1=df['DJ'][df['Rank']==1]
label1=rank1.unique()
label1=list(label1)
rank1=list(rank1)
value1=[]
for i in label1:
    value1.append(rank1.count(i))

rank2=df['DJ'][df['Rank']==2]
label2=rank2.unique()
label2=list(label2)
rank2=list(rank2)
value2=[]
for i in label2:
    value2.append(rank2.count(i))

rank3=df['DJ'][df['Rank']==3]
label3=rank3.unique()
label3=list(label3)
rank3=list(rank3)
value3=[]
for i in label3:
    value3.append(rank3.count(i))

fig = {
  "data": [
    {
      "values": value1,
      "labels": label1,
      "domain": {"x": [0, .33]},
      "hoverinfo":"label+value",
      "hole": .4,
      "type": "pie"
    },
    {
      "values": value2,
      "labels": label2,
      "textposition":"inside",
      "domain": {"x": [.34, .66]},
      "hoverinfo":"label+value",
      "hole": .4,
      "type": "pie"
    },
    {
      "values": value3,
      "labels": label3,
      "domain": {"x": [.67, 1]},
      "hoverinfo":"label+value",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Who Wins Most?",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "No.1",
                "x": 0.13,
                "y": 0
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "N0.2",
                "x": 0.51,
                "y": 0
            },
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "No.3",
                "x": 0.87,
                "y": 0
            }
        ]
    }
}
plotly.offline.iplot(fig, filename='donut')