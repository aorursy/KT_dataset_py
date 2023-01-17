# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from wordcloud import WordCloud

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
timesData = pd.read_csv("/kaggle/input/fifa19/data.csv")
timesData.info()
timesData.head(10)
new_data = timesData[['Name','Club','Release Clause','Overall','Age']].copy()

new_data.dropna(inplace = True)

Club = list(new_data['Club'].unique())

Release = list(new_data['Release Clause'])

m = [ float(i[1:-1]) if 'M' == i[-1] else int(i[1:-1])/1000  for i in Release]

new_data['Release Clause'] = m
df = new_data.iloc[:100,:]



import plotly.graph_objs as go



trace1 = go.Scatter(

                    x = df.Overall,

                    y = df.Age,

                    mode = "lines",

                    name = "Age",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= df.Name)

trace2 = go.Scatter(

                    x = df.Overall,

                    y = df['Release Clause'],

                    mode = "lines+markers",

                    name = "Release Clause",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= df.Name)

data = [trace1, trace2]

layout = dict(title = 'Release Clause and Age vs Overall of Top 100 Footballer',

              xaxis= dict(title= 'Overall',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
dfBarcelona = new_data[new_data.Club == 'FC Barcelona']

dfJuventus = new_data[new_data.Club == 'Juventus']

dfPSG = new_data[new_data.Club == 'Paris Saint-Germain']



trace1 = go.Scatter(

                    x = dfBarcelona.Overall,

                    y = dfBarcelona['Release Clause'],

                    mode = "markers",

                    name = "FC Barcelona",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                    text= dfBarcelona.Name)

trace2 = go.Scatter(

                    x = dfJuventus.Overall,

                    y = dfJuventus['Release Clause'],

                    mode = "markers",

                    name = "Juventus",

                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),

                    text= dfJuventus.Name)

trace3 = go.Scatter(

                    x = dfPSG.Overall,

                    y = dfPSG['Release Clause'],

                    mode = "markers",

                    name = "Paris Saint-Germain",

                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),

                    text= dfPSG.Name)



data = [trace1, trace2, trace3]

layout = dict(title = 'FC Barcelona vs Juventus vs Paris Saint-Germain ',

              xaxis= dict(title= 'Overall',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Release Clause',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
dfBarcelona = new_data[new_data.Club == 'FC Barcelona'].iloc[:3,:]

trace1 = go.Bar(

                x = dfBarcelona.Name,

                y = dfBarcelona['Release Clause'],

                name = "Release Clause",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = dfBarcelona.Club)

trace2 = go.Bar(

                x = dfBarcelona.Name,

                y = dfBarcelona.Overall,

                name = "Overall",

                marker = dict(color = 'rgba(255, 255, 128, 0.5)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = dfBarcelona.Club)

data = [trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
dfBarcelona = new_data[new_data.Club == 'FC Barcelona'].iloc[:3,:]



x = dfBarcelona.Name



trace1 = {

  'x': x,

  'y': dfBarcelona['Release Clause'],

  'name': 'Release Clause',

  'type': 'bar'

};

trace2 = {

  'x': x,

  'y': dfBarcelona.Overall,

  'name': 'Overall',

  'type': 'bar'

};

data = [trace1, trace2];

layout = {

  'xaxis': {'title': 'Top 3 Footballer'},

  'barmode': 'relative',

  'title': 'Release Clause and Overall of top 3 Footballer in Barcelona'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)
from plotly import tools

dfJuventus = new_data[new_data.Club == 'Juventus'].iloc[:7,:]



y_saving = [each for each in dfJuventus.Overall]

y_net_worth  = [each for each in dfJuventus.Age]

x_saving = [each for each in dfJuventus.Name]

x_net_worth  = [each for each in dfJuventus.Name]



trace0 = go.Bar(

                x=y_saving,

                y=x_saving,

                marker=dict(color='rgba(171, 50, 96, 0.6)',line=dict(color='rgba(171, 50, 96, 1.0)',width=1)),

                name='Overall',

                orientation='h',

)

trace1 = go.Scatter(

                x=y_net_worth,

                y=x_net_worth,

                mode='lines+markers',

                line=dict(color='rgb(63, 72, 204)'),

                name='Age',

)



layout = dict(

                title='Overall and Age',

                yaxis=dict(showticklabels=True,domain=[0, 0.85]),

                yaxis2=dict(showline=True,showticklabels=False,linecolor='rgba(102, 102, 102, 0.8)',linewidth=2,domain=[0, 0.85]),

                xaxis=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0, 0.42]),

                xaxis2=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0.47, 1],side='top',dtick=25),

                legend=dict(x=0.029,y=1.038,font=dict(size=10) ),

                margin=dict(l=200, r=20,t=70,b=70),

                paper_bgcolor='rgb(248, 248, 255)',

                plot_bgcolor='rgb(248, 248, 255)',

                )



annotations = []

y_s = np.round(y_saving, decimals=2)

y_nw = np.rint(y_net_worth)

# Adding labels

for ydn, yd, xd in zip(y_nw, y_s, x_saving):

    # labeling the scatter savings

    annotations.append(dict(xref='x2', yref='y2', y=xd, x=ydn - 4,text='{:,}'.format(ydn),font=dict(family='Arial', size=12,color='rgb(63, 72, 204)'),showarrow=False))

    # labeling the bar net worth

    annotations.append(dict(xref='x1', yref='y1', y=xd, x=yd + 3,text=str(yd),font=dict(family='Arial', size=12,color='rgb(171, 50, 96)'),showarrow=False))



layout['annotations'] = annotations



fig = tools.make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,

                          shared_yaxes=False, vertical_spacing=0.001)



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)



fig['layout'].update(layout)

iplot(fig)
nat_ovr = timesData[['Nationality','Overall']].copy()

nat_ovr.dropna(inplace=True)

England = [sum(nat_ovr[nat_ovr.Nationality == 'England'].Overall)/len(nat_ovr[nat_ovr.Nationality == 'England'].Overall)]

Germany = [sum(nat_ovr[nat_ovr.Nationality == 'Germany'].Overall)/len(nat_ovr[nat_ovr.Nationality == 'Germany'].Overall)]

Spain = [sum(nat_ovr[nat_ovr.Nationality == 'Spain'].Overall)/len(nat_ovr[nat_ovr.Nationality == 'Spain'].Overall)]

Argentina = [sum(nat_ovr[nat_ovr.Nationality == 'Argentina'].Overall)/len(nat_ovr[nat_ovr.Nationality == 'Argentina'].Overall)]

France = [sum(nat_ovr[nat_ovr.Nationality == 'France'].Overall)/len(nat_ovr[nat_ovr.Nationality == 'France'].Overall)]

pie1_list = [*England,*Germany,*Spain,*Argentina,*France]

labels = ['England' , 'Germany','Spain','Argentina','France']

fig = {

  "data": [

    {

      "values": pie1_list,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": "ortalama derece",

      "hoverinfo":"label+percent+name",

      "hole": .3,

      "type": "pie"

    },],

  "layout": {

        "title":"ulkelerin ortalama genel dereceleri",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "genel dereceler",

                "x": 0.20,

                "y": 1

            },

        ]

    }

}



iplot(fig)
new_data['Potential'] = timesData['Potential']



df = new_data.iloc[:50:]

footballer_size  = new_data.Age

Value_color = new_data['Release Clause']

data = [

    {

        'y': df.Potential,

        'x': df.Overall,

        'mode': 'markers',

        'marker': {

            'color': Value_color,

            'size': footballer_size,

            'showscale': True

        },

        "text" :  df.Name    

    }

]

iplot(data)
trace1 = go.Scatter3d(

    x=new_data.Overall,

    y=new_data.Age,

    z=new_data['Release Clause'],

    mode='markers',

    marker=dict(

        size=10,

        color='rgb(255,0,0)',                # set color to an array/list of desired values      

    )

)



data = [trace1]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    )

    

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
timesData.columns
timesData.Position
ST = timesData[timesData.Position == 'ST'].Overall

GK = timesData[timesData.Position == 'GK'].Overall



trace1 = go.Histogram(

    x=ST,

    opacity=0.75,

    name = "ST",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))

trace2 = go.Histogram(

    x=GK,

    opacity=0.75,

    name = "GK",

    marker=dict(color='rgba(12, 50, 196, 0.6)'))



data = [trace1, trace2]

layout = go.Layout(barmode='overlay',

                   title=' ST and GK',

                   xaxis=dict(title='Overall'),

                   yaxis=dict( title='Frequency'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
Nationality = list(timesData.Nationality)



plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(Nationality))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
RM = timesData[timesData.Position == 'RM']

LM = timesData[timesData.Position == 'LM']



trace0 = go.Box(

    y=RM.Overall,

    name = 'RM',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)

trace1 = go.Box(

    y=LM.Overall,

    name = 'LM',

    marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)

data = [trace0, trace1]

iplot(data)
PSG = timesData[timesData.Club == 'Paris Saint-Germain']

GS = timesData[timesData.Club == 'Galatasaray SK']



trace1 = go.Box(

    y = GS.Overall,

    name = 'GS',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)

trace2 = go.Box(

    y = PSG.Overall,

    name = 'PSG',

    marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)

data = [trace1, trace2]

iplot(data)
import plotly.figure_factory as ff

# prepare data

data = new_data[["Release Clause","Overall", "Age"]].loc[:400]

data["index"] = np.arange(1,len(data)+1)

# scatter matrix

fig = ff.create_scatterplotmatrix(data, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=700, width=700)

iplot(fig)
new_data[['ShotPower','Position']] = timesData[['ShotPower','Position']]

dataframe = new_data[new_data.Position == 'ST'].loc[:400]

trace1 = go.Scatter(

    x=dataframe.Overall,

    y=dataframe['Release Clause'],

    name = "Release Clause",

    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

)

# second line plot

trace2 = go.Scatter(

    x=dataframe.Overall,

    y=dataframe.ShotPower,

    xaxis='x2',

    yaxis='y2',

    name = "ShotPower",

    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),

)

data = [trace1, trace2]

layout = go.Layout(

    xaxis2=dict(

        domain=[0.6, 0.95],

        anchor='y2',        

    ),

    yaxis2=dict(

        domain=[0.6, 0.95],

        anchor='x2',

    ),

    title = 'ShotPower and Release Clause vs Overall of Santrafor'



)



fig = go.Figure(data=data, layout=layout)

iplot(fig)

trace1 = go.Scatter(

    x=dataframe.Overall,

    y=dataframe['Release Clause'],

    name = "Release Clause"

)

trace2 = go.Scatter(

    x=dataframe.Overall,

    y=dataframe.Age,

    xaxis='x2',

    yaxis='y2',

    name = "Age"

)

trace3 = go.Scatter(

    x=dataframe.Overall,

    y=dataframe.Potential,

    xaxis='x3',

    yaxis='y3',

    name = "Potential"

)

trace4 = go.Scatter(

    x=dataframe.Overall,

    y=dataframe.ShotPower,

    xaxis='x4',

    yaxis='y4',

    name = "ShotPower"

)

data = [trace1, trace2, trace3, trace4]

layout = go.Layout(

    xaxis=dict(

        domain=[0, 0.45]

    ),

    yaxis=dict(

        domain=[0, 0.45]

    ),

    xaxis2=dict(

        domain=[0.55, 1]

    ),

    xaxis3=dict(

        domain=[0, 0.45],

        anchor='y3'

    ),

    xaxis4=dict(

        domain=[0.55, 1],

        anchor='y4'

    ),

    yaxis2=dict(

        domain=[0, 0.45],

        anchor='x2'

    ),

    yaxis3=dict(

        domain=[0.55, 1]

    ),

    yaxis4=dict(

        domain=[0.55, 1],

        anchor='x4'

    ),

    title = 'Release Clause, Age, Potential and Shot Power VS Overall of Santrafor'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)