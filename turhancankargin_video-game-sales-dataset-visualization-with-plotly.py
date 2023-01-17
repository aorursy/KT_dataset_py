import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotly

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Load data that we will use.

vgsales = pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")

vgsales.head()
# As you can see from info method. There are 16598.

# However, Year has 16327 entries. That means Year has NAN value.

# Also Year should be integer but it is given as float. Therefore we will convert it.

# In addition, publisher has NAN values.

vgsales.info()
# Lets start with dropping nan values

vgsales.dropna(how="any",inplace = True)

# Then convert data from float to int

vgsales.Year = vgsales.Year.astype(int)

vgsales.info()
# prepare data frame

df = vgsales.iloc[:100,:]



# import graph objects as "go"

import plotly.graph_objs as go



# Creating trace1

trace1 = go.Scatter(

                    x = df.Rank,

                    y = df.NA_Sales,

                    mode = "lines",

                    name = "NA_Sales",

                    marker = dict(color = 'rgba(69, 40, 202, 0.8)'),

                    text= df.Name)

# Creating trace2

trace2 = go.Scatter(

                    x = df.Rank,

                    y = df.EU_Sales,

                    mode = "lines+markers",

                    name = "EU_Sales",

                    marker = dict(color = 'rgba(227, 134, 211, 0.8)'),

                    text= df.Name)

data = [trace1, trace2]

layout = dict(title = 'Sales in North America (in millions) and Sales in Europe (in millions) vs Rank of Top 100 Video Games',

              xaxis= dict(title= 'World Rank',ticklen= 9,zeroline= True)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
# prepare data frames

df2010 = vgsales[vgsales.Year == 2010].iloc[:100,:]

df2011 = vgsales[vgsales.Year == 2011].iloc[:100,:]

df2012 = vgsales[vgsales.Year == 2012].iloc[:100,:]

# import graph objects as "go"

import plotly.graph_objs as go

# creating trace1

trace1 =go.Scatter(

                    x = df2010.Rank,

                    y = df2010.Global_Sales,

                    mode = "markers",

                    name = "2010",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                    text= df2010.Name)

# creating trace2

trace2 =go.Scatter(

                    x = df2011.Rank,

                    y = df2011.Global_Sales,

                    mode = "markers",

                    name = "2011",

                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),

                    text= df2011.Name)

# creating trace3

trace3 =go.Scatter(

                    x = df2012.Rank,

                    y = df2012.Global_Sales,

                    mode = "markers",

                    name = "2012",

                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),

                    text= df2012.Name)

data = [trace1, trace2, trace3]

layout = dict(title = 'Total Worldwide Sales (in millions) vs World Rank of top 100 Games with 2010, 2011 and 2012 years',

              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Worldwide Sales',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
# prepare data frames

df2015 = vgsales[vgsales.Year == 2015].iloc[:3,:]

df2015
# import graph objects as "go"

import plotly.graph_objs as go

# create trace1 

trace1 = go.Bar(

                x = df2015.Name,

                y = df2015.EU_Sales,

                name = "Sales in Europe",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df2015.Platform)

# create trace2 

trace2 = go.Bar(

                x = df2015.Name,

                y = df2015.JP_Sales,

                name = "Sales in Japan",

                marker = dict(color = 'rgba(255, 255, 128, 0.5)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df2015.Platform)

data = [trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# data preparation

df2014 = vgsales[vgsales.Year == 2014].iloc[:5,:]

pie1 = df2014.Global_Sales

labels = df2014.Name

# figure

fig = {

  "data": [

    {

      "values": pie1,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": "Toatal Sales Rates",

      "hoverinfo":"label+percent+name",

      "hole": .3,

      "type": "pie"

    },],

  "layout": {

        "title":"Video Games Total Sales Rates",

        "annotations": [

            { "font": { "size": 1},

              "showarrow": False,

              "text": "Total Sales",

                "x": 0.20,

                "y": 1

            },

        ]

    }

}

iplot(fig)
# data preparation

df2015 = vgsales[vgsales.Year == 2015].iloc[:20,:]



data = [

    {

        'y': df2015.NA_Sales,

        'x': df2015.Rank,

        'mode': 'markers',

        'marker': {

            'color': df2015.EU_Sales,

            'size': df2015.Global_Sales,

            'showscale': True

        },

        "text" :  df2015.Name    

    }

]

iplot(data)
# prepare data

x2011 = vgsales[vgsales.Year == 2011].iloc[:200,:]

x2012 = vgsales[vgsales.Year == 2012].iloc[:200,:]

x2011=x2011.Global_Sales

x2012=x2012.Global_Sales



trace1 = go.Histogram(

    x=x2011,

    opacity=0.75,

    name = "2011",

    marker=dict(color='rgba(0, 250, 0, 0.6)'))

trace2 = go.Histogram(

    x=x2012,

    opacity=0.75,

    name = "2012",

    marker=dict(color='rgba(190, 0, 150, 0.6)'))



data = [trace1, trace2]

layout = go.Layout(barmode='overlay',

                   title=' Global Sales in 2011 and 2012',

                   xaxis=dict(title='Global Sales'),

                   yaxis=dict( title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
# data prepararion

x2011 = vgsales.Name[vgsales.Year == 2011]

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(x2011))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
# data preparation

x2015 = vgsales[vgsales.Year == 2015].iloc[:100,:] # For first 100 Video Games



trace0 = go.Box(

    y=x2015.Global_Sales,

    name = 'Global Sales of  Video Games in 2015',

    marker = dict(

        color = 'rgb(255, 0, 0)',

    )

)

trace1 = go.Box(

    y=x2015.EU_Sales,

    name = 'Europe Sales of  Video Games in 2015',

    marker = dict(

        color = 'rgb(0, 0, 255)',

    )

)

data = [trace0, trace1]

iplot(data)
# import figure factory

import plotly.figure_factory as ff

# prepare data

dataframe = vgsales[vgsales.Year == 2015]

data2015 = dataframe.loc[:,["NA_Sales","EU_Sales", "JP_Sales"]]

data2015["index"] = np.arange(1,len(data2015)+1)

# scatter matrix

fig = ff.create_scatterplotmatrix(data2015, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=700, width=700)

iplot(fig)
# first line plot

trace1 = go.Scatter(

    x=dataframe.Rank,

    y=dataframe.Global_Sales,

    name = "Global_Sales",

    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

)

# second line plot

trace2 = go.Scatter(

    x=dataframe.Rank,

    y=dataframe.EU_Sales,

    xaxis='x2',

    yaxis='y2',

    name = "EU_Sales",

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

    title = 'EU_Sales and Global_Sales vs Rank of Video Games'



)



fig = go.Figure(data=data, layout=layout)

iplot(fig)

# create trace 1 that is 3d scatter

trace1 = go.Scatter3d(

    x=dataframe.Rank,

    y=dataframe.Other_Sales,

    z=dataframe.JP_Sales,

    mode='markers',

    marker=dict(

        size=10,

        color='rgb(0,0,255)',                # set color to an array/list of desired values      

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
trace1 = go.Scatter(

    x=dataframe.Rank,

    y=dataframe.Global_Sales,

    name = "Global_Sales"

)

trace2 = go.Scatter(

    x=dataframe.Rank,

    y=dataframe.EU_Sales,

    xaxis='x2',

    yaxis='y2',

    name = "EU_Sales"

)

trace3 = go.Scatter(

    x=dataframe.Rank,

    y=dataframe.JP_Sales,

    xaxis='x3',

    yaxis='y3',

    name = "JP_Sales"

)

trace4 = go.Scatter(

    x=dataframe.Rank,

    y=dataframe.NA_Sales,

    xaxis='x4',

    yaxis='y4',

    name = "NA_Sales"

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

    title = 'NA_Sales, EU_Sales, JP_Sales and Global_Sales VS World Rank of Video Games'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)