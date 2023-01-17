# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objs as go

import matplotlib.pyplot as plt

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Load data that we will use.

rus = pd.read_csv("../input/russian-demography/russian_demography.csv")



# Any results you write to the current directory are saved as output.
rus.info()
rus.head()
df = rus[rus.year == 1995].iloc[:,:]



# Creating trace1

trace1 = go.Scatter(

                    x = df.region,

                    y = df.urbanization,

                    mode = "lines",

                    name = "urbanization",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'))



# Creating trace2

trace2 = go.Scatter(

                    x = df.region,

                    y = df.gdw,

                    mode = "lines+markers",

                    name = "gdw",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)')

    )                

data = [trace1, trace2]

layout = dict(title = 'Urbanization VS GDW Ratios',xaxis= dict(title= 'Region name',ticklen= 5,zeroline= False))

fig = dict(data = data, layout = layout)

iplot(fig)
# prepare data frames

df2015 = rus[rus.year == 2015].iloc[:,:]

df2016 = rus[rus.year == 2016].iloc[:,:]

df2017 = rus[rus.year == 2017].iloc[:,:]

# import graph objects as "go"

import plotly.graph_objs as go

# creating trace1

trace1 =go.Scatter(

                    x = df2015.region,

                    y = df2015.urbanization,

                    mode = "markers",

                    name = "2015",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                   )

# creating trace2

trace2 =go.Scatter(

                    x = df2016.region,

                    y = df2016.urbanization,

                    mode = 'markers',

                    name = "2016",

                    marker = dict(color = 'rgba(255, 128, 2, 0.8)')

)

# creating trace3

trace3 =go.Scatter(x = df2017.region, y = df2017.urbanization, mode = "markers", name = "2017", marker = dict(color = 'rgba(0, 255, 200, 0.8)'))

data = [trace1, trace2, trace3]

layout = dict(title = 'urbanization rate depending on region in 2015, 2016, 2017',

              xaxis= dict(title= 'Region',ticklen = 5, zeroline = False),

              yaxis= dict(title= 'Urbanization',ticklen = 7, zeroline = False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
dfTy = rus[rus.region == 'Tyumen Oblast'].iloc[:29, :]

dfVg = rus[rus.region == 'Volgograd Oblast'].iloc[:29, :]

dfYs = rus[rus.region == 'Yaroslavl Oblast'].iloc[:29, :]



#creating trace1

tr1 = go.Scatter(

    x = dfTy.gdw, 

    y = dfTy.death_rate, 

    name = 'Tyumen Oblast', 

    mode = 'lines',   

    marker = dict(color = 'rgba(255, 128, 255, 0.8)')

)



#trace2

tr2 = go.Scatter(

    x = dfVg.gdw, 

    y = dfTy.death_rate, 

    name = 'Volgograd Oblast', 

    mode = 'lines',   

    marker = dict(color = 'rgba(235, 167, 3, 0.8)')

)



#and trace 3

tr3 = go.Scatter(

    x = dfYs.gdw, 

    y = dfYs.death_rate, 

    name = 'Yaroslavl Oblast', 

    mode = 'lines',   

    marker = dict(color = 'rgba(155, 228, 5, 0.8)')

)



data = [tr1, tr2, tr3]

layout = dict(title = 'death rate depending on gdw in Tyumen, Yaroslavl and Volgograd Oblasts.',

              xaxis= dict(title= 'gdw',ticklen = 5, zeroline = False),

              yaxis= dict(title= 'death rate',ticklen = 7, zeroline = False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
#I want to make this part with motion later



#Examining birth-death rates in 1999



df99 = rus[rus.year == 1999]





fig = go.Figure()





x = df99.region



trace1 = {

  'x': x,

  'y': df99.birth_rate,

  'name': 'birth rate',

  'type': 'bar'

};

trace2 = {'x': x,

          'y': df99.death_rate,

          'name':'death rate',

          'type': 'bar'

};

data = [trace1, trace2];

layout = {

  'xaxis': {'title': 'Top 3 universities'},

  'barmode': 'relative',

  'title': 'Birt and death rates ogf every oblast in 1999'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)

labels = rus.region

values = rus.urbanization

colors = [float(each) for each in rus.urbanization]



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=5,

                  marker=dict(line=dict( width=2)))

fig.show()



#examining urbanization of an oblast every year.



dfCr = rus[rus.year == 1995]

sizeCr  = dfCr.urbanization

sizeCr.dropna(inplace=True)



colorCr = dfCr.npg



for each in sizeCr:

    if each == 'nan':

        each.replace(each, '0')

        float(each) 



data = [

    {

        'y': dfCr.urbanization,

        'x': dfCr.npg,

        'mode': 'markers',

        'marker': {

            'color': colorCr ,

            'size': sizeCr ,

            'showscale': True

        },

        "text" :  dfCr.region   

    }

]

fig = go.Figure(data)



fig.show()

df2011 = rus.gdw[rus.year == 2011]

df2012 = rus.gdw[rus.year == 2012]

df2013 = rus.gdw[rus.year == 2013]



#trace1

tr1 = go.Histogram(

    x=df2011,

    opacity=0.75,

    name = "2011",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))



#trace2

tr2 = go.Histogram(

    x=df2012,

    opacity=0.75,

    name = "2012",

    marker=dict(color='rgba(171, 5, 96, 0.6)'))



#trace3

tr3 = go.Histogram(

    x=df2013,

    opacity=0.75,

    name = "2013",

    marker=dict(color='rgba(2, 50, 96, 0.6)'))



data = [tr1, tr2, tr3]



layout = go.Layout(barmode='overlay',

                   title=' general demographic weight in 2011, 2012 and 2013',

                   xaxis=dict(title='GDW'),

                   yaxis=dict( title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
# data prepararion

from wordcloud import WordCloud

x2011 = rus.region[rus.year == 2011]

plt.subplots(figsize=(15,15))

wordcloud = WordCloud(

                          background_color='blue',

                          width=512,

                          height=384

                         ).generate(" ".join(x2011))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
x2015 = rus[rus.region == 'Altai Krai']



import copy 

b = copy.copy(x2015.death_rate)

b = b/max(b)

trace0 = go.Box(

    y=b,

    name = 'death rates over years in Altai Krai',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)



a = copy.copy(x2015.gdw)

a = a/max(a)





trace1 = go.Box(

    y=a,

    name = 'gdw of Altai Krai in years',

    marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)

data = [trace0, trace1]

iplot(data)
# import figure factory

import plotly.figure_factory as ff

# prepare data

dataframe = rus[rus.year == 2015].iloc[:,:]

data2015 = dataframe.loc[:,["npg","gdw", "urbanization"]]

data2015["index"] = np.arange(1,len(data2015)+1)

# scatter matrix

fig = ff.create_scatterplotmatrix(data2015, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=700, width=700)

iplot(fig)
# create trace 1 that is 3d scatter

trace1 = go.Scatter3d(

    x=rus.npg,

    y=rus.urbanization,

    z=rus.gdw,

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