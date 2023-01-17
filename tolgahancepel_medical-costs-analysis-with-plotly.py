import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
#os
import os
print(os.listdir("../input"))

df = pd.read_csv("../input/insurance.csv")
df.head()
df.info()
df.region.unique()
gender_list = [df[df.sex == "female"].sex.value_counts().tolist(), df[df.sex == "male"].sex.value_counts().tolist()]
gender_list = [gender_list[0][0], gender_list[1][0]]
gender_list
labels = ["Female", "Male"]
values = gender_list
colors = ['#FEBFB3', '#b3c8fe']
trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='percent', 
               textfont=dict(size=20),
               marker=dict(colors=colors,
                           line=dict(color='#000000', width=2)))
data = [trace]
layout = go.Layout(title='Rate of Males & Females')
fig = go.Figure(data = data, layout = layout)
iplot(fig)
dict_regions= {'low' : df[df.bmi < 18.5].charges.mean(),
               'normal' : df[(df.bmi > 18.5) & (df.bmi < 24.9)].charges.mean(),
               'high' : df[df.bmi > 24.9].charges.mean(),
             }
df_bmi = pd.DataFrame.from_dict(dict_regions, orient='index')
df_bmi.reset_index(inplace=True)
df_bmi.columns = ['bmi', 'mean_value']
df_bmi
my_color = ['rgb(254,224,39)','rgb(102,189,99)','rgb(215,48,39)']
trace=go.Bar(
            x=df_bmi.bmi,
            y=df_bmi.mean_value,
            text="Mean Medical Costs",
            marker=dict(
                color=my_color,
                line=dict(
                color=my_color,
                width=1.5),
            ),
            opacity=0.7)

data = [trace]
layout = go.Layout(title = 'Body Mass Index Means',
              xaxis = dict(title = 'BMI'),
              yaxis = dict(title = 'Mean Charges'))
fig = go.Figure(data = data, layout = layout)
iplot(fig)
trace0 = go.Box(
    y=df[df.sex == "female"].bmi,
    name = 'Female',
    marker = dict(
        color = 'rgb(158, 1, 66)',
    )
)
trace1 = go.Box(
    y=df[df.sex == "male"].bmi,
    name = 'Male',
    marker = dict(
        color = 'rgb(50, 136, 189)',
    )
)
layout = go.Layout(title ='BMI of Females and Males',
              xaxis = dict(title = 'Gender'),
              yaxis = dict(title = 'BMI'))
data = [trace0, trace1]
fig = go.Figure(data = data, layout = layout)
iplot(fig)
charges_sorted = df.copy()
sort_index = (df['charges'].sort_values(ascending=False)).index.values
charges_sorted = df.reindex(sort_index)
charges_sorted.reset_index(inplace=True)
#charges_sorted = charges_sorted.head(250)
charges_sorted.head()
# bmi values above-below
trace0 = go.Scatter(
    x = charges_sorted.head(250).charges,
    y = charges_sorted.head(250).bmi[charges_sorted.head(250).bmi < 18.5],
    name = 'Low',
    mode = 'markers',
    marker = dict(
        size = 10,
        color = 'rgb(254,224,39)',
        line = dict(
            width = 1,
            color = 'rgb(0, 0, 0)'
        )
    )
)

trace1 = go.Scatter(
    x = charges_sorted.head(250).charges,
    y = charges_sorted.head(250).bmi[(charges_sorted.head(250).bmi > 18.5) & (charges_sorted.head(250).bmi < 24.9)],
    name = 'Normal',
    mode = 'markers',
    marker = dict(
        size = 10,
        color = 'rgb(102,189,99)',
        line = dict(
            width = 1,
        )
    )
)

trace2 = go.Scatter(
    y = charges_sorted.head(250).bmi[charges_sorted.head(250).bmi > 24.9],
    x = charges_sorted.head(250).charges,
    name = 'High',
    mode = 'markers',
    marker = dict(
        size = 10,
        color = 'rgb(215,48,39)',
        line = dict(
            width = 1,
        )
    )
)
data = [trace0, trace1,trace2]
layout = dict(title = 'BMI of the Most 250 Medical Costs',
              yaxis = dict(zeroline = False,title = "BMI"),
              xaxis = dict(zeroline = False,title = "Medical Cost"),
             )
fig = go.Figure(data = data, layout = layout)

iplot(fig)
dict_regions= {'southwest' : df[df.region == "southwest"].charges.mean(),
              'southeast' : df[df.region == "southeast"].charges.mean(),
              'northwest' : df[df.region == "northwest"].charges.mean(),
              'northeast' : df[df.region == "northeast"].charges.mean()
             }
df_regions = pd.DataFrame.from_dict(dict_regions, orient='index')
df_regions.reset_index(inplace=True)
df_regions.columns = ['regions', 'charges']

df_regions
import plotly.graph_objs as go
import colorlover as cl

trace=go.Bar(
            x=df_regions.regions,
            y=df_regions.charges,
            text="Mean Medical Costs",
            marker=dict(
                color=cl.scales['12']['qual']['Paired'],
                line=dict(
                color=cl.scales['12']['qual']['Paired'],
                width=1.5),
            ),
            opacity=0.8)

data = [trace]
layout = go.Layout(title ='Medical Cost Means by Regions',
              xaxis = dict(title = 'Region'),
              yaxis = dict(title = 'Medical Cost'))
fig = go.Figure(data = data, layout = layout)
iplot(fig)
smoker_list = [df[df.smoker == "yes"].smoker.value_counts().tolist(), df[df.smoker == "no"].smoker.value_counts().tolist()]
smoker_list = [smoker_list[0][0], smoker_list[1][0]]
smoker_list
labels = ["Smoker", "Non-Smoker"]
values = smoker_list
colors = ['#feb3b3', '#c5feb3']
trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='percent', 
               textfont=dict(size=20),
               marker=dict(colors=colors,
                           line=dict(color='#000000', width=2)))
data = [trace]
layout = go.Layout(title='Rate of Smokers & Non-Smokers')
fig = go.Figure(data = data, layout = layout)
iplot(fig)
charges_sorted.head()
trace0 = go.Scatter(
    x = charges_sorted.index,
    y = charges_sorted[charges_sorted.smoker == "yes"].charges,
    name = "Smokers",
    mode='lines',
    marker=dict(
        size=12,
        color = "red", #set color equal to a variable
    )
)

trace1 = go.Scatter(
    x = charges_sorted.index,
    y = charges_sorted[charges_sorted.smoker == "no"].charges,
    name = "Non-Smokers",
    mode='lines',
    marker=dict(
        size=12,
        color = "green", #set color equal to a variable
    )
)


data = [trace0,trace1]
layout = go.Layout(title = 'Medical Costs of Smoker vs Non-Smokers',
              xaxis = dict(title = 'Persons'),
              yaxis = dict(title = 'Medical Costs'),)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
df_smokers = df[df.smoker == "yes"]
df_smokers.reset_index(inplace=True)
df_non_smokers = df[df.smoker == "no"]
df_non_smokers.reset_index(inplace=True)
trace0 = go.Histogram(
    x=df_non_smokers.children,
    opacity=0.75,
    name = "Non-Smokers",
    marker=dict(color='rgba(166, 217, 106, 1)'))

trace1 = go.Histogram(
    x=df_smokers.children,
    opacity=0.75,
    name = "Smokers",
    marker=dict(color='rgba(244, 109, 67, 1)'))

data = [trace0,trace1]
layout = go.Layout(barmode='overlay',
                   title='Childrens of Smokers vs Non-Smokers',
                   xaxis=dict(title='Number of Children'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)