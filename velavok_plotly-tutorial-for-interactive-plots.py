#!pip install plotly
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.offline as py
import warnings
import pycountry
warnings.filterwarnings('ignore')

py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
PATH = '../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv'
data = pd.read_csv(PATH)
data.head(3)
colors = ['#f4cb42', '#cd7f32', '#a1a8b5'] #gold,bronze,silver
medal_counts = data.Medal.value_counts(sort=True)
labels = medal_counts.index
values = medal_counts.values

pie = go.Pie(labels=labels, values=values, marker=dict(colors=colors))
layout = go.Layout(title='Medal distribution')
fig = go.Figure(data=[pie], layout=layout)
py.iplot(fig)
topn = 10
male = data[data.Sex=='M']
female = data[data.Sex=='F']
count_male = male.dropna().NOC.value_counts()[:topn].reset_index()
count_female = female.dropna().NOC.value_counts()[:topn].reset_index()

pie_men = go.Pie(labels=count_male['index'],values=count_male.NOC,name="Men",hole=0.4,domain={'x': [0,0.46]})
pie_women = go.Pie(labels=count_female['index'],values=count_female.NOC,name="Women",hole=0.4,domain={'x': [0.5,1]})

layout = dict(title = 'Top-10 countries with medals by sex', font=dict(size=15), legend=dict(orientation="h"),
              annotations = [dict(x=0.2, y=0.5, text='Men', showarrow=False, font=dict(size=20)),
                             dict(x=0.8, y=0.5, text='Women', showarrow=False, font=dict(size=20)) ])

fig = dict(data=[pie_men, pie_women], layout=layout)
py.iplot(fig)
games = data[data.Season=='Summer'].Games.unique()
games.sort()
sport_counts = np.array([data[data.Games==game].groupby("Sport").size().shape[0] for game in games])
bar = go.Bar(x=games, y=sport_counts, marker=dict(color=sport_counts, colorscale='Reds', showscale=True))
layout = go.Layout(title='Number of sports in the summer Olympics by year')
fig = go.Figure(data=[bar], layout=layout)
py.iplot(fig)
topn = 10
top10 = data.dropna().NOC.value_counts()[:topn]

gold = data[data.Medal=='Gold'].NOC.value_counts()
gold = gold[top10.index]
silver = data[data.Medal=='Silver'].NOC.value_counts()
silver = silver[top10.index]
bronze = data[data.Medal=='Bronze'].NOC.value_counts()
bronze = bronze[top10.index]

bar_gold = go.Bar(x=gold.index, y=gold, name = 'Gold', marker=dict(color = '#f4cb42'))
bar_silver = go.Bar(x=silver.index, y=silver, name = 'Silver', marker=dict(color = '#a1a8b5'))
bar_bronze = go.Bar(x=bronze.index, y=bronze, name = 'Bronze', marker=dict(color = '#cd7f32'))

layout = go.Layout(title='Top-10 countries with medals', yaxis = dict(title = 'Count of medals'))

fig = go.Figure(data=[bar_gold, bar_silver, bar_bronze], layout=layout)
py.iplot(fig)
tmp = data.groupby(['Sport'])['Height', 'Weight'].agg('mean').dropna()
df1 = pd.DataFrame(tmp).reset_index()
tmp = data.groupby(['Sport'])['ID'].count()
df2 = pd.DataFrame(tmp).reset_index()
dataset = df1.merge(df2) #DataFrame with columns 'Sport', 'Height', 'Weight', 'ID'

scatterplots = list()
for sport in dataset['Sport']:
    df = dataset[dataset['Sport']==sport]
    trace = go.Scatter(
        x = df['Height'],
        y = df['Weight'],
        name = sport,
        marker=dict(
            symbol='circle',
            sizemode='area',
            sizeref=10,
            size=df['ID'])
    )
    scatterplots.append(trace)
                         
layout = go.Layout(title='Mean height and weight by sport', 
                   xaxis=dict(title='Height, cm'), 
                   yaxis=dict(title='Weight, kg'),
                   showlegend=True)

fig = dict(data = scatterplots, layout = layout)
py.iplot(fig)
men = data[data.Sex=='M'].Age
women = data[data.Sex=='F'].Age

box_m = go.Box(x=men, name="Male", fillcolor='navy')
box_w = go.Box(x=women, name="Female", fillcolor='lime')
layout = go.Layout(title='Age by sex')
fig = go.Figure(data=[box_m, box_w], layout=layout)
py.iplot(fig)
#!pip install pycountry
def get_name(code):
    '''
    Translate code to name of the country
    '''
    try:
        name = pycountry.countries.get(alpha_3=code).name
    except:
        name=code
    return name

country_number = pd.DataFrame(data.NOC.value_counts())
country_number['country'] = country_number.index
country_number.columns = ['number', 'country']
country_number.reset_index().drop(columns=['index'], inplace=True)
country_number['country'] = country_number['country'].apply(lambda c: get_name(c))
country_number.head(3)
worldmap = [dict(type = 'choropleth', locations = country_number['country'], locationmode = 'country names',
                 z = country_number['number'], autocolorscale = True, reversescale = False, 
                 marker = dict(line = dict(color = 'rgb(180,180,180)', width = 0.5)), 
                 colorbar = dict(autotick = False, title = 'Number of athletes'))]

layout = dict(title = 'The Nationality of Athletes', geo = dict(showframe = False, showcoastlines = True, 
                                                                projection = dict(type = 'Mercator')))

fig = dict(data=worldmap, layout=layout)
py.iplot(fig, validate=False)
