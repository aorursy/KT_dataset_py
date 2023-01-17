# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd



import plotly.graph_objects as go



import plotly.express as px



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
netflix=pd.read_csv('/kaggle/input/netflix-shows/netflix_titles_nov_2019.csv')



netflix.shape
netflix.head(5)
netflix['season_count']=netflix.apply(lambda x : x['duration'].split(" ")[0] if "Season" in x['duration'] else "" , axis = 1)



netflix.shape
netflix.head()
group_netflix=netflix.type.value_counts()



trace=go.Pie(labels=group_netflix.index, values=group_netflix.values,pull=[0.05])

layout = go.Layout(title="TV Shows VS Movies", height=400, legend=dict(x=1.1, y=1.3))

fig = go.Figure(data=[trace], layout=layout)

fig.update_layout(height=500, width=700)

fig.show()
netflix['date_added']=pd.to_datetime(netflix['date_added'])

netflix['year_added']=netflix['date_added'].dt.year



movie_per_year=[]

tv_shows_per_year=[]

for i in range(2010,2020):

    h=netflix.loc[(netflix['type']=='Movie') & (netflix.year_added==i)]

    g=netflix.loc[(netflix['type']=='TV Show') & (netflix.year_added==i)]

    movie_per_year.append(len(h))

    tv_shows_per_year.append(len(g))

    

trace1 = go.Scatter(x=[i for i in range(2008,2020)], y = movie_per_year,mode='lines+markers', name='Movies')

trace2 = go.Scatter(x=[i for i in range(2008,2020)], y = tv_shows_per_year,mode='lines+markers', name='TV Shows')

data = [trace1, trace2]

layout = go.Layout(title="content added over the years", legend=dict(x=0.1, y=1.1, orientation='h'))



fig = go.Figure(data, layout=layout)

fig.show()
top_countries=netflix.country.value_counts()



top_countries=top_countries[:15][::-1]

trace = go.Bar(x=top_countries.values, y=top_countries[:15].index, orientation='h', name='', marker=dict(color='#6ad49b'))



data=[trace]

layout=go.Layout(title='Countries with most content', height=700, legend=dict(x=0.1, y=1.1))

fig = go.Figure(data, layout=layout)

fig.show()
my_first_submission = pd.DataFrame({"PassengerId": test_one.PassengerId, "Survived": test_one.Survived})

my_first_submission.to_csv("my_first_submission.csv", index=False)