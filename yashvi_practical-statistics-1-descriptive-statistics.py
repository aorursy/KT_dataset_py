from IPython.core.interactiveshell import InteractiveShell 
InteractiveShell.ast_node_interactivity = "all"

import numpy as np
from numpy import percentile
import pandas as pd
import pandas_profiling 
from scipy.stats import norm

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.colors import n_colors
from plotly.subplots import make_subplots


titanic_data=pd.read_csv('../input/titanic/train.csv')
netflix_data=pd.read_csv('../input/netflix-shows/netflix_titles.csv')
airbnb_data=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
fig = go.Figure(go.Bar(
    x=titanic_data['Sex'],y=titanic_data['Sex'].value_counts(),
))
fig.update_layout(title_text='Frequency of Male and Female in titanic',xaxis_title="Gender",yaxis_title="Count")

net_category=netflix_data['type'].value_counts().to_frame().reset_index().rename(columns={'index':'type','type':'count'})
fig = go.Figure([go.Pie(labels=net_category['type'], values=net_category['count'])])
fig=fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=15,insidetextorientation='radial')
fig=fig.update_layout(title="Netflix Show Types",title_x=0.5)
fig.show()
titanic_data.head()
sns.distplot(titanic_data['Age'],kde=False)
titanic_data.describe().tail(5)
mean = np.mean(titanic_data['Age'])
std = np.std(titanic_data['Age'])
def z_score(value, mean, std):
    return (value - mean) / std
import random
values = []
for i in list(range(0,5)):
    value = random.choice(titanic_data.Age)
    values.append(value)
print(values)
for val in values:
    z = z_score(val, mean, std)
    print(z)
ax = sns.distplot(titanic_data['Age'], kde = True)

ax=plt.axvline(titanic_data['Age'].mean(), color='green')
ax=plt.axvline(titanic_data['Age'].mean()-titanic_data['Age'].std(), color='red')
ax=plt.axvline(titanic_data['Age'].mean()+titanic_data['Age'].std(), color='red')
ax=plt.axvline(titanic_data['Age'].mean()-2*titanic_data['Age'].std(), color='blue')
ax=plt.axvline(titanic_data['Age'].mean()+2*titanic_data['Age'].std(), color='blue')

fig = go.Figure(go.Box(y=titanic_data['Age'],name="Age")) 
fig.update_layout(title="Distribution of Age")
age_male=titanic_data[titanic_data['Sex']=='male']['Age']
age_female=titanic_data[titanic_data['Sex']=='female']['Age']
fig = go.Figure()
fig=fig.add_trace(go.Box(y=age_male,
                     marker_color="blue",
                     name="Male age"))
fig=fig.add_trace(go.Box(y=age_female,
                     marker_color="red",
                     name="female age"))
fig.update_layout(title="Age Distribution of male and female")

titanic_data.head()
g = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=titanic_data,
                height=6, kind="bar")
g.set_ylabels("survival probability")
top_release_india=netflix_data[(netflix_data['country']=='India')&
                    ((netflix_data['release_year']==2015)|(netflix_data['release_year']==2016)|(netflix_data['release_year']==2017)|(netflix_data['release_year']==2018)|
                    (netflix_data['release_year']==2019)|(netflix_data['release_year']==2020))]['release_year'].value_counts().to_frame().reset_index().rename(columns={'index':'release_year','release_year':'count'})

top_release_us=netflix_data[(netflix_data['country']=='United States')&
                    ((netflix_data['release_year']==2015)|(netflix_data['release_year']==2016)|(netflix_data['release_year']==2017)|(netflix_data['release_year']==2018)|
                    (netflix_data['release_year']==2019)|(netflix_data['release_year']==2020))]['release_year'].value_counts().to_frame().reset_index().rename(columns={'index':'release_year','release_year':'count'})

fig = go.Figure()
ax=fig.add_trace(go.Bar(x=top_release_india['release_year'],
                y=top_release_india['count'],
                name='India',
                marker_color='blue'
                ))
ax=fig.add_trace(go.Bar(x=top_release_us['release_year'],
                y=top_release_us['count'],
                name='United States',
                marker_color='violet'
                ))

fig.update_layout(title_text='Netflix shows by India/US over past 5 years',xaxis_title="Year",yaxis_title="Number of Shows",
                  barmode='stack') # by default it is group, else barmode='group'

tips = sns.load_dataset("tips")
ax = sns.scatterplot(x="total_bill", y="tip", data=tips)
ax = sns.scatterplot(x="total_bill", y="tip", hue="time",
                     data=tips)
ax = sns.scatterplot(x="total_bill", y="tip", size="size",
                     data=tips)
g = sns.relplot(x="total_bill", y="tip",
                 col="time", hue="day", style="day",
                 kind="scatter", data=tips)
corr_matrix=titanic_data.corr()
corr_matrix
sns.heatmap(corr_matrix, annot=True)
plt.show()