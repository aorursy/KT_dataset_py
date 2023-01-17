# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as ws
import plotly.express as px
import plotly.graph_objs as go
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')
ws.filterwarnings ("ignore")
pd.set_option('display.max_rows', None)
df = pd.read_csv("/kaggle/input/indian-chess-grandmasters/indian_grandmasters_july_2020.csv")
df.head()
df.info()
df.head(1)
df[df['Gender']=='F'].head(1)
total_gms = len(df)
print(total_gms)
temp = df["Gender"].value_counts().reset_index()
temp.iplot(kind='pie',labels='index',values='Gender', title='Gender distribution of Grandmasters', hole = 0.5, colors=['#FF414D','#9B116F'])
df
yearwise_dist = df.Year_of_becoming_GM.value_counts().reset_index().rename(columns={'index':'Year_of_becoming_GM', 'Year_of_becoming_GM':'Total players'})
yearwise_dist.loc[23]=[2005, 0]
yearwise_dist.sort_values(by='Year_of_becoming_GM', inplace=True)
fig = go.Figure(data=go.Scatter(x=yearwise_dist["Year_of_becoming_GM"], y=yearwise_dist["Total players"], mode='lines+markers'))
fig.update_layout(title="No. of Grandmasters per year",xaxis_title="Year", yaxis_title="Count")
fig.update(layout=dict(title=dict(x=0.5))) 
fig.show()
top_10 = df.sort_values(by='Classical_Rating', ascending=False)[:10][::-1].reset_index(drop=True)
fig = px.bar(top_10, x ='Classical_Rating', y ='Name', title ='Top 10 Grandmasters based on Classical Rating', color='Classical_Rating', color_continuous_scale='Reds')
fig.update(layout=dict(title=dict(x=0.5))) 
fig.show()
top_10 = df.sort_values(by='Rapid_rating', ascending=False)[:10][::-1].reset_index(drop=True)
fig = px.bar(top_10, x ='Rapid_rating', y ='Name', title ='Top 10 Grandmasters based on Rapid Rating', color='Rapid_rating', color_continuous_scale='greens')
fig.update(layout=dict(title=dict(x=0.5))) 
fig.show()
top_10 = df.sort_values(by='Blitz_rating', ascending=False)[:10][::-1].reset_index(drop=True)
fig = px.bar(top_10, x ='Blitz_rating', y ='Name', title ='Top 10 Grandmasters based on Blitz Rating', color='Blitz_rating', color_continuous_scale='Blues')
fig.update(layout=dict(title=dict(x=0.5))) 
fig.show()
birth_year = df.Year_of_birth
current_year = 2020
age_value =  current_year - birth_year
age_df=df.copy()
age_df['Age'] = age_value
avg_age = round(np.nanmean(age_value), 1)
avg_age
fig = px.histogram(age_df, x="Age", color="Gender", title ='Age Distribution of Grandmasters', color_discrete_sequence=['#1f77b4', '#17becf'], opacity=0.5)
fig.update(layout=dict(title=dict(x=0.5))) 
fig.show()
fig = px.histogram(df, x = 'Year_of_birth', color="Gender", title ='Year of Birth Distribution of Grandmasters', color_discrete_sequence=['#1f77b4', '#17becf'], opacity=0.5)
fig.update(layout=dict(title=dict(x=0.5))) 
fig.show()
temp = df["State"].value_counts()
temp.iplot(kind='bar', xTitle = 'State', yTitle = "Count of Grandmasters", title ='No. of Grandmasters vs Indian State', color ='#FD7055')
temp = df["State"].value_counts().reset_index().rename(columns={'index':'State', 'State':'Total Grandmasters'})
fig = px.pie(temp, values='Total Grandmasters', names='State',
             title='State-wise Distribution of Indian Grandmasters', height=800, width=800)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update(layout=dict(title=dict(x=0.5))) 
fig.show()