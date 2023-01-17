import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.plotly as py

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()

%matplotlib inline
data = pd.read_csv("../input/results.csv")
data.info()
data.describe()
def result(row):

      

    if row['home_score'] > row['away_score']:

        return row['home_team']

    elif row['home_score'] < row['away_score']:

        return row['away_team']

    else:

        return('Tie')

    
data['results'] = data[['home_score','away_score','home_team','away_team']].apply(result,axis=1)

data.head(2)
data_corr = data.corr()

sns.heatmap(data_corr,cmap='plasma_r', lw = 1, annot = True)
type(data['date'].iloc[0])
data['date'] = pd.to_datetime(data['date'])

time = data['date'].iloc[0]

time.year
data['month'] = data['date'].apply(lambda time: time.month)

data['year']  = data['date'].apply(lambda time: time.year)
data['month'] = data['month'].map({1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 

                                 10:'Oct', 11:'Nov', 12:'Dec'})
data.head(2)
match_count = data['year'].value_counts()

match_count.head(2)
plt.figure(figsize=(20,8))

sns.lineplot(match_count.index, match_count.values, color='red', lw=2)

plt.title('Number of matches played in each year', fontsize=20)

plt.ylabel('No of matches', fontsize=12)

plt.xlabel('Year', fontsize=12)
plt.figure(figsize=(20,8))

sns.countplot(x='results',data=data, order =data['results'].value_counts()[1:20].index)

plt.title('Top 20 Countries with most wins', fontsize=20)

plt.ylabel('No of wins', fontsize=12)

plt.xlabel('Country', fontsize=12)
brazil = data[data['results']=='Brazil']['city'].value_counts()[:20]
plt.figure(figsize=(18,10))

sns.barplot(brazil.values,brazil.index,palette='autumn')

plt.title("Favourite grounds for Brazil when they win", fontsize=20)

plt.ylabel('City', fontsize=12)

plt.xlabel('No of times won', fontsize=12)
def home_wins(row):

        

        if row['home_team'] == row['results']:

            return("Home")

        elif row['home_team'] != row['results'] and row['results'] == 'Tie':

            return("Tie")

        else:

            return("Away")

       
data['home_win'] = pd.DataFrame(data[['home_team','results']].apply(home_wins,axis=1))

data.head()
plt.figure(figsize=(7,6))

sns.countplot(x='home_win', palette='rainbow',hue='neutral',data=data[data.home_win != 'Tie'])

plt.title('Home vs Away wins', fontsize=20)

plt.ylabel('No of wins', fontsize=12)

plt.xlabel('Home or Away', fontsize=12)

tour = data['tournament'].value_counts()

tour.head()
data1 = dict(

      values = tour.values[:20],

      labels = tour.index[:20],

      domain = {"x": [0, .5]},

      hoverinfo = "label+percent+name",

      type =  "pie")

layout1 = dict(

        title =  "Top 20 most played Leagues",

            )

fig = go.Figure([data1],layout1)

iplot(fig)
total_scores = data[['home_score','away_score','country','month','year']]

total_scores['total_score'] = total_scores['home_score'] + total_scores['away_score']

total_scores.head()
plt.figure(figsize=(25,10))

dj = total_scores.pivot_table(index='month',columns='year',values='total_score')

sns.heatmap(dj,cmap='cividis_r',linecolor='white', lw = 0.2)

plt.title('Goals scored across each month and year', fontsize=20)

plt.ylabel('Years', fontsize=12)

plt.xlabel('Months', fontsize=12)

goals_scored = total_scores.groupby('country').sum()
data2 = dict(type = 'choropleth',

            colorscale = 'Portland',

            locations = goals_scored.index,

            locationmode = 'country names',

            z = goals_scored['total_score'],

            text = goals_scored.index,

            colorbar = {'title':'No of Goals'})



layout2 = dict(title = 'Number of goals scored in various Venues',

               geo = dict(showframe = False, projection = {'type':'natural earth'}))



choromap2 = go.Figure([data2],layout2)



iplot(choromap2)
country_grouped = total_scores.groupby('country').sum().sort_values('total_score',ascending=False)[:30]
ax = plt.figure(figsize=(15,14))

sns.barplot(x="total_score", y=country_grouped.index, data=country_grouped, color ='yellow', label="Total_score")

sns.barplot(x="home_score", y=country_grouped.index, data=country_grouped, color = 'green', label="Away_score")

ax.legend(ncol=2, loc="upper right", frameon=True)

plt.title("Total & Away goals scored in the country's History", fontsize=20)

plt.ylabel('Country', fontsize=12)

plt.xlabel('No of goals', fontsize=12)

aaa= data['home_team'].value_counts()

bbb = data['away_team'].value_counts()

team_matches = pd.concat([aaa,bbb],axis=1)
team_matches.columns = ['home_matches','away_matches']

team_matches['total_matches'] = team_matches['home_matches'] + team_matches['away_matches']

team_matches_sort = team_matches.sort_values('total_matches',ascending=False)
team_matches_sort.iplot(kind='scatter',title='Number of matches played', xTitle='Country', yTitle='No of matches',theme='pearl')

venues = pd.DataFrame(data['country'].value_counts())

venues.head()
data3 = dict(type = 'choropleth',

            colorscale = 'Reds',

            locations = venues.index,

            locationmode = 'country names',

            z = venues['country'],

            text = venues.index,

            colorbar = {'title':'No of matches hosted'})



layout3 = dict(title = 'Number of matches hosted by various Venues',

               geo = dict(showframe = False, projection = {'type':'natural earth'}))



choromap3 = go.Figure([data3],layout3)

iplot(choromap3)
