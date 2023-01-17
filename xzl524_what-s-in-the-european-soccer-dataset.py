%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

pd.set_option('display.width',500)

pd.set_option('display.max_columns',25)

pd.set_option('display.notebook_repr_html',True)

import seaborn as sns

sns.set_style('whitegrid')

sns.set_context('notebook')

import sqlite3

import bs4

from scipy.stats import poisson

from scipy.stats import norm

import pickle
con=sqlite3.connect('../input/database.sqlite')

cursor=con.execute("select name from sqlite_master where type='table'")

cursor.fetchall()
player=pd.read_sql_query('select * from Player',con)

match=pd.read_sql_query('select * from match',con)

league=pd.read_sql_query('select * from league',con)

country=pd.read_sql_query('select * from Country',con)

team=pd.read_sql_query('select * from Team',con)
country
country.info()
league
league.info()
team.head()
team.info()
team.describe()
team.isnull().sum()
team[team['team_fifa_api_id'].isnull()]
#team.groupby('team_api_id',as_index=False).count().sort('id',ascending=False).head()
#team.groupby('team_fifa_api_id',as_index=False).count().sort('id',ascending=False).head()
team[team['team_fifa_api_id'].isin([301,111560,111429])]
team.describe(include=['O'])
#team.groupby('team_long_name',as_index=False).count().sort('id',ascending=False).head()
team[team['team_long_name'].isin(['Royal Excel Mouscron','Widzew Łódź','Polonia Bytom'])]
#team.groupby('team_short_name',as_index=False).count().sort('id',ascending=False).head(10)
player.head()
player.info()
player.isnull().sum()
player.describe()
player.describe(include=['O'])
#player.groupby('player_name',as_index=False).count().sort('player_api_id',ascending=False).head(10)
player[player['player_name']=='Danilo']
#player.groupby('birthday',as_index=False).count().sort('player_api_id',ascending=False).head(4)
player[player['birthday']=='1989-03-02 00:00:00'].head(3)
sns.distplot(player['height'],bins=20,kde=True,

            kde_kws={"bw": 4, "label": "KDE"})

plt.axvline(player['height'].mean(), 0, 1, color='r', label='Average Height')

plt.xlabel('Player Height Distribution (cm)')

plt.legend()
sns.distplot(player['weight'],bins=20,

            kde_kws={"bw": 4, "label": "KDE"})

plt.axvline(player['weight'].mean(), 0, 1, color='r', label='Average Weight')

plt.xlabel('Player Weight Distribution (lbs)')

plt.legend()
plt.scatter(player.weight,player.height,alpha=0.5,s=10)