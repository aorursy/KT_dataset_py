

import pandas as pd

import numpy as np

import  matplotlib.pyplot as plt

import  seaborn as sns

import warnings

fig_size = [80, 80]

plt.rcParams['figure.figsize'] = fig_size

warnings.filterwarnings('ignore')

%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('../input/bgg_db_2017_04.csv', encoding='latin1')



# Data Cleaning

missing = df[(df['min_players'] < 1) 

          | (df['max_players'] < 1) | (df['avg_time'] < 1) 

           | (df['min_time'] < 1) | (df['max_time'] < 1) | (df['year'] < 1950) | (df['max_players'] > 10)]

df = df[(df['min_players'] >= 1) & (df['max_players'] >= 1)  

        & (df['avg_time'] >= 1) & (df['min_time'] >= 1) & (df['max_time'] >= 1) & (df['year'] > 1950) 

        & (df['max_players'] < 10)]

df['avg_players'] = (df['min_players'] + df['max_players'])/2 

df.category = df.category.astype('category')
bins = [-1, 1, 2, 3, 4, 5]

df['weight_cat'] = pd.cut(df['weight'], bins=bins, labels=bins[1:])

df['weight_cat']



weight_avg = [df[df['weight_cat'] == i]['avg_rating'] for i in range(1,6)]

weight_geek = [df[df['weight_cat'] == i]['geek_rating'] for i in range(1,6)]



f, axes = plt.subplots(1, 2, figsize = (18, 10), sharex = True, sharey = True) 



k1 = sns.violinplot(data=weight_avg, ax = axes[0] )

k2 = sns.violinplot(data=weight_geek, ax = axes[1])

axes[0].set(xlabel='weight range', ylabel='avgerage rating', xticklabels=['0-1', '1-2', '2-3', '3-4', '4-5'])

axes[1].set(xlabel='weight range', ylabel='geek rating', xticklabels=['0-1', '1-2', '2-3', '3-4', '4-5'])

df['player_range'] = df['min_players'].astype(str) + '-' + df['max_players'].astype(str)

player_range =  df['player_range'].value_counts()

player_range = player_range[player_range > 50]

vis = sns.barplot(x = player_range.index, y= player_range)

sns.set(font_scale = 2)

vis.set(xlabel='player range', ylabel = 'count')

plt.rcParams["figure.figsize"] = [40, 20]
player_counts =  df['avg_players'].value_counts()

player_counts = player_counts[player_counts > 50]

vis = sns.barplot(x = player_counts.index, y = player_counts)

sns.set(font_scale = 2)

plt.title('Mean Player Counts')

vis.set(xlabel='average players')

vis.set(ylabel='player count')

plt.show()
game_count = df['year'].value_counts()

game_count = game_count[game_count > 25]

sns.barplot(x = game_count.index, y = game_count)

plt.title('Mean Player Counts')

vis.set(xlabel='player count')
f, axes = plt.subplots(1, 2, figsize = (18, 10), sharex = True, sharey = True) 

k1 = sns.kdeplot(df['weight'], df['avg_rating'] ,ax = axes[0])

k2 = sns.kdeplot(df['weight'], df['geek_rating'], ax = axes[1])
sns.kdeplot(df['avg_rating'], df['geek_rating'], shade = True, cmap = 'Reds')

sns.set(font_scale = 2)

plt.rcParams["figure.figsize"] = [18, 9]

vis = sns.distplot(df['age'], bins = 30)
vis = sns.boxplot(data = df, x = 'age', y = 'weight')