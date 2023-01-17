import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import os

print(os.listdir("../input"))

data = pd.read_csv('../input/results.csv')
data.head()
data.home_team.value_counts().head(10)
data.away_team.value_counts().head(10)
data.tournament.value_counts().head(10)
data['home_win'] = data['home_score'] #filler values for now



for i in range(data.shape[0]): #for each row

    

    home_score = data.at[i,'home_score'] #get home score

    away_score = data.at[i,'away_score'] #get away score

    

    if home_score > away_score: #home score win

        data.at[i,'home_win'] = 1

    elif away_score > home_score: #away score win

        data.at[i,'home_win'] = 0

    else:

        data.at[i,'home_win'] = 0.5 #tie
data['home_win'].head()
unique_countries_home = list(data.home_team.unique())

unique_countries_away = list(data.away_team.unique())

unique_countries = unique_countries_home + unique_countries_away

#Only get unique values by converting to set, then list

unique_countries = list(set(unique_countries))
country_stats = pd.DataFrame({"country":unique_countries})
country_stats.country.value_counts()
data.head(3)
#Initialize 4 columns

col_names = ['away_win','away_lose','home_win','home_lose']

for name in col_names:

    country_stats[name] = 0 #placeholder value



#Counting process

for i in range(data.shape[0]):

    

    #Get indexes in country_stat for home and away country

    home_index = country_stats[country_stats['country']==data.at[i,'home_team']].index.values.astype(int)[0]

    away_index = country_stats[country_stats['country']==data.at[i,'away_team']].index.values.astype(int)[0]

    

    #Add 1 to either away win or away lose to away team, and home win or home lose to home team

    if data.at[i,'home_win'] == 1: #The home team has won

        country_stats.at[home_index,'home_win'] += 1

        country_stats.at[away_index,'away_lose'] += 1

    elif data.at[i,'home_win'] == 0: #The home team has lost

        country_stats.at[home_index,'home_lose'] += 1

        country_stats.at[away_index,'away_win'] += 1

    else: #tie. We will just do nothing, since no one really won or lost.

        pass
country_stats.head()
#Get total losses and wins

country_stats['lose'] = country_stats['away_lose'] + country_stats['home_lose']

country_stats['win'] = country_stats['away_win'] + country_stats['home_win']



#Get total games played away and at home, and in general

country_stats['home'] = country_stats['home_win'] + country_stats['home_lose']

country_stats['away'] = country_stats['away_win'] + country_stats['away_lose']

country_stats['games'] = country_stats['home'] + country_stats['away']



#Win-to-lose ratio (the bigger the better)

country_stats['win_lose_ratio'] = country_stats['win']/country_stats['lose']
country_stats.head()
country_stats[country_stats.lose == 0]
#Dealing with two pesky infinity cases, we can just delete them as they are

#not significant

country_stats.drop(list(country_stats[country_stats.lose == 0].index.values.astype(int)),inplace=True)
country_stats[country_stats.lose == 0]
country_stats.head()
grid = sns.JointGrid(country_stats.lose, country_stats.win, space=0, size=7, ratio=5)

grid.plot_joint(plt.scatter, color="b")

plt.plot([0, 0], [700, 700], linewidth=100)

plt.title('Win to Lose Ratio Scatterplot')
sns.distplot(country_stats.win_lose_ratio,rug=True)

plt.title("Country Win to Lose Ratio Distribution Plot")
country_stats['away_win_lose_ratio'] = country_stats['away_win']/country_stats['away_lose']

country_stats['home_win_lose_ratio'] = country_stats['home_win']/country_stats['home_lose']



country_stats['home_away_degree'] = country_stats['home_win_lose_ratio'] - country_stats['away_win_lose_ratio']
country_stats.head()
country_stats['home_away_degree'].nlargest(5)
country_stats.loc[list(country_stats[country_stats.home_away_degree == float('inf')].index.values.astype(int))]
problem_indexes = list(country_stats[country_stats.home_away_degree == float('inf')].index.values.astype(int))

bad_columns = ['home_win_lose_ratio','home_away_degree']

for column in bad_columns:

    for index in problem_indexes:

        country_stats.at[index,column] = np.nan
country_stats['home_away_degree'].nlargest(5)
country_stats.loc[country_stats['home_away_degree'].nlargest(5).index.values.astype(int)]
country_stats[country_stats.country=='Jersey']
country_stats['home_away_degree'].nsmallest(5)
country_stats.loc[country_stats['home_away_degree'].nsmallest(3).index.values.astype(int)]
problem_indexes = country_stats['home_away_degree'].nsmallest(3).index.values.astype(int)

bad_columns = ['away_win_lose_ratio','home_away_degree']

for column in bad_columns:

    for index in problem_indexes:

        country_stats.at[index,column] = np.nan
country_stats['home_away_degree'].nsmallest(5)
country_stats.loc[country_stats['home_away_degree'].nsmallest(5).index.values.astype(int)]
top_cs = country_stats

top_cs = top_cs.iloc[0:0] #clearing out all data

top_cs.drop('country',axis=1,inplace=True)
top_cs
for i in range(5):

    top_cs.loc[i] = 0
top_cs
top_cs.reset_index()



def get_country(index): #function to get country by index

    return country_stats.loc[index]['country']

    

#For each column

for column in top_cs.columns:

    

    placement_index = 0

    

    #For each index in a list of top indexes by column value

    for index in list(country_stats[column].nlargest(5).index.values.astype(int)):

        

        #Assign value to country

        top_cs.loc[placement_index,column] = get_country(index)

        #Next index

        placement_index += 1
top_cs