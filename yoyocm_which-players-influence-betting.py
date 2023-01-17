import sqlite3 as lite

import pandas as pd



database = '../input/database.sqlite'

conn = lite.connect(database)



team = "Barcelona"



query = 'SELECT * FROM Match'

dfmatch = pd.read_sql(query, conn)



# Use team names instead of ids

query = 'SELECT team_api_id, team_long_name FROM Team'

dfteam = pd.read_sql(query, conn)

seteam = pd.Series(data=dfteam['team_long_name'].values, index=dfteam['team_api_id'].values)

dfmatch['home_team_name'] = dfmatch['home_team_api_id'].map(seteam)

dfmatch['away_team_name'] = dfmatch['away_team_api_id'].map(seteam)



# Use country names instead of ids

query = 'SELECT id, name FROM Country'

dfcountry = pd.read_sql(query, conn)

secountry = pd.Series(data=dfcountry['name'].values, index=dfcountry['id'].values)

dfmatch['country'] = dfmatch['country_id'].map(secountry)



# Countries with highest attendance

countries = ['England', 'Spain', 'Germany', 'Italy', 'France']

dfmatch = dfmatch[dfmatch['country'].isin(countries)]



# Use player names instead of ids

query = 'SELECT player_api_id, player_name FROM Player'

dfplayer = pd.read_sql(query, conn)

seplayer = pd.Series(data=dfplayer['player_name'].values, index=dfplayer['player_api_id'].values)

for z in [x+'_player_'+str(y) for x in ['home', 'away'] for y in range(1, 12)]:

    dfmatch[z+'_name'] = dfmatch[z].map(seplayer)

    

dfhome = dfmatch[ dfmatch['home_team_name'].str.contains(team) ]

dfaway = dfmatch[ dfmatch['away_team_name'].str.contains(team) ]


cplayers=[]

for index, row in dfhome.iterrows():

    players = ""

    for z in ['home_player_'+str(y)+'_name' for y in range(1, 12)]:

        players = players + "," + dfhome[z]

    cplayers.append(players)

    

dfhome['players'] = cplayers



cplayers=[]

for index, row in dfaway.iterrows():

    players = ""

    for z in ['away_player_'+str(y)+'_name' for y in range(1, 12)]:

        players = players + ", " + dfaway[z]

    cplayers.append(players)

    

dfaway['players'] = cplayers
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

import re



REGEX = re.compile(r",\s*")

def tokenize(text):

    return [tok.strip().lower() for tok in REGEX.split(text)]



dfhome['bet'] = dfhome['B365H']

dfaway['bet'] = dfaway['B365A']



bets = np.concatenate((dfhome['bet'].values,dfaway['bet'].values))

players = np.concatenate((dfhome['players'].values[1],dfaway['players'].values[1]))





data = pd.DataFrame(bets)

data['players'] = players

data = data.dropna()



# Extract players features 

count_vect = CountVectorizer(tokenizer=tokenize,max_features=20)

X_counts = count_vect.fit_transform(data['players'].str[1:])



data = pd.merge(data,pd.DataFrame(X_counts.toarray()),left_index=True, right_index=True)



data = data.drop("players",axis=1).dropna()
from sklearn.ensemble import RandomForestRegressor



y = data["0_x"]

data = data.drop("0_x",axis=1)



clf = RandomForestRegressor(n_estimators=10000)

clf = clf.fit(data, y)

print("Score : " + str(clf.score(data, y)))



importances = clf.feature_importances_

std = np.std([tree.feature_importances_ for tree in clf.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

inv_vocab = {v: k for k, v in count_vect.vocabulary_.items()}

print("\nFeature ranking:")



for x in indices:

    print(inv_vocab[x] + " " + str(importances[x]))