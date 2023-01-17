import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("../input/IPL_Match_Data.csv")

df.head()
df.describe()
df.isna().sum() #To check number of null values in al columns
df.drop(columns = ['umpire3'], inplace = True) #Deleted umpire 3 column as it had all null values
df = df[df['result'] != 'no result'] #Removing the matches having no result
df['venue'].unique() #Checking diff venues
df1 = df.drop(columns = ['id', 'season', 'city', 'date']) #These dont look useful, also venue is more important than city

df1.head()
df1['dl_applied'].unique() #Checking values in dl_applied
df1['win_by_wickets'] = df1['win_by_wickets'].apply(lambda x: 1 if x > 0 else 0)
df1['win_by_runs'] = df1['win_by_runs'].apply(lambda x: 1 if x > 0 else 0) #Mapping all the values to 1 if win else 0.
                                                                            #The exact number doesnt matter

df1['win_by_wickets'].unique(), df1['win_by_runs'].unique()
teams = list(df1['team1'].unique())

teams #Check different teams
teams_dict = dict.fromkeys(teams)

j = 0

for i in teams_dict:
    teams_dict[i] = j
    j += 1
    
teams_dict #Creating a dictionary of teams to map in the columns for decision trees
for i in ['team1', 'team2', 'toss_winner', 'winner']:
    df1[i] = df1[i].apply(lambda x: teams_dict.get(x)) #Applied numbers to each team (Same was important)
    
df1.head()
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn import preprocessing

for i in ['toss_decision', 'player_of_match', 'venue', 'umpire1', 'umpire2']:
    df1[i] = preprocessing.LabelEncoder().fit_transform(df1[i])
    
df1.head()
df1.drop(columns = ['player_of_match'], inplace = True)

from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import train_test_split as tts

x = df1.drop(columns = ['result', 'win_by_wickets', 'win_by_runs', 'winner', 'umpire2', 'umpire1', 'venue'])
y = df1['winner']

x_train, x_test, y_train, y_test = tts(x,y, random_state = 17)

regressor = dtc()

regressor.fit(x_train, y_train)

regressor.score(x_test, y_test)
from random import randint as rand

predictions = []

team1 = list(x_test['team1'])
team2 = list(x_test['team2'])

for i in range(len(team1)):
    if rand(1,2) == 1:
        predictions.append(team1[i])
    else:
        predictions.append(team2[i])
        
predictions

regressor.score(x_test, predictions)
from sklearn.ensemble import RandomForestClassifier as rfc

x = df1.drop(columns = ['result', 'win_by_wickets', 'win_by_runs', 'winner', 'umpire2', 'umpire1', 'venue'])
y = df1['winner']

x_train, x_test, y_train, y_test = tts(x,y, random_state = 17)

regressor = rfc()

regressor.fit(x_train, y_train)

regressor.score(x_test, y_test)
from sklearn.ensemble import GradientBoostingClassifier as gbc

x = df1.drop(columns = ['result', 'win_by_wickets', 'win_by_runs', 'winner', 'umpire2', 'umpire1', 'venue'])
y = df1['winner']

x_train, x_test, y_train, y_test = tts(x,y, random_state = 17)

regressor = gbc()

regressor.fit(x_train, y_train)

regressor.score(x_test, y_test)