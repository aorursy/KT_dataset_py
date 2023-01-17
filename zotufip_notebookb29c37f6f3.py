%matplotlib inline

import numpy as np

import pandas as pd

import sqlalchemy as sql

from pandas import DataFrame
engine = sql.create_engine("sqlite:///../input/database.sqlite")

with engine.connect() as conn, conn.begin():

    match_data = pd.read_sql_table("Match",conn)

    leagues = pd.read_sql_table("League",conn)

    teams = pd.read_sql_table("Team",conn)
def winTag(row):

    if(row.team_id == row.home_team_api_id):

        team_goals = row.home_team_goal

        opponent_goals = row.away_team_goal

    else:

        team_goals = row.away_team_goal

        opponent_goals = row.home_team_goal

    return 3 if team_goals>opponent_goals else 1 if team_goals == opponent_goals else 0 #We assign standard amount of points for every type of result
prepared_data = match_data[match_data.league_id == 15688] #Poland Ekstraklasa

unique_teams_ids = prepared_data.home_team_api_id.append(prepared_data.away_team_api_id).unique()

seasons = prepared_data.season.unique()

team_matches_with_history = DataFrame(columns=['id','season','stage','date','home_team_api_id','away_team_api_id','home_team_goal','away_team_goal'])

for team in unique_teams_ids:

    d = prepared_data[(prepared_data.home_team_api_id == team) | (prepared_data.away_team_api_id == team)]

    d = d[['id','season','stage','date','home_team_api_id','away_team_api_id','home_team_goal','away_team_goal']]

    d.sort_values(by='date', inplace=True)

    d['team_id'] = team

    d['result'] = d.apply(winTag,axis=1)

    for i in range(5):

        d['prev'+str(i+1)] = d.result.shift(i+1)

        d.loc[d.stage<=i+1,10+i:] = 0

        

    team_matches_with_history = team_matches_with_history.append(d)

team_matches_with_history = team_matches_with_history[['id','prev1','prev2','prev3','prev4','prev5','team_id']]

prepared_data = prepared_data.merge(team_matches_with_history,left_on=['id','home_team_api_id'],right_on=['id','team_id'])

prepared_data.rename(columns={'prev1':'home_prev1','prev2':'home_prev2','prev3':'home_prev3','prev4':'home_prev4','prev5':'home_prev5'},inplace=True)

prepared_data.drop('team_id',axis=1,inplace=True)

prepared_data = prepared_data.merge(team_matches_with_history,left_on=['id','away_team_api_id'],right_on=['id','team_id'])

prepared_data.rename(columns={'prev1':'away_prev1','prev2':'away_prev2','prev3':'away_prev3','prev4':'away_prev4','prev5':'away_prev5'},inplace=True)

prepared_data.drop('team_id',axis=1,inplace=True)

prepared_data.sort_values('date',inplace=True)

prepared_data.fillna(value=0,inplace=True)

team_names = teams[['team_long_name','team_api_id']]

prepared_data = prepared_data.merge(team_names,left_on='home_team_api_id',right_on='team_api_id')

prepared_data.rename(columns={'team_long_name':'home_team_long_name'},inplace=True)

prepared_data.drop(['team_api_id'],axis=1,inplace=True)

prepared_data = prepared_data.merge(team_names,left_on='away_team_api_id',right_on='team_api_id')

prepared_data.rename(columns={'team_long_name':'away_team_long_name'},inplace=True)

prepared_data.drop(['team_api_id'],axis=1,inplace=True)

prepared_data
tagger = lambda x: "1" if x.home_team_goal > x.away_team_goal else "X" if x.home_team_goal == x.away_team_goal else "2"

prepared_data['result'] = prepared_data.apply(tagger,axis=1)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder



team_columns = prepared_data[['home_team_long_name','away_team_long_name']].T.to_dict().values()

from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)

teams_as_features = DataFrame(dv.fit_transform(team_columns))
ml_feed = prepared_data[['home_prev1','home_prev2','home_prev3','home_prev4','home_prev5','away_prev1','away_prev2','away_prev3','away_prev4','away_prev5']]

ml_feed = pd.concat([ml_feed,teams_as_features],axis=1,join='inner')

ml_feed



from sklearn.preprocessing import normalize

ml_feed = normalize(ml_feed)

from sklearn.cross_validation import train_test_split

X_t,X_v,Y_t,Y_v = train_test_split(ml_feed,prepared_data['result'],test_size=0.3)
from sklearn.svm import SVC

svc = SVC()

svc.fit(X_t,Y_t)

svc.score(X_v,Y_v)