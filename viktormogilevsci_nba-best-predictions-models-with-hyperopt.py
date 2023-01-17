import pandas as pd

import numpy as np



from hpsklearn import HyperoptEstimator, any_classifier

from hyperopt import tpe



from sklearn import metrics

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



pd.set_option('display.max_columns', None)  

pd.set_option('display.expand_frame_repr', False)

pd.set_option('max_colwidth', -1)



import pickle

import warnings

warnings.filterwarnings('ignore')
raw_game_data = pd.read_csv('../input/nba-1219-seasons-datasets/2012-18_teamBoxScore.csv')

raw_game_data
games_stat = raw_game_data.copy()

games_stat.columns = map(str.upper, games_stat.columns)

games_stat['TEAMRSLT'] = games_stat.apply(lambda x: 1 if x['TEAMRSLT'] == 'Win' else 0, axis=1)



id = 1;

game_ids = [];

for i in range(0, len(games_stat), 2):

    game_ids.append(id)

    game_ids.append(id)

    id += 1



games_stat['GAME_ID'] = game_ids



importance_columns = ['TEAMFGM', 'TEAMFGA', 'TEAMFG%', 'TEAM3PM', 'TEAM3PA', 'TEAM3P%', 'TEAMFTM', 

                      'TEAMFTA', 'TEAMFT%', 'TEAMAST', 'TEAMSTL', 'TEAMBLK', 'TEAMPF', 

                      'OPPTFGM', 'OPPTFGA', 'OPPTFG%', 'OPPT3PM', 'OPPT3PA', 'OPPT3P%', 'OPPTFTM',

                      'OPPTFTA', 'OPPTFT%', 'OPPTAST', 'OPPTSTL', 'OPPTBLK', 'OPPTPF']



def get_columns_mean(columns, data_frame_describe, data_frame):

    for column in columns:

        mean = data_frame_describe[column]['mean']

        data_frame[column] = round(mean, 5)

    

def get_teams_mean(game_id, home_team, away_team):

    columns = ['TEAMFGM', 'TEAMFGA', 'TEAMFG%', 'TEAM3PM', 'TEAM3PA', 'TEAM3P%', 'TEAMFTM',

           'TEAMFTA', 'TEAMFT%', 'TEAMOREB', 'TEAMDREB', 'TEAMREB', 'TEAMAST', 'TEAMTOV', 'TEAMSTL',

           'TEAMBLK', 'TEAMPF']

    

    opposite_column = ['OPPTFGM', 'OPPTFGA', 'OPPTFG%', 'OPPT3PM', 'OPPT3PA', 'OPPT3P%', 'OPPTFTM', 

            'OPPTFTA', 'OPPTFT%', 'OPPTOREB', 'OPPTDREB', 'OPPTREB', 'OPPTAST', 'OPPTTOV',  'OPPTSTL', 

            'OPPTBLK', 'OPPTPF']

        

    HOME = games_stat.loc[(games_stat['GAME_ID'] < game_id) & (games_stat['TEAMABBR'] == home_team), :][-5:]

    AWAY = games_stat.loc[(games_stat['GAME_ID'] < game_id) & (games_stat['TEAMABBR'] == away_team), :][-5:]



    HOME = HOME.filter(columns)

    AWAY = AWAY.filter(columns)



    get_columns_mean(HOME.columns, HOME.describe(), HOME)

    HOME = HOME.iloc[-1:,:]



    get_columns_mean(AWAY.columns, AWAY.describe(), AWAY)

    AWAY = AWAY.iloc[-1:,:]



    rename_column = dict()

    for i in range(len(columns)):

        rename_column[columns[i]] = opposite_column[i]

    

    AWAY.rename(columns=rename_column, inplace=True)

    

    HOME['key'] = 1

    AWAY['key'] = 1

    AWAY_HOME = pd.merge(HOME, AWAY, how='outer')

    del AWAY_HOME['key']



    AWAY_HOME = AWAY_HOME.filter(importance_columns)

    AWAY_HOME = list(AWAY_HOME.iloc[0,:])

    return AWAY_HOME
games = []

results = []



for step in range(50, len(games_stat), 2):

    team_home = games_stat.iloc[step + 1]

    team_away = games_stat.iloc[step]

    game_id = team_home['GAME_ID']

    team_home_name = team_home['TEAMABBR']

    team_away_name = team_away['TEAMABBR']

    result = team_home['TEAMRSLT']



    game = get_teams_mean(game_id, team_home_name, team_away_name)

    games.append(game)

    results.append(result)
game_data_training = pd.DataFrame(np.array(games), columns=importance_columns)

game_data_result = pd.DataFrame(np.array(results), columns=['TEAMRSLT'])

game_data_result = game_data_result['TEAMRSLT']



game_data_training.fillna(0, inplace = True)

game_data_training.isnull().sum().max()
scaler = MinMaxScaler()

scaler = scaler.fit(game_data_training)

game_data_scaled_training = pd.DataFrame(scaler.transform(game_data_training))

game_data_scaled_training.columns = game_data_training.columns



pca = PCA(n_components=15)

pca = pca.fit(game_data_training)

stats_transformed = pca.fit_transform(game_data_training)

stats_transformed.shape
x_train, x_test, y_train, y_test = train_test_split(game_data_training.to_numpy(), game_data_result.to_numpy(), test_size=0.2)



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
estim = HyperoptEstimator(algo=tpe.suggest,

                          max_evals=60,

                          trial_timeout=60)



# kaggle doesn't pass new version of Hyperopt, try clone this notebook and make it manually



# estim.fit(x_train, y_train)

# print(estim.score(x_test, y_test))

# print(estim.best_model())