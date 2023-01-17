import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

import sys

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

%matplotlib inline



if not sys.warnoptions:

    import warnings

    warnings.simplefilter("ignore")
pd.set_option('display.max_columns', 100)



# import basic information from past regular seasons and tournaments

df_tourney_all_compact = pd.read_csv('../input/mens-machine-learning-competition-2019/Stage2DataFiles/NCAATourneyCompactResults.csv')

df_regular_all_compact = pd.read_csv('../input/mens-machine-learning-competition-2019/Stage2DataFiles/RegularSeasonCompactResults.csv')



df_regular_all_compact.head()
# import detailed information from past regular seasons and tournaments

df_tourney_all_detailed = pd.read_csv('../input/mens-machine-learning-competition-2019/Stage2DataFiles/NCAATourneyDetailedResults.csv')

df_regular_all_detailed = pd.read_csv('../input/mens-machine-learning-competition-2019/Stage2DataFiles/RegularSeasonDetailedResults.csv')



df_regular_all_detailed.head()
df_teams = pd.read_csv('../input/mens-machine-learning-competition-2019/Stage2DataFiles/Teams.csv')



# remove D1 season information

df_teams = df_teams.drop(['FirstD1Season', 'LastD1Season'], axis=1)



# add winning team names (tourney)

df_teams = df_teams.rename(columns={"TeamID": "WTeamID"})

df_tourney_all_detailed = df_tourney_all_detailed.merge(df_teams, on='WTeamID')

df_tourney_all_detailed = df_tourney_all_detailed.rename(columns={"TeamName": "WTeamName"})



# add losing team names (tourney)

df_teams = df_teams.rename(columns={"WTeamID": "LTeamID"})

df_tourney_all_detailed = df_tourney_all_detailed.merge(df_teams, on='LTeamID')

df_tourney_all_detailed = df_tourney_all_detailed.rename(columns={"TeamName": "LTeamName"})



# add winning team names (regular)

df_teams = df_teams.rename(columns={"LTeamID": "WTeamID"})

df_regular_all_detailed = df_regular_all_detailed.merge(df_teams, on='WTeamID')

df_regular_all_detailed = df_regular_all_detailed.rename(columns={"TeamName": "WTeamName"})



# add losing team names (regular)

df_teams = df_teams.rename(columns={"WTeamID": "LTeamID"})

df_regular_all_detailed = df_regular_all_detailed.merge(df_teams, on='LTeamID')

df_regular_all_detailed = df_regular_all_detailed.rename(columns={"TeamName": "LTeamName"})



df_regular_all_detailed.head()
# create stats for winning teams

df_regular_all_detailed['WFGPerc'] = df_regular_all_detailed['WFGM'] / df_regular_all_detailed['WFGA']

df_regular_all_detailed['W3Perc'] = df_regular_all_detailed['WFGM3'] / df_regular_all_detailed['WFGA3']

df_regular_all_detailed['WFTPerc'] = df_regular_all_detailed['WFTM'] / df_regular_all_detailed['WFTA']

df_regular_all_detailed['WPoss'] = 0.96 * (df_regular_all_detailed['WFGA'] - df_regular_all_detailed['WOR'] + df_regular_all_detailed['WTO'] + (.44 * df_regular_all_detailed['WFTA']))

df_regular_all_detailed['WOrtg'] = (df_regular_all_detailed['WScore'] * 100) / df_regular_all_detailed['WPoss']

df_regular_all_detailed['WDrtg'] = (df_regular_all_detailed['LScore'] * 100) / df_regular_all_detailed['WPoss']

df_regular_all_detailed['WETSPerc'] = (df_regular_all_detailed['WFGM'] + .5 * df_regular_all_detailed['WFGM3']) / df_regular_all_detailed['WFGA']

df_regular_all_detailed['WTSPerc'] = df_regular_all_detailed['WScore'] / (2 * (df_regular_all_detailed['WFGA'] + (.44 * df_regular_all_detailed['WFTA'])))



# create stats for losing Teams

df_regular_all_detailed['LFGPerc'] = df_regular_all_detailed['LFGM'] / df_regular_all_detailed['LFGA']

df_regular_all_detailed['L3Perc'] = df_regular_all_detailed['LFGM3'] / df_regular_all_detailed['LFGA3']

df_regular_all_detailed['LFTPerc'] = df_regular_all_detailed['LFTM'] / df_regular_all_detailed['LFTA']

df_regular_all_detailed['LPoss'] = 0.96 * (df_regular_all_detailed['LFGA'] - df_regular_all_detailed['LOR'] + df_regular_all_detailed['LTO'] + (.44 * df_regular_all_detailed['LFTA']))

df_regular_all_detailed['LOrtg'] = (df_regular_all_detailed['LScore'] * 100) / df_regular_all_detailed['LPoss']

df_regular_all_detailed['LDrtg'] = (df_regular_all_detailed['WScore'] * 100) / df_regular_all_detailed['LPoss']

df_regular_all_detailed['LETSPerc'] = (df_regular_all_detailed['LFGM'] + .5 * df_regular_all_detailed['LFGM3']) / df_regular_all_detailed['LFGA']

df_regular_all_detailed['LTSPerc'] = df_regular_all_detailed['LScore'] / (2 * (df_regular_all_detailed['LFGA'] + (.44 * df_regular_all_detailed['LFTA'])))
df_regular_all_avgs = pd.DataFrame()

df_regular_season_avgs = pd.DataFrame()



# create season averages for many statistics

df_regular_all_avgs['Wins'] = df_regular_all_detailed['WTeamID'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.WTeamID, df_regular_all_detailed.WTeamName]).count()

df_regular_all_avgs['Losses'] = df_regular_all_detailed['LTeamID'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.LTeamID]).count()

df_regular_all_avgs = df_regular_all_avgs.replace([np.NaN, -np.NaN], 0)

df_regular_all_avgs['TotGames'] = df_regular_all_avgs['Wins'] + df_regular_all_avgs['Losses']

df_regular_all_avgs['WinPerc'] = df_regular_all_avgs['Wins'] / df_regular_all_avgs['TotGames']



df_regular_all_avgs['WPointScoredAvg'] = df_regular_all_detailed['WScore'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.WTeamID]).mean()

df_regular_all_avgs['LPointScoredAvg'] = df_regular_all_detailed['LScore'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.LTeamID]).mean()

df_regular_all_avgs = df_regular_all_avgs.replace([np.NaN, -np.NaN], 0)

df_regular_all_avgs['SPointScoredAvg'] = df_regular_all_avgs['WPointScoredAvg'] * df_regular_all_avgs['WinPerc'] + df_regular_all_avgs['LPointScoredAvg'] * (1 - df_regular_all_avgs['WinPerc'])



df_regular_all_avgs['WPointAllowedAvg'] = df_regular_all_detailed['LScore'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.WTeamID]).mean()

df_regular_all_avgs['LPointAllowedAvg'] = df_regular_all_detailed['WScore'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.LTeamID]).mean()

df_regular_all_avgs = df_regular_all_avgs.replace([np.NaN, -np.NaN], 0)

df_regular_all_avgs['SPointAllowedAvg'] = df_regular_all_avgs['WPointAllowedAvg'] * df_regular_all_avgs['WinPerc'] + df_regular_all_avgs['LPointAllowedAvg'] * (1 - df_regular_all_avgs['WinPerc'])



df_regular_all_avgs['WTOs'] = df_regular_all_detailed['WTO'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.WTeamID]).sum()

df_regular_all_avgs['LTOs'] = df_regular_all_detailed['LTO'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.LTeamID]).sum()

df_regular_all_avgs = df_regular_all_avgs.replace([np.NaN, -np.NaN], 0)

df_regular_all_avgs['STOs'] = df_regular_all_avgs['WTOs'] + df_regular_all_avgs['LTOs']



df_regular_all_avgs['TOsPerGame'] = df_regular_all_avgs['STOs'] / df_regular_all_avgs['TotGames']



df_regular_all_avgs['WFGPercAvg'] = df_regular_all_detailed['WFGPerc'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.WTeamID]).mean()

df_regular_all_avgs['LFGPercAvg'] = df_regular_all_detailed['LFGPerc'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.LTeamID]).mean()

df_regular_all_avgs = df_regular_all_avgs.replace([np.NaN, -np.NaN], 0)

df_regular_all_avgs['SFGPercAvg'] = df_regular_all_avgs['WFGPercAvg'] * df_regular_all_avgs['WinPerc'] + df_regular_all_avgs['LFGPercAvg'] * (1 - df_regular_all_avgs['WinPerc'])



df_regular_all_avgs['W3PercAvg'] = df_regular_all_detailed['W3Perc'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.WTeamID]).mean()

df_regular_all_avgs['L3PercAvg'] = df_regular_all_detailed['L3Perc'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.LTeamID]).mean()

df_regular_all_avgs = df_regular_all_avgs.replace([np.NaN, -np.NaN], 0)

df_regular_all_avgs['S3PercAvg'] = df_regular_all_avgs['W3PercAvg'] * df_regular_all_avgs['WinPerc'] + df_regular_all_avgs['L3PercAvg'] * (1 - df_regular_all_avgs['WinPerc'])



df_regular_all_avgs['WFTPercAvg'] = df_regular_all_detailed['WFTPerc'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.WTeamID]).mean()

df_regular_all_avgs['LFTPercAvg'] = df_regular_all_detailed['LFTPerc'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.LTeamID]).mean()

df_regular_all_avgs = df_regular_all_avgs.replace([np.NaN, -np.NaN], 0)

df_regular_all_avgs['SFTPercAvg'] = df_regular_all_avgs['WFTPercAvg'] * df_regular_all_avgs['WinPerc'] + df_regular_all_avgs['LFTPercAvg'] * (1 - df_regular_all_avgs['WinPerc'])



df_regular_all_avgs['WPossAvg'] = df_regular_all_detailed['WPoss'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.WTeamID]).mean()

df_regular_all_avgs['LPossAvg'] = df_regular_all_detailed['LPoss'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.LTeamID]).mean()

df_regular_all_avgs = df_regular_all_avgs.replace([np.NaN, -np.NaN], 0)

df_regular_all_avgs['SPossAvg'] = df_regular_all_avgs['WPossAvg'] * df_regular_all_avgs['WinPerc'] + df_regular_all_avgs['LPossAvg'] * (1 - df_regular_all_avgs['WinPerc'])



df_regular_all_avgs['WOrtgAvg'] = df_regular_all_detailed['WOrtg'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.WTeamID]).mean()

df_regular_all_avgs['LOrtgAvg'] = df_regular_all_detailed['LOrtg'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.LTeamID]).mean()

df_regular_all_avgs = df_regular_all_avgs.replace([np.NaN, -np.NaN], 0)

df_regular_all_avgs['SOrtgAvg'] = df_regular_all_avgs['WOrtgAvg'] * df_regular_all_avgs['WinPerc'] + df_regular_all_avgs['LOrtgAvg'] * (1 - df_regular_all_avgs['WinPerc'])



df_regular_all_avgs['WDrtgAvg'] = df_regular_all_detailed['WDrtg'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.WTeamID]).mean()

df_regular_all_avgs['LDrtgAvg'] = df_regular_all_detailed['LDrtg'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.LTeamID]).mean()

df_regular_all_avgs = df_regular_all_avgs.replace([np.NaN, -np.NaN], 0)

df_regular_all_avgs['SDrtgAvg'] = df_regular_all_avgs['WDrtgAvg'] * df_regular_all_avgs['WinPerc'] + df_regular_all_avgs['LDrtgAvg'] * (1 - df_regular_all_avgs['WinPerc'])



df_regular_all_avgs['WETSPercAvg'] = df_regular_all_detailed['WETSPerc'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.WTeamID]).mean()

df_regular_all_avgs['LETSPercAvg'] = df_regular_all_detailed['LETSPerc'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.LTeamID]).mean()

df_regular_all_avgs = df_regular_all_avgs.replace([np.NaN, -np.NaN], 0)

df_regular_all_avgs['SETSPercAvg'] = df_regular_all_avgs['WETSPercAvg'] * df_regular_all_avgs['WinPerc'] + df_regular_all_avgs['LETSPercAvg'] * (1 - df_regular_all_avgs['WinPerc'])



df_regular_all_avgs['WTSPercAvg'] = df_regular_all_detailed['WTSPerc'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.WTeamID]).mean()

df_regular_all_avgs['LTSPercAvg'] = df_regular_all_detailed['LTSPerc'].groupby([df_regular_all_detailed.Season, df_regular_all_detailed.LTeamID]).mean()

df_regular_all_avgs = df_regular_all_avgs.replace([np.NaN, -np.NaN], 0)

df_regular_all_avgs['STSPercAvg'] = df_regular_all_avgs['WTSPercAvg'] * df_regular_all_avgs['WinPerc'] + df_regular_all_avgs['LTSPercAvg'] * (1 - df_regular_all_avgs['WinPerc'])



df_regular_all_avgs['TotPointsScored'] = df_regular_all_avgs['SPointScoredAvg'] * df_regular_all_avgs['TotGames']

df_regular_all_avgs['TotPointsAllowed'] = df_regular_all_avgs['SPointAllowedAvg'] * df_regular_all_avgs['TotGames']

df_regular_all_avgs['PointDiffAvg'] = (df_regular_all_avgs['TotPointsScored'] - df_regular_all_avgs['TotPointsAllowed']) / df_regular_all_avgs['TotGames']



df_regular_all_avgs['Pythag'] = (df_regular_all_avgs['TotPointsScored'] ** 14) / ((df_regular_all_avgs['TotPointsScored'] ** 14) + (df_regular_all_avgs['TotPointsAllowed'] ** 14))



df_regular_all_avgs.reset_index(inplace = True)

df_regular_all_avgs = df_regular_all_avgs.rename(columns={"WTeamID": "TeamID"})

df_regular_all_avgs = df_regular_all_avgs.rename(columns={"WTeamName": "TeamName"})



# just keep the entire season averages

df_regular_season_avgs = df_regular_all_avgs.drop(['WPointScoredAvg', 'LPointScoredAvg', 'WPointAllowedAvg', 'LPointAllowedAvg', 'WTOs', 'LTOs', 'WFGPercAvg', 'LFGPercAvg', 'W3PercAvg', 'L3PercAvg',

                                                   'WFTPercAvg', 'LFTPercAvg', 'WPossAvg', 'LPossAvg', 'WOrtgAvg', 'LOrtgAvg', 'WDrtgAvg', 'LDrtgAvg', 'WETSPercAvg', 'LETSPercAvg', 'WTSPercAvg', 

                                                   'LTSPercAvg'], axis=1)



# show stats for Arizona!

df_regular_season_avgs.loc[df_regular_all_avgs['TeamID'] == 1112]
# plot winning percentage vs. various other statistics

fig = plt.figure(figsize=(20, 20))

grid = plt.GridSpec(2, 2, wspace=0.25, hspace=0.25)



plt.subplot(grid[0, :1])

sns.scatterplot(x = df_regular_season_avgs['SOrtgAvg'], y = df_regular_season_avgs['WinPerc'], color='#3c7f99')

plt.tick_params(axis='both', which='both',length=0)

plt.tick_params(axis='both', which='major', labelsize=16)

plt.xlabel('Season Offensive Rating', fontsize = 16)

plt.ylabel('Win Percentage', fontsize = 16)



plt.subplot(grid[0, 1:])

sns.scatterplot(x = df_regular_season_avgs['SDrtgAvg'], y = df_regular_season_avgs['WinPerc'], color='#3c7f99')

plt.tick_params(axis='both', which='both',length=0)

plt.tick_params(axis='both', which='major', labelsize=16)

plt.xlabel('Season Defensive Rating', fontsize = 16)

plt.ylabel('Win Percentage', fontsize = 16)



plt.subplot(grid[1, :1])

sns.scatterplot(x = df_regular_season_avgs['SETSPercAvg'], y = df_regular_season_avgs['WinPerc'], color='#3c7f99')

plt.tick_params(axis='both', which='both',length=0)

plt.tick_params(axis='both', which='major', labelsize=16)

plt.xlabel('Season SETSPercAvg', fontsize = 16)

plt.ylabel('Win Percentage', fontsize = 16)



plt.subplot(grid[1, 1:])

sns.scatterplot(x = df_regular_season_avgs['STSPercAvg'], y = df_regular_season_avgs['WinPerc'], color='#3c7f99')

plt.tick_params(axis='both', which='both',length=0)

plt.tick_params(axis='both', which='major', labelsize=16)

plt.xlabel('Season STSPercAvg', fontsize = 16)

plt.ylabel('Win Percentage', fontsize = 16)



sns.set(style="whitegrid")
# create correlation heatmap to find important statistics

fig = plt.figure(figsize=(18, 18))

matrix = df_regular_season_avgs[['WinPerc', 'SPointScoredAvg', 'SPointAllowedAvg', 'TOsPerGame', 'SFGPercAvg', 

                                 'S3PercAvg', 'SFTPercAvg', 'SOrtgAvg', 'SDrtgAvg', 'SETSPercAvg', 'STSPercAvg', 'PointDiffAvg', 'Pythag']].corr()



mask = np.zeros_like(matrix, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(225, 45, as_cmap=True)



sns.heatmap(matrix, mask=mask, cmap=cmap, center=0, annot=True, square=True, linewidths=0.25, cbar_kws={'shrink': 0.25})

plt.tick_params(axis='both', which='both',length=0)

plt.tick_params(axis='both', which='major', labelsize=16);
# filter down to just the "important" statistics

df_regular_season_avgs = df_regular_season_avgs.drop(['Wins', 'Losses', 'TotGames', 'SPointScoredAvg', 'SPointAllowedAvg', 'STOs', 'TOsPerGame', 'S3PercAvg', 'SFTPercAvg', 'SPossAvg', 'SETSPercAvg', 'STSPercAvg', 'TotPointsScored', 'TotPointsAllowed'], axis=1)
df_tourney_era_avgs = pd.DataFrame()

df_regular_era_avgs = pd.DataFrame()



# keep only seasons after 2005

df_tourney_era_detailed = df_tourney_all_detailed[df_tourney_all_detailed.Season > 2005]

df_regular_era_avgs = df_regular_season_avgs[df_regular_season_avgs.Season > 2005]



# show more Arizona stats!

df_regular_era_avgs.loc[df_regular_all_avgs['TeamID'] == 1112]
# read in all past tournament seeding

df_seeds = pd.read_csv('../input/mens-machine-learning-competition-2019/Stage2DataFiles/NCAATourneySeeds.csv')

df_seeds['SeedNum'] = df_seeds['Seed'].apply(lambda x : int(x[1:3]))

df_seeds = df_seeds.drop(['Seed'], axis = 1)



# combine with past season averages

df_regular_era_avgs_seeds = pd.DataFrame()

df_regular_era_avgs_seeds = df_regular_era_avgs.merge(df_seeds, on=['TeamID', 'Season'])



# even more Arizona stats!

df_regular_era_avgs_seeds.loc[df_regular_era_avgs_seeds['TeamID'] == 1112]
# attach regular season statistics to previous tournament matchups

df_model_wins = pd.DataFrame()

df_model_losses = pd.DataFrame()



df_tourney_all_compact = df_tourney_all_compact.rename(columns={"WTeamID": "TeamID"})

df_model_wins = df_regular_era_avgs_seeds.merge(df_tourney_all_compact, on=['TeamID', 'Season'])

df_model_wins.insert(loc=len(df_model_wins.columns), column='Outcome', value=1)



df_tourney_all_compact = df_tourney_all_compact.rename(columns={"TeamID": "WTeamID"})

df_tourney_all_compact = df_tourney_all_compact.rename(columns={"LTeamID": "TeamID"})

df_model_losses = df_regular_era_avgs_seeds.merge(df_tourney_all_compact, on=['TeamID', 'Season'])

df_model_losses.insert(loc=len(df_model_losses.columns), column='Outcome', value=0)



# past Arizona tourney wins! :)

df_model_wins.loc[df_model_wins['TeamID'] == 1112]
# past Arizona tourney losses :(

df_model_losses.loc[df_model_losses['TeamID'] == 1112]
# remove extra information

df_model_wins = df_model_wins.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis = 1)

df_model_losses = df_model_losses.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis = 1)



df_model_wins = df_model_wins.rename(columns={"LTeamID": "OppTeamID"})

df_model_losses = df_model_losses.rename(columns={"WTeamID": "OppTeamID"})



# combine wins and losses, add in regular season statistics for the other team

df_model = pd.DataFrame()

df_model = df_model_wins.append(df_model_losses)



df_model = df_model.rename(columns={"TeamID": "TeamID_x"})

df_model = df_model.rename(columns={"OppTeamID": "TeamID"})

df_model = df_model.merge(df_regular_era_avgs_seeds, on=['TeamID', 'Season'])

df_model = df_model.rename(columns={"TeamID": "TeamID_y"})



# show past Arizona matchups with regular season stats for both teams, and the outcome

df_model.loc[df_model['TeamID_x'] == 1112]
# get the Colley rankings for the Massey ordinals

df_massey_ordinals = pd.read_csv('../input/mens-machine-learning-competition-2019/MasseyOrdinals/MasseyOrdinals.csv')

df_massey_ordinals = df_massey_ordinals[df_massey_ordinals.SystemName == 'COL']



# just keep the end of season ranking

df_massey_ordinals = df_massey_ordinals[df_massey_ordinals.RankingDayNum == df_massey_ordinals.RankingDayNum.max()]



# add rankings to the previous matchups

df_model = df_model.rename(columns={"TeamID_x": "TeamID"})

df_model = df_model.merge(df_massey_ordinals, on=['TeamID', 'Season'])

df_model = df_model.rename(columns={"TeamID": "TeamID_x"})

df_model = df_model.rename(columns={"OrdinalRank": "ColleyRank_x"})



df_model = df_model.rename(columns={"TeamID_y": "TeamID"})

df_model = df_model.merge(df_massey_ordinals, on=['TeamID', 'Season'])

df_model = df_model.rename(columns={"TeamID": "TeamID_y"})

df_model = df_model.rename(columns={"OrdinalRank": "ColleyRank_y"})



# drop team names and other information to get ready for training

df_model = df_model.drop(['Season', 'TeamID_x', 'TeamName_x', 'TeamID_y', 'TeamName_y', 'RankingDayNum_x', 'SystemName_x', 'RankingDayNum_y', 'SystemName_y'], axis = 1)

df_model.head()
# only use the ColleyRank feature

y = df_model.Outcome

X = df_model[['ColleyRank_x', 'ColleyRank_y']]



# 65/35 split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=1)



# tune random forest

tune_rfmodel_colley = RandomForestClassifier(random_state=1)

param_grid = { 

    'n_estimators': [50, 75, 100],

    'max_depth' : [5,6,7]

}

CV_rfmodel_colley = GridSearchCV(estimator=tune_rfmodel_colley, param_grid=param_grid, cv= 5)

CV_rfmodel_colley.fit(X_train, y_train)

print('Best rfmodel_colley:', CV_rfmodel_colley.best_params_)



# tune k-nearest neighbors

tune_knnmodel_colley = KNeighborsClassifier()

param_grid = { 

    'n_neighbors': [20, np.floor(np.sqrt(len(df_model))).astype(int), 100, 200]

}

CV_knnmodel_colley = GridSearchCV(estimator=tune_knnmodel_colley, param_grid=param_grid, cv= 5)

CV_knnmodel_colley.fit(X_train, y_train)

print('Best knnmodel_colley:', CV_knnmodel_colley.best_params_)



# tune logistic regression

tune_lrmodel_colley = LogisticRegression()

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.fit_transform(X_test)

param_grid = { 

    'penalty': ['l1', 'l2', 'none'],

    'C' : np.logspace(-2, 2, 10),

}

CV_lrmodel_colley = GridSearchCV(estimator=tune_lrmodel_colley, param_grid=param_grid, cv= 5)

CV_lrmodel_colley.fit(X_train_scaled, y_train)

print('Best lrmodel_colley:', CV_lrmodel_colley.best_params_)
# create random forest model

rfmodel_colley = RandomForestClassifier(max_depth =  6, n_estimators = 50, random_state=1)

rfmodel_colley.fit(X_train, y_train)

y_pred = rfmodel_colley.predict(X_test)

print('Training Accuracy RF:', accuracy_score(y_train, rfmodel_colley.predict(X_train)))

print('Testing Accuracy RF:', accuracy_score(y_test, y_pred))



# create k-nearest neighbors model

knnmodel_colley = KNeighborsClassifier(n_neighbors = 200)

knnmodel_colley.fit(X_train, y_train)

y_pred = knnmodel_colley.predict(X_test)

print('Training Accuracy KNN:', accuracy_score(y_train, knnmodel_colley.predict(X_train)))

print('Testing Accuracy KNN:', accuracy_score(y_test, y_pred))



# create logistic regression model

lrmodel_colley = LogisticRegression(C = 0.01, penalty = 'l2')

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)

lrmodel_colley.fit(X_train, y_train)

y_pred = lrmodel_colley.predict(X_test)

print('Training Accuracy LR:', accuracy_score(y_train, lrmodel_colley.predict(X_train)))

print('Testing Accuracy LR:', accuracy_score(y_test, y_pred))
# only all features

y = df_model.Outcome

X = df_model.drop(['Outcome'], axis = 1)



# 65/35 split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=1)



# tune random forest

tune_rfmodel_all = RandomForestClassifier(random_state=1)

param_grid = { 

    'n_estimators': [50, 75, 100],

    'max_depth' : [5,6,7]

}

CV_rfmodel_all = GridSearchCV(estimator=tune_rfmodel_all, param_grid=param_grid, cv= 5)

CV_rfmodel_all.fit(X_train, y_train)

print('Best rfmodel_all:', CV_rfmodel_all.best_params_)



# tune k-nearest neighbors

tune_knnmodel_all = KNeighborsClassifier()

param_grid = { 

    'n_neighbors': [20, np.floor(np.sqrt(len(df_model))).astype(int), 100, 200]

}

CV_knnmodel_all = GridSearchCV(estimator=tune_knnmodel_all, param_grid=param_grid, cv= 5)

CV_knnmodel_all.fit(X_train, y_train)

print('Best knnmodel_all:', CV_knnmodel_all.best_params_)



# tune logistic regression

tune_lrmodel_all = LogisticRegression()

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.fit_transform(X_test)

param_grid = { 

    'penalty': ['l1', 'l2', 'none'],

    'C' : np.logspace(-2, 2, 10)

}

CV_lrmodel_all = GridSearchCV(estimator=tune_lrmodel_all, param_grid=param_grid, cv= 5)

CV_lrmodel_all.fit(X_train_scaled, y_train)

print('Best lrmodel_all:', CV_lrmodel_all.best_params_)
# create random forest model

rfmodel_all = RandomForestClassifier(max_depth =  5, n_estimators = 50, random_state=1)

rfmodel_all.fit(X_train, y_train)

y_pred = rfmodel_all.predict(X_test)

print('Training Accuracy RF:', accuracy_score(y_train, rfmodel_all.predict(X_train)))

print('Testing Accuracy RF:', accuracy_score(y_test, y_pred))



# create k-nearest neighbors model

knnmodel_all = KNeighborsClassifier(n_neighbors = 200)

knnmodel_all.fit(X_train, y_train)

y_pred = knnmodel_all.predict(X_test)

print('Training Accuracy KNN:', accuracy_score(y_train, knnmodel_all.predict(X_train)))

print('Testing Accuracy KNN:', accuracy_score(y_test, y_pred))



# create logistic regression model

lrmodel_all = LogisticRegression(C = 0.027825594022071243, penalty = 'l2')

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)

lrmodel_all.fit(X_train, y_train)

y_pred = lrmodel_all.predict(X_test)

print('Training Accuracy LR:', accuracy_score(y_train, lrmodel_all.predict(X_train)))

print('Testing Accuracy LR:', accuracy_score(y_test, y_pred))
# choose just the 2019 season averages

df_2019_avgs_seeds = pd.DataFrame()

df_2019_avgs_seeds = df_regular_era_avgs_seeds.loc[df_regular_era_avgs_seeds['Season'] == 2019]



# get all potential matchups for the 2019 tournament using the sample submission file

df_predict = pd.read_csv('../input/mens-machine-learning-competition-2019/SampleSubmissionStage2.csv')



df_predict[['Season', 'TeamID_x', 'TeamID_y']] = df_predict['ID'].str.split('_',expand=True)

df_predict = df_predict.drop(['ID','Pred'], axis = 1)

df_predict = df_predict.astype('int64')

df_predict.head()
# start creating the 2019 matchups

df_2019_avgs_seeds = df_2019_avgs_seeds.rename(columns={"TeamID": "TeamID_x"})

df_predict = df_predict.merge(df_2019_avgs_seeds, on=['TeamID_x', 'Season'])

df_2019_avgs_seeds = df_2019_avgs_seeds.rename(columns={"TeamID_x": "TeamID_y"})

df_predict = df_predict.merge(df_2019_avgs_seeds, on=['TeamID_y', 'Season'])



# add in the season ending Colley rankings

df_massey_ordinals = pd.read_csv('../input/march-madness-analytics-2020/MDataFiles_Stage2/MMasseyOrdinals.csv')

df_massey_ordinals = df_massey_ordinals[df_massey_ordinals.SystemName == 'COL']

df_massey_ordinals = df_massey_ordinals[df_massey_ordinals.RankingDayNum == df_massey_ordinals.RankingDayNum.max()]



df_predict = df_predict.rename(columns={"TeamID_x": "TeamID"})

df_predict = df_predict.merge(df_massey_ordinals, on=['TeamID', 'Season'])

df_predict = df_predict.rename(columns={"TeamID": "TeamID_x"})

df_predict = df_predict.rename(columns={"OrdinalRank": "ColleyRank_x"})



df_predict = df_predict.rename(columns={"TeamID_y": "TeamID"})

df_predict = df_predict.merge(df_massey_ordinals, on=['TeamID', 'Season'])

df_predict = df_predict.rename(columns={"TeamID": "TeamID_y"})

df_predict = df_predict.rename(columns={"OrdinalRank": "ColleyRank_y"})



# reorganize for easy viewing

df_predict = df_predict.sort_values(by=['TeamID_x', 'TeamID_y'])

df_predict = df_predict[['WinPerc_x', 'SFGPercAvg_x', 'SOrtgAvg_x', 'SDrtgAvg_x', 'PointDiffAvg_x', 'Pythag_x', 'SeedNum_x', 'WinPerc_y', 'SFGPercAvg_y', 'SOrtgAvg_y', 'SDrtgAvg_y', 'PointDiffAvg_y', 'Pythag_y', 'SeedNum_y', 'ColleyRank_x', 'ColleyRank_y']]

X_19 = df_predict

X_19_colley = X_19[['ColleyRank_x', 'ColleyRank_y']]

X_19.head()
# create Kaggle submissions

winning_percentage_1 = rfmodel_colley.predict_proba(X_19_colley)[:,1]

df_submission = pd.read_csv('../input/mens-machine-learning-competition-2019/SampleSubmissionStage2.csv')

df_submission.Pred = winning_percentage_1

df_submission.to_csv('rfmodel_colley.csv', index=False)



winning_percentage_2 = knnmodel_colley.predict_proba(X_19_colley)[:,1]

df_submission = pd.read_csv('../input/mens-machine-learning-competition-2019/SampleSubmissionStage2.csv')

df_submission.Pred = winning_percentage_2

df_submission.to_csv('knnmodel_colley.csv', index=False)



winning_percentage_3 = lrmodel_colley.predict_proba(X_19_colley)[:,1]

df_submission = pd.read_csv('../input/mens-machine-learning-competition-2019/SampleSubmissionStage2.csv')

df_submission.Pred = winning_percentage_3

df_submission.to_csv('lrmodel_colley.csv', index=False)



winning_percentage_4 = rfmodel_all.predict_proba(X_19)[:,1]

df_submission = pd.read_csv('../input/mens-machine-learning-competition-2019/SampleSubmissionStage2.csv')

df_submission.Pred = winning_percentage_4

df_submission.to_csv('rfmodel_all.csv', index=False)



winning_percentage_5 = knnmodel_all.predict_proba(X_19)[:,1]

df_submission = pd.read_csv('../input/mens-machine-learning-competition-2019/SampleSubmissionStage2.csv')

df_submission.Pred = winning_percentage_5

df_submission.to_csv('knnmodel_all.csv', index=False)



winning_percentage_6 = lrmodel_all.predict_proba(X_19)[:,1]

df_submission = pd.read_csv('../input/mens-machine-learning-competition-2019/SampleSubmissionStage2.csv')

df_submission.Pred = winning_percentage_6

df_submission.to_csv('lrmodel_all.csv', index=False)
!pip install binarytree

!pip install bracketeer==0.2.0
# create .png brackets

from bracketeer import build_bracket



b = build_bracket(

        outputPath='rfmodel_colley.png',

        submissionPath='rfmodel_colley.csv',

        teamsPath='../input/mens-machine-learning-competition-2019/Stage2DataFiles/Teams.csv',

        seedsPath='../input/mens-machine-learning-competition-2019/Stage2DataFiles/NCAATourneySeeds.csv',

        slotsPath='../input/mens-machine-learning-competition-2019/Stage2DataFiles/NCAATourneySlots.csv',

        year=2019

)



b = build_bracket(

        outputPath='knnmodel_colley.png',

        submissionPath='knnmodel_colley.csv',

        teamsPath='../input/mens-machine-learning-competition-2019/Stage2DataFiles/Teams.csv',

        seedsPath='../input/mens-machine-learning-competition-2019/Stage2DataFiles/NCAATourneySeeds.csv',

        slotsPath='../input/mens-machine-learning-competition-2019/Stage2DataFiles/NCAATourneySlots.csv',

        year=2019

)



b = build_bracket(

        outputPath='lrmodel_colley.png',

        submissionPath='lrmodel_colley.csv',

        teamsPath='../input/mens-machine-learning-competition-2019/Stage2DataFiles/Teams.csv',

        seedsPath='../input/mens-machine-learning-competition-2019/Stage2DataFiles/NCAATourneySeeds.csv',

        slotsPath='../input/mens-machine-learning-competition-2019/Stage2DataFiles/NCAATourneySlots.csv',

        year=2019

)



b = build_bracket(

        outputPath='rfmodel_all.png',

        submissionPath='rfmodel_all.csv',

        teamsPath='../input/mens-machine-learning-competition-2019/Stage2DataFiles/Teams.csv',

        seedsPath='../input/mens-machine-learning-competition-2019/Stage2DataFiles/NCAATourneySeeds.csv',

        slotsPath='../input/mens-machine-learning-competition-2019/Stage2DataFiles/NCAATourneySlots.csv',

        year=2019

)



b = build_bracket(

        outputPath='knnmodel_all.png',

        submissionPath='knnmodel_all.csv',

        teamsPath='../input/mens-machine-learning-competition-2019/Stage2DataFiles/Teams.csv',

        seedsPath='../input/mens-machine-learning-competition-2019/Stage2DataFiles/NCAATourneySeeds.csv',

        slotsPath='../input/mens-machine-learning-competition-2019/Stage2DataFiles/NCAATourneySlots.csv',

        year=2019

)



b = build_bracket(

        outputPath='lrmodel_all.png',

        submissionPath='lrmodel_all.csv',

        teamsPath='../input/mens-machine-learning-competition-2019/Stage2DataFiles/Teams.csv',

        seedsPath='../input/mens-machine-learning-competition-2019/Stage2DataFiles/NCAATourneySeeds.csv',

        slotsPath='../input/mens-machine-learning-competition-2019/Stage2DataFiles/NCAATourneySlots.csv',

        year=2019

)