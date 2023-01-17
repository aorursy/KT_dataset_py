ratings_initialization = 1500 # where to start Elo ratings

k_series = [10, 20] # how much importance to give to recent games - higher value assigns more importance

home_adv_series = [100, 200] # value of home court advantage

revert_series = [0.4, 0.6] # how much mean reversion to allow after seasons - note if revert_option = 'None', still need to have at least one value here

revert_option = ['None', 'NCAA', 'Conf'] # options for mean reversion - 'None', 'NCAA', or 'Conf'. NCAA reverts to ratings_initialization, 'Conf' revers to end of season mean for each conference

first_season = [2000, 2010] # what season to start ELO calculation in

keep_seasons = [2014, 2015, 2016, 2017, 2018] # which seasons to keep end of season ELOs, for prediction and analysis
# for data manipulation

import pandas as pd

import numpy as np 



# for tracking simulation time

import time



# for individual log loss function evaluation

import math



# for vector log loss metric evaluation

import sklearn

from sklearn import metrics



# for visualization

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
regular_season = pd.read_csv("../input/datafiles/RegularSeasonCompactResults.csv")

tourney = pd.read_csv("../input/datafiles/NCAATourneyCompactResults.csv")

teams = pd.read_csv("../input/datafiles/Teams.csv")

seeds = pd.read_csv("../input/datafiles/NCAATourneySeeds.csv")

conferences = pd.read_csv("../input/datafiles/TeamConferences.csv").rename({'TeamID':'team_id', 'Season': 'season'}, axis='columns')

predictions = pd.read_csv("../input/SampleSubmissionStage1.csv")

results = pd.read_csv("../input/datafiles/NCAATourneyCompactResults.csv")
# Create tourney data indicator in regular season and tourney data

regular_season['Tourney_Ind'] = 'N'

tourney['Tourney_Ind'] = 'Y'

# Append tourney data to regular season data

games = regular_season.append(tourney)



# Columns to help identify teams, store Elo ratings

games['Team1ID'] = games.apply(lambda r: '_'.join(map(str, [r['Season']]+sorted([r['WTeamID']]))), axis=1)

games['Team2ID'] = games.apply(lambda r: '_'.join(map(str, [r['Season']]+sorted([r['LTeamID']]))), axis=1)

games['Team1Elo'] = ratings_initialization

games['Team2Elo'] = ratings_initialization

games['margin'] = games['WScore'] - games['LScore']

#games['elo'] = None



# Create dictionary to store Elo ratings, for quick iteration

df_Team1 = pd.DataFrame({'Team': games['Team1ID'], 'Elo': games['Team1Elo']})

df_Team2 = pd.DataFrame({'Team': games['Team2ID'], 'Elo': games['Team2Elo']})

df_Team = df_Team1.append(df_Team2)

df_Team = df_Team.drop_duplicates(['Team'])

df_Team['Season'] = df_Team['Team'].map(lambda x: x.split('_')[0]).astype(int)

df_Team['Team'] = df_Team['Team'].map(lambda x: x.split('_')[1]).astype(int)

elo_dict = dict(zip(df_Team['Team'], df_Team['Elo']))



# Merge in conference abbreviations for teams

games = games.merge(conferences.rename(columns={'team_id': 'WTeamID', 'season':'Season'}), how='inner', on=['WTeamID', 'Season']).rename(columns={'ConfAbbrev': 'WTeam_ConfAbbrev'})

games = games.merge(conferences.rename(columns={'team_id': 'LTeamID', 'season':'Season'}), how='inner', on=['LTeamID', 'Season']).rename(columns={'ConfAbbrev': 'LTeam_ConfAbbrev'})
def elo_dataframe(df, team_id):

   # Creates dataframe ready to take in Elo simulations 

    d = games.copy()

    d = d.loc[(d['WTeamID'] == team_id) | (d['LTeamID'] == team_id), :].reset_index(drop=True)

    d.loc[(d['WTeamID'] == team_id), 'conference'] = d['WTeam_ConfAbbrev']

    d.loc[(d['LTeamID'] == team_id), 'conference'] = d['LTeam_ConfAbbrev']

    d.drop_duplicates(['Season'], keep='last', inplace=True)

    out = pd.DataFrame({

        'team_id': team_id,

        'season': d['Season'],

        'conference': d['conference']

    })

    return(out)



def elo_update(w_elo, l_elo, margin, k):

    ''' Calculates updated Elo rankings after taking in game results '''

    

    elo_diff = w_elo - l_elo

    pred = (1. / (10. ** (-(w_elo - l_elo) / 400.) + 1.))

    mult = math.log(abs(margin)+1)*(2.2/((w_elo-l_elo)*0.001+2.2))

    update = k * mult * (1 - pred)

    

    return(pred, update)



def calculate_elo(in_df, out_df, scenario_dict, k, home_adv, revert, option):

    ''' Iterates through all games in the dataframe, calculating Elo ratings at each point in time '''

    preds = []

    elo_df = pd.DataFrame(columns=['team_id', 'season', 'elo'])

    elo_df['team_id'] = elo_df['team_id'].astype(int)

    elo_df['season'] = elo_df['season'].astype(int)

    scenario_dict = {x:ratings_initialization for x, y in scenario_dict.items()}

                    

    for i in range(in_df.shape[0]):

            # Get key data from current row

            w = in_df.at[i, 'WTeamID']

            l = in_df.at[i, 'LTeamID']

            margin = in_df.at[i, 'margin']

            wloc = in_df.at[i, 'WLoc']

            # Does either team get a home-court advantage?

            w_ad, l_ad, = 0., 0.

            if wloc == "H":

                w_ad += home_adv

            elif wloc == "A":

                l_ad += home_adv

            # Get elo updates as a result of the game

            pred, update = elo_update(scenario_dict[w] + w_ad,

                                      scenario_dict[l] + l_ad, 

                                      margin,

                                      k)

            scenario_dict[w] += update

            scenario_dict[l] -= update

            preds.append(pred)

            # If game is last of the regular season for a given scenario, stores the elo in the elo_df dataframe, to be included in the output

            if ((i+1 in range(in_df.shape[0])) and (in_df.at[i+1, 'Tourney_Ind'] == 'Y') and (in_df.at[i, 'Tourney_Ind'] == 'N')) or (i == in_df.shape[0]):

                elo = pd.DataFrame.from_dict(scenario_dict, orient='index', columns=['elo']).reset_index().rename(columns={'index':'team_id'})

                elo['season']= in_df.at[i, 'Season']

                elo_df = elo_df.append(elo, sort=True)

               # print(elo_df.shape)

            # If a new season is started, account for some mean reversion 

            if (i+1 in range(in_df.shape[0])) and (in_df.at[i, 'Season'] < in_df.at[i+1, 'Season']):

                if option == 'NCAA':

                    scenario_dict = {x:((revert*ratings_initialization) + ((1-revert)*y)) for x, y in scenario_dict.items()}

                elif option == 'Conf':

                    conf_elo_df = elo.merge(out_df, how='inner', on=['team_id', 'season']) #brings in conferences

                    conf_elo_df = conf_elo_df[conf_elo_df['season']==in_df.at[i, 'Season']] #only keep current season values

                    conf_elo_df['conference_elo'] = conf_elo_df['elo'].groupby(conf_elo_df['conference']).transform('mean') # group elos by conference to find conference means

                    conf_elo_df['new_elo'] = (conf_elo_df['elo']*(1-revert))+(conf_elo_df['conference_elo']*revert)

                    scenario_dict.update({x:y for x,y in zip(conf_elo_df['team_id'],conf_elo_df['new_elo'])})

                else:

                    scenario_dict = scenario_dict

                               

    out_df = out_df.merge(elo_df, how='inner', on=['team_id', 'season'])   

    return(out_df)
start = time.time()



#Set up dataframes to hold Elo ratings - only setup to keep results that we can test on (2014-2018 tournaments)

games = games.sort_values(by=['Season', 'DayNum'], ascending=True).reset_index(drop=True)

seasons_keep = games[games['Season'].isin(keep_seasons)]

team_ids = set(seasons_keep['WTeamID']).union(set(seasons_keep['LTeamID']))

elo_list = [elo_dataframe(games, i) for i in team_ids]

season_elos = pd.concat(elo_list)

scenario_elos = season_elos



for season in first_season:

    # Set games dataframe to only include seasons you want to iterate over

    games_indexed = games[games['Season'] >= season]

    # Make sure games index is in order, otherwise functions won't work

    games_indexed = games_indexed.sort_values(by=['Season', 'DayNum'], ascending=True).reset_index(drop=True)

    

    #Run through scenarios

    for k in k_series:

        for home_adv in home_adv_series:

            for revert in revert_series:

                for option in revert_option:

            # Calculate Elo ratings

                    scenario = calculate_elo(in_df = games_indexed, out_df=scenario_elos, scenario_dict=elo_dict, k=k, home_adv=home_adv, revert=revert, option=option).rename({

                        'elo':'elo_'+str(season)+'_'+str(k)+'_'+str(home_adv)+'_'+str(revert)+'_'+str(option)}, axis='columns')

                    season_elos =season_elos.merge(scenario, how='inner', on=['team_id', 'season', 'conference'])

        

print("Completed in", time.time() - start, "seconds")
# Data prep

predictions['Season'] = predictions['ID'].map(lambda x: x.split('_')[0]).astype(int)

predictions['Team1'] = predictions['ID'].map(lambda x: x.split('_')[1]).astype(int)

predictions['Team2'] = predictions['ID'].map(lambda x: x.split('_')[2]).astype(int)



elos1 = season_elos.rename(columns={'team_id':'Team1', 'season': 'Season'})

elos2 = season_elos.rename(columns={'team_id':'Team2', 'season': 'Season'})



predictions = pd.merge(left=predictions, right=elos1, how='left', on=['Team1', 'Season'])

predictions = pd.merge(left=predictions, right=elos2, how='left', on=['Team2', 'Season'])
# Prediction function

def elo_pred(elo1, elo2):

    pred = 1. / (10. ** (-(np.subtract(elo1, elo2)) / 400.) + 1.)

    return pred
# Create predictions

for season in first_season:

    for k in k_series:

        for home_adv in home_adv_series:

            for revert in revert_series:

                for option in revert_option:

        # Calculate Elo ratings

                     predictions['pred_'+str(season)+'_'+str(k)+'_'+str(home_adv)+'_'+str(revert)+'_'+str(option)]= elo_pred(predictions['elo_'+str(season)+'_'+str(k)+'_'+str(home_adv)+'_'+str(revert)+'_'+str(option)+'_x'],

                             predictions['elo_'+str(season)+'_'+str(k)+'_'+str(home_adv)+'_'+str(revert)+'_'+str(option)+'_y'])
# Load in actual results

results = results.loc[results['DayNum'] > 135] 

results['Team1'] = results[['WTeamID','LTeamID']].min(axis=1)

results['Team2'] = results[['WTeamID','LTeamID']].max(axis=1)

predictions = pd.merge(predictions, results, how='inner', on=['Season','Team1','Team2'])

predictions['result'] = (predictions['WTeamID'] == predictions['Team1']).astype(int)
# Total log loss for each iteration of ELO

logloss=pd.DataFrame(columns=['name', 'log_loss'])

name = []

calc = []



for season in first_season:

    for k in k_series:

        for home_adv in home_adv_series:

            for revert in revert_series:

                for option in revert_option:

        # Calculate Elo ratings

                    name_i = str(season)+"_"+str(k)+"_"+str(home_adv)+"_"+str(revert)+"_"+str(option)

                    calc_i = sklearn.metrics.log_loss(predictions['result'], predictions['pred_'+str(season)+"_"+str(k)+'_'+str(home_adv)+'_'+str(revert)+'_'+str(option)], eps=1e-15) 

                    

                    name.append(name_i)

                    calc.append(calc_i)



logloss['name'] = name

logloss['log_loss'] = calc
# Let's only keep top 20 results, for comparison

logloss = logloss.sort_values(by='log_loss', ascending=True, axis=0)

logloss_top10 = logloss[:9]

logloss_top10
# Calculate season by season results for scenarios with top 10 total log loss

logloss_season=pd.DataFrame(columns=['name', 'season', 'log_loss']) 

name = []

season = []

calc = []



for s in predictions['Season'].unique():

    data = predictions[predictions['Season']==s]

    for scenario in logloss_top10['name'].unique(): 

        # Calculate Elo ratings

                    name_i = scenario

                    season_i = s

                    calc_i = sklearn.metrics.log_loss(data['result'], data['pred_'+str(scenario)],eps=1e-15) 

                    

                    name.append(name_i)

                    season.append(season_i)

                    calc.append(calc_i)



logloss_season['name'] = name

logloss_season['season'] = season

logloss_season['log_loss'] = calc
# Visualize season by season results of predictions

sns.lineplot(x=logloss_season['season'], y=logloss_season['log_loss'], hue=logloss_season['name'], legend=False) 
# Pick a scenario to further analyze

scenario = '2000_10_100_0.4_None'
analysis = predictions[['ID', 'Season', 'Team1', 'Team2', 'conference_x', 'conference_y', 'elo_'+scenario+'_x',

                       'elo_'+scenario+'_y', 'pred_'+scenario, 'result', 'WScore', 'LScore']]

analysis = analysis.rename(columns={'conference_x':'conference1', 'conference_y':'conference2', 

                                   'elo_'+scenario+'_x':'elo1','elo_'+scenario+'_y': 'elo2',

                                   'pred_'+scenario:'pred'})

#Bring in team names

analysis = analysis.merge(teams[['TeamID', 'TeamName']],left_on=['Team1'], right_on=['TeamID'], how='left').rename(columns={'TeamName':'TeamName1'})

analysis = analysis.merge(teams[['TeamID', 'TeamName']],left_on=['Team2'], right_on=['TeamID'], how='left').rename(columns={'TeamName':'TeamName2'})
# Calculate log loss for individual games to see where biggest errors are



def logloss(true_label, predicted, eps=1e-15):

  p = np.clip(predicted, eps, 1 - eps)

  if true_label == 1:

    return -math.log(p)

  else:

    return -math.log(1 - p)



analysis['logloss']=0

analysis['logloss']=analysis['logloss'].astype(float)

for i in range(len(analysis.index)):

    analysis['logloss'][i] = logloss(analysis['result'][i], analysis['pred'][i], eps=1e-15) 
#Create dataframe for rankings analysis

rankings = analysis[['Season', 'TeamName1', 'conference1', 'elo1']].drop_duplicates()

rankings2 = analysis[['Season', 'TeamName2', 'conference2', 'elo2']].drop_duplicates().rename(columns={'TeamName2':'TeamName1', 'conference2':'conference1', 'elo2':'elo1'})

rankings = rankings.append(rankings2, sort=True).drop_duplicates()



#Create dataframe for evaluating biggest misses

misses = analysis[['Season', 'TeamName1', 'TeamName2', 'elo1', 'elo2', 'pred', 'WScore', 'LScore', 'logloss']]
#2018 rankings

rankings[rankings['Season']==2018].sort_values('elo1', ascending=False).head(10)
#2018 biggest misses

misses[misses['Season']==2018].sort_values('logloss', ascending=False).head(10)