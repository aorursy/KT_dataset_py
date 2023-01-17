import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline
#connect to the SQLLite database
path = "../input/"
database = path + 'database.sqlite'

con = sqlite3.connect(database)

#check if connection works by asking for all tables
tables = pd.read_sql("""SELECT name
                        FROM sqlite_master
                        WHERE type='table';""", con)
tables
df_teams = pd.read_sql_query(
    "SELECT * FROM Team;", con)
df_teams.head()
df_teamattr = pd.read_sql_query(
    "SELECT * FROM Team_Attributes;", con)
df_teamattr.head()
df_matchscores = pd.read_sql_query(
    "SELECT id, country_id, league_id, season, match_api_id, home_team_api_id, away_team_api_id, home_team_goal, away_team_goal FROM Match;", con)
df_matchscores.head()
#query the database for players per match
df_match_players = pd.read_sql_query(
    "SELECT id, season, home_team_api_id, away_team_api_id, home_player_1, home_player_2, home_player_3, home_player_4, home_player_5, home_player_6, home_player_7, home_player_8, home_player_9, home_player_10, home_player_11, away_player_1, away_player_2, away_player_3, away_player_4, away_player_5, away_player_6, away_player_7, away_player_8, away_player_9, away_player_10, away_player_11 FROM Match;", con)
df_match_players.tail()
#get players for home_teams per season
df_players_home_season = df_match_players[['home_team_api_id', 'season', 'home_player_1', 'home_player_2', 'home_player_3', 'home_player_4', 'home_player_5', 'home_player_6', 'home_player_7', 'home_player_8', 'home_player_9', 'home_player_10', 'home_player_11']]

#pivot the table and drop the null values for players
df_players_home_season = pd.melt(df_players_home_season, id_vars=['home_team_api_id', 'season'], var_name='position', value_name='player').dropna()

#change columnname
df_players_home_season.rename(columns={'home_team_api_id': 'team_api_id'}, inplace=True)
#drop column position
df_players_home_season.drop(columns=['position'], inplace=True)
#since players play more than one match in a season we need to remover duplicates
df_players_home_season=df_players_home_season.drop_duplicates(subset=['team_api_id', 'season', 'player'], keep='first')

df_players_home_season.shape
#get players for away_teams per season
df_players_away_season = df_match_players[['away_team_api_id', 'season', 'away_player_1', 'away_player_2', 'away_player_3', 'away_player_4', 'away_player_5', 'away_player_6', 'away_player_7', 'away_player_8', 'away_player_9', 'away_player_10', 'away_player_11']]

#pivot the table and drop the null values for players
df_players_away_season = pd.melt(df_players_away_season, id_vars=['away_team_api_id', 'season'], var_name='position', value_name='player').dropna()

#change columnname
df_players_away_season.rename(columns={'away_team_api_id': 'team_api_id'}, inplace=True)
#drop column position
df_players_away_season.drop(columns=['position'], inplace=True)
#since players play more than one match in a season we need to remover duplicates
df_players_away_season=df_players_away_season.drop_duplicates(subset=['team_api_id', 'season', 'player'], keep='first')

df_players_away_season.shape
#Now combine the two to get a list of all players
df_team_season_players = df_players_home_season.append(df_players_away_season, ignore_index=True)
#remove duplicates
df_team_season_players=df_team_season_players.drop_duplicates(subset=['team_api_id', 'season', 'player'], keep='first')

df_team_season_players.head()
df_players = pd.read_sql_query(
    "SELECT * FROM Player;", con)
df_players.head()
df_playerattr = pd.read_sql_query(
    "SELECT * FROM Player_Attributes;", con)
df_playerattr.head()
df_teams.info()
df_teams[df_teams.duplicated(['team_api_id'])]
df_teamattr.info()
pd.read_sql_query("SELECT buildUpPlayDribblingClass, buildUpPlayDribbling FROM Team_Attributes WHERE buildUpPlayDribbling IS NULL GROUP BY buildUpPlayDribblingClass;", con)
df_dribbling = df_teamattr[['buildUpPlayDribblingClass', 'buildUpPlayDribbling']]
df_dribblinglittle = df_dribbling[df_dribbling.buildUpPlayDribblingClass == 'Little']
df_dribblinglittle['buildUpPlayDribbling'].describe()
df_teamattr['buildUpPlayDribbling'].fillna(30, inplace=True)
df_teamattr.info()
df_teamattr.team_api_id.value_counts().describe()
df_match_players.info()
df_matchscores.info()
df_matchscores[df_matchscores.duplicated(['match_api_id'])]
df_players.info()
df_players[df_players.duplicated(['id'])]
df_matchscores.loc[df_matchscores['home_team_goal'] - df_matchscores['away_team_goal'] > 0, 'points_home_team'] = 3
df_matchscores.loc[df_matchscores['home_team_goal'] - df_matchscores['away_team_goal'] == 0, 'points_home_team'] = 1
df_matchscores.loc[df_matchscores['home_team_goal'] - df_matchscores['away_team_goal'] < 0, 'points_home_team'] = 0
df_matchscores.loc[df_matchscores['home_team_goal'] - df_matchscores['away_team_goal'] > 0, 'points_away_team'] = 0
df_matchscores.loc[df_matchscores['home_team_goal'] - df_matchscores['away_team_goal'] == 0, 'points_away_team'] = 1
df_matchscores.loc[df_matchscores['home_team_goal'] - df_matchscores['away_team_goal'] < 0, 'points_away_team'] = 3
df_matchscores.head()
#sum points for home_team
df_points_season_home = df_matchscores.groupby(['home_team_api_id', 'season'])['points_home_team'].agg(['sum','count']).reset_index()
#rename columns
df_points_season_home.columns = ['team_api_id', 'season', 'points', 'matches']
df_points_season_home.head()
#sum points for away_team
df_points_season_away = df_matchscores.groupby(['away_team_api_id', 'season'])['points_away_team'].agg(['sum','count']).reset_index()
#rename columns
df_points_season_away.columns = ['team_api_id', 'season', 'points', 'matches']
df_points_season_away.head()
#join the 2 dataframes 
df_points_season_both = pd.concat([df_points_season_home, df_points_season_away])

#and sum points and matches for each team as home_team and away_team
df_team_season_points = df_points_season_both.groupby(['team_api_id', 'season']).sum().reset_index()

#add a column points per match to calculate the average points per match
df_team_season_points['points_per_match'] = df_team_season_points['points']/df_team_season_points['matches']

#add columns for season start en end date assuming that the seasons start on July 1st and end on June 30th. We will need these columns to compare with the team attributes later.
df_team_season_points['seasonstart'] = df_team_season_points['season'].str[:4] + '-07-01'
df_team_season_points['seasonend'] = df_team_season_points['season'].str[5:] + '-06-30'

#merge with the team dataframe to add names to the teams
df_teamname_season_points = pd.merge(df_teams[['team_api_id', 'team_long_name']] , df_team_season_points, how='right', on='team_api_id')

#sort descending by points_per_match
df_teamname_season_points.sort_values(by='points_per_match', ascending=False).head(10)
df_teamname_season_points.sort_values(by='points_per_match', ascending=True).head(10)
df_team_season_points[['points','matches','points_per_match']].describe(percentiles=[.1, .25, .5, .75, .9])
df_team_top10 = df_team_season_points[(df_team_season_points['points_per_match'] > 2.029412)]
df_team_top10.shape
#merge df_team_top10 and df_teamattr
df_teamattr_filtered = pd.merge(df_team_top10, df_teamattr, on='team_api_id', how='outer')

#filter merged df by season                  
df_teamattr_top10 = df_teamattr_filtered.loc[df_teamattr_filtered.date.between(df_teamattr_filtered.seasonstart, df_teamattr_filtered.seasonend)]

df_teamattr_top10.describe()
df_team_bottom90 = df_team_season_points[(df_team_season_points['points_per_match'] <= 2.029412)]

#merge df_team_top10 and df_teamattr
df_teamattr_filtered90 = pd.merge(df_team_bottom90, df_teamattr, on='team_api_id', how='outer')

#filter merged df by season                  
df_teamattr_bottom90 = df_teamattr_filtered90.loc[df_teamattr_filtered90.date.between(df_teamattr_filtered90.seasonstart, df_teamattr_filtered90.seasonend)]

df_teamattr_bottom90.describe()
df_teamattr_top10_means = df_teamattr_top10[['buildUpPlaySpeed',
 'buildUpPlayDribbling',
 'buildUpPlayPassing',
 'chanceCreationPassing',
 'chanceCreationCrossing',
 'chanceCreationShooting',
 'defencePressure',
 'defenceAggression',
 'defenceTeamWidth']].mean()
df_teamattr_top10_means
df_teamattr_bottom90_means = df_teamattr_bottom90[['buildUpPlaySpeed',
 'buildUpPlayDribbling',
 'buildUpPlayPassing',
 'chanceCreationPassing',
 'chanceCreationCrossing',
 'chanceCreationShooting',
 'defencePressure',
 'defenceAggression',
 'defenceTeamWidth']].mean()
#del df_teamattr_bottom90_means[0:6]
#df_teamattr_bottom90_means.drop(df_teamattr_bottom90_means.columns[[1]], axis=1, inplace=True)
df_teamattr_bottom90_means
#create labels from the column headers
labels = ['buildUpPlaySpeed',
 'buildUpPlayDribbling',
 'buildUpPlayPassing',
 'chanceCreationPassing',
 'chanceCreationCrossing',
 'chanceCreationShooting',
 'defencePressure',
 'defenceAggression',
 'defenceTeamWidth']

#create the radar chart
stats10=df_teamattr_top10_means.values
stats90=df_teamattr_bottom90_means.values

angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
# close the plot
stats10=np.concatenate((stats10,[stats10[0]]))
stats90=np.concatenate((stats90,[stats90[0]]))
angles=np.concatenate((angles,[angles[0]]))

fig=plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, stats10, 'o-', linewidth=2, label='top 10%')
ax.fill(angles, stats10, alpha=0.25)
ax.plot(angles, stats90, 'o-', linewidth=2, label='rest 90%')
ax.fill(angles, stats90, alpha=0.25)
ax.set_thetagrids(angles * 180/np.pi, labels)
ax.set_title('Team Attributes for top 10% teams compared to rest')
ax.grid(True)
ax.legend();
#get the team attributes for FC Porto for season 2010/2011
df_teamattr_fcporto = df_teamattr_top10[(df_teamattr_top10['team_api_id'] == 9773) & (df_teamattr_top10['season'] == '2010/2011')]
df_teamattr_fcporto
#filter team attributes FC Porto
df_teamattr_fcporto_means = df_teamattr_fcporto[['buildUpPlaySpeed',
 'buildUpPlayDribbling',
 'buildUpPlayPassing',
 'chanceCreationPassing',
 'chanceCreationCrossing',
 'chanceCreationShooting',
 'defencePressure',
 'defenceAggression',
 'defenceTeamWidth']].mean()
df_teamattr_fcporto_means
#get the team attributes for Willem II for season 2010/2011
df_teamattr_willemii = df_teamattr_bottom90[(df_teamattr_bottom90['team_api_id'] == 8525) & (df_teamattr_bottom90['season'] == '2010/2011')]
df_teamattr_willemii
#filter team attributes Willem II
df_teamattr_willemii_means = df_teamattr_willemii[['buildUpPlaySpeed',
 'buildUpPlayDribbling',
 'buildUpPlayPassing',
 'chanceCreationPassing',
 'chanceCreationCrossing',
 'chanceCreationShooting',
 'defencePressure',
 'defenceAggression',
 'defenceTeamWidth']].mean()
df_teamattr_willemii_means
#create the radar chart
stats_fcporto=df_teamattr_fcporto_means.values
stats_willemii=df_teamattr_willemii_means.values

angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
# close the plot
stats_fcporto=np.concatenate((stats_fcporto,[stats_fcporto[0]]))
stats_willemii=np.concatenate((stats_willemii,[stats_willemii[0]]))
angles=np.concatenate((angles,[angles[0]]))

fig=plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, stats_fcporto, 'o-', linewidth=2, label='FC Porto')
ax.fill(angles, stats_fcporto, alpha=0.25)
ax.plot(angles, stats_willemii, 'o-', linewidth=2, label='Willem II')
ax.fill(angles, stats_willemii, alpha=0.25)
ax.set_thetagrids(angles * 180/np.pi, labels)
ax.set_title('Team Attributes for FC Porto season 2010/2011 vs Willem II season 2010/2011')
#ax.set_title2('volgende regel')
plt.suptitle('Comparing most and least effective team')
ax.grid(True)
ax.legend();
#merge df_team_top10 and df_team_season_players
df_players_top10 = pd.merge(df_team_top10[['team_api_id', 'season']], df_team_season_players, on=['team_api_id', 'season'], how='outer')
df_players_top10.head()
#merge with player to get players birthdate height and weight
df_players_top10bhw = pd.merge(df_players_top10, df_players[['player_api_id','birthday','height','weight']], how='outer', left_on=['player'], right_on=['player_api_id'])
#calculate age of player for season on December 31st
df_players_top10bhw['birthday'] = pd.to_datetime(df_players_top10bhw['birthday'])
df_players_top10bhw['age'] = df_players_top10bhw['season'].str.slice(0, 4).astype(int) - df_players_top10bhw['birthday'].dt.year

df_players_top10bhw.tail()
#to merge the attributes dataframe with the df_players_top10 dataframe we need to add a column season for each attributes row based on date
#first set date to datetime object to manipulate
df_playerattr['date'] = pd.to_datetime(df_playerattr['date'])
df_playerattr['season'] = np.where(df_playerattr['date'].dt.month>6, df_playerattr['date'].dt.year.astype(str) + '/' + (df_playerattr['date'].dt.year + 1).astype(str) , (df_playerattr['date'].dt.year - 1).astype(str) + '/' + (df_playerattr['date'].dt.year).astype(str))
df_playerattr.head()
#merge df_players_top10 and player attributes on player and season
df_playersattr_top10 = pd.merge(df_players_top10bhw, df_playerattr, how='left', left_on=['player', 'season'], right_on=['player_api_id', 'season'])

df_playersattr_top10.head()
#list all columns
labels = df_playersattr_top10.columns.values
labels
#columns to drop: we will only keep the players attributes including age height and weight
columns_to_drop = labels[[0,1,2,3,4,8,9,10,11]]
columns_to_drop
df_playersattr_top10 = df_playersattr_top10.drop(columns=columns_to_drop)
df_playersattr_top10.head()

#calculate the mean player attributes for the top 10% effective teams
df_playersattr_top10_condensed = df_playersattr_top10.mean()
df_playersattr_top10_condensed

#first we need to add birthdate, height and weight to player attributes
df_playerattrbhw = pd.merge(df_players[['player_api_id','birthday', 'height', 'weight']], df_playerattr, how='right', on='player_api_id')
df_playerattrbhw.head()
#then we need to calculate age
df_playerattrbhw['birthday'] = pd.to_datetime(df_playerattrbhw['birthday'])
df_playerattrbhw['age'] = df_playerattrbhw['season'].str.slice(0, 4).astype(int) - df_playerattrbhw['birthday'].dt.year
df_playerattrbhw.head() 
labels2 = df_playerattrbhw.columns.values
labels2
#columns to drop: we will only keep the players attributes including age height and weight
columns_to_drop2 = labels2[[0,1,4,5,6]]
columns_to_drop2
df_playerattrbhw = df_playerattrbhw.drop(columns=columns_to_drop2)
df_playerattrbhw.head()
#move column age to 3d column
df_playerattrbhw = df_playerattrbhw[['height', 'weight', 'age', 'overall_rating', 'potential',
       'preferred_foot', 'attacking_work_rate', 'defensive_work_rate',
       'crossing', 'finishing', 'heading_accuracy', 'short_passing',
       'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
       'agility', 'reactions', 'balance', 'shot_power', 'jumping',
       'stamina', 'strength', 'long_shots', 'aggression', 'interceptions',
       'positioning', 'vision', 'penalties', 'marking', 'standing_tackle',
       'sliding_tackle', 'gk_diving', 'gk_handling', 'gk_kicking',
       'gk_positioning', 'gk_reflexes', 'season']]
df_playerattrbhw.head()
#calculate the mean player attributes BEWARE this takes a lot of time
df_playersattr_condensed = df_playerattrbhw.mean()
df_playersattr_condensed

#compare player attributesfor top 10% and all teams
d = {'top10': df_playersattr_top10_condensed, 'all': df_playersattr_condensed}
df_playersattr_compared = pd.DataFrame(data=d)
#calculate delta in percentage
df_playersattr_compared['delta_perc'] = ((df_playersattr_compared['top10'] - df_playersattr_compared['all'])/df_playersattr_compared['all'])*100
df_playersattr_compared = df_playersattr_compared.sort_values(by='delta_perc', ascending=False).reset_index()
df_playersattr_compared
df_playersattr_compared = df_playersattr_compared.sort_values(by='delta_perc', ascending=True).reset_index()
df_playersattr_compared['positive'] = df_playersattr_compared['delta_perc'] > 0

data = df_playersattr_compared
group_data = df_playersattr_compared['delta_perc']
group_names = df_playersattr_compared['index']
order = df_playersattr_compared.index.values
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize=(10, 20))
ax.barh(order, group_data, color=df_playersattr_compared.positive.map({True: '#99CC99', False: '#F4874B'}))
ax.set(xlim=[-8, 4])
ax.set_yticklabels(labels=df_playersattr_compared['index'])
ax.set_yticks(np.arange(0, 38, step=1))

plt.xlabel('difference in percentage of top 10 team players to all players', size = 20)
plt.ylabel('Player Attributes', size = 20)
plt.title('Mean score of top 10% team players compared to all players', size = 20)
plt.xticks(size = 16)
plt.yticks(size = 16);
