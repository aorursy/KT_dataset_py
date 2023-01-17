# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
countries=pd.read_csv('/kaggle/input/soccer-data/Soccer Data/country.csv')
leagues=pd.read_csv('/kaggle/input/soccer-data/Soccer Data/league.csv')
matches=pd.read_csv('/kaggle/input/soccer-data/Soccer Data/match.csv')
players=pd.read_csv('/kaggle/input/soccer-data/Soccer Data/player.csv')
player_attributes=pd.read_csv('/kaggle/input/soccer-data/Soccer Data/player_attributes.csv')
teams=pd.read_csv('/kaggle/input/soccer-data/Soccer Data/team.csv')
team_attributes=pd.read_csv('/kaggle/input/soccer-data/Soccer Data/team_attributes.csv')

print(countries.shape) #(11, 2)
print(leagues.shape) #(11, 3)
print(matches.shape) #(25979, 115)
print(players.shape) #(11060, 7)
print(player_attributes.shape) #(183978, 42)
print(teams.shape) #(299, 5)
print(team_attributes.shape) #(1458, 25)





#missing values
print('missing values in: \n')
print('countries: \n', countries.isnull().sum(),'\n')
print('leagues: \n',leagues.isnull().sum(),'\n')
print('matches: \n',matches.isnull().sum(),'\n')
print('players: \n',players.isnull().sum(),'\n')
print('player_attributes: \n',player_attributes.isnull().sum(),'\n')
print('teams: \n', teams.isnull().sum(),'\n')
print('team_attributes: \n', team_attributes.isnull().sum(),'\n')

print('=========================col with missing values=================================')
##all collumns with missing values
missing_val_colM=[(index,value) for index, value in matches.isnull().sum().items() if value>0]
missing_val_colP=[(index,value) for index, value in player_attributes.isnull().sum().items() if value>0]
missing_val_colT=[(index,value) for index, value in teams.isnull().sum().items() if value>0]
missing_val_colTA=[(index,value) for index, value in team_attributes.isnull().sum().items() if value>0]
print('in matches: \n', missing_val_colM)
print('in players: \n', missing_val_colP)
print('in teams: \n', missing_val_colT)
print('in team_attributes: \n', missing_val_colTA)
#process missing values


print('-------------------------------------------------------------------------------------')
##fill missing odds with mean odds from all non null odds for the match
for col in ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 
            'LBH', 'LBD', 'LBA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'SJH', 'SJD', 
            'SJA', 'VCH', 'VCD', 'VCA', 'GBH', 'GBD', 'GBA', 'BSH', 'BSD', 'BSA']:
    odds_mean= matches[col].sum()/ matches[col].count()
    matches[col]=matches[col].fillna(odds_mean)
    
##fill missing overall_rating and potential with mean in player_attributes
rating_mean=player_attributes['overall_rating'].sum()/player_attributes['overall_rating'].count()
player_attributes['overall_rating']=player_attributes['overall_rating'].fillna(rating_mean)
print(player_attributes['overall_rating'].isna().sum())

potential_mean=player_attributes['potential'].sum()/player_attributes['potential'].count()
player_attributes['potential']=player_attributes['potential'].fillna(rating_mean)
print(player_attributes['potential'].isna().sum())

##

#duplicated values
duplicateMatches = matches[matches.duplicated()]
duplicatePlayers = players[players.duplicated()]
duplicatePlayer_attributes = player_attributes[player_attributes.duplicated()]
duplicateTeams = teams[teams.duplicated()]
duplicateTeam_attributes = team_attributes[team_attributes.duplicated()]

Mcols = duplicateMatches.columns.tolist()
Pcols = duplicatePlayers.columns.tolist()
PAcols = duplicatePlayer_attributes.columns.tolist()
Tcols = duplicateTeams.columns.tolist()
TAcols = duplicateTeam_attributes.columns.tolist()

print('duplicated rows in: \n', 'Matches: \n', [cols for cols in Mcols if len(cols)>=0], '\n',
     'Players: \n', Pcols, '\n',
     'Player_attributes: \n', PAcols, '\n',
     'Teams: \n', Tcols, '\n',
     'Team_attributes: \n', TAcols)
#NO DUPLICATES
#merging dataframes

##countries with leagues
countries_leagues = countries.merge(leagues,left_on="id",right_on="id",how="outer")
countries_leagues = countries_leagues.drop("id",axis = 1)
countries_leagues = countries_leagues.rename(columns={'name_x':"country", 'name_y':"league"})
print('countries_leagues: \n',
      countries_leagues, '\n')

##subsetted matches with leagues
matchessub= matches[['id', 'country_id', 'league_id', 'season', 'stage', 'date',
                   'match_api_id', 'home_team_api_id', 'away_team_api_id',
                    'home_team_goal', 'away_team_goal']]
matchessub = matchessub.drop("id",axis=1)
matches_leagues = matchessub.merge(countries_leagues,left_on="country_id",right_on="country_id",how="outer")

print('--------------------------------------------------------------------------------------------------------')

##players and player attributes
players_n_attributes = player_attributes.merge(players,left_on="player_api_id", right_on="player_api_id", how="outer")
players_n_attributes = players_n_attributes.drop("player_fifa_api_id_y", axis =1)
players_n_attributes = players_n_attributes.drop("id_x", axis =1)
players_n_attributes = players_n_attributes.rename(columns={"player_fifa_api_id_x":"player_fifa_api_id", "id_y":"pid"})
##teams and team attributes
teams_n_attributes = team_attributes.merge(teams,left_on="team_api_id", right_on="team_api_id", how="outer")
teams_n_attributes = teams_n_attributes.drop("id_x", axis =1)
teams_n_attributes = teams_n_attributes.drop("team_fifa_api_id_y", axis =1)
teams_n_attributes = teams_n_attributes.rename(columns={"team_fifa_api_id_x":"team_fifa_api_id", "id_y":"tid"})

print('players_n_attributes: \n',
        players_n_attributes,'\n--------------------------------------------------------------------------------\n',
     'teams_n_attributes: \n',
        teams_n_attributes)

print('--------------------------------------------------------------------------------------------------------')
##matches_leagues with teams
matches_leagues_teams=matches_leagues.merge(teams, left_on="home_team_api_id", right_on="team_api_id", how="outer")
matches_leagues_teams=matches_leagues_teams.drop("id", axis=1)
matches_leagues_teams=matches_leagues_teams.drop("team_fifa_api_id", axis=1)
matches_leagues_teams=matches_leagues_teams.merge(teams, left_on="away_team_api_id", right_on="team_api_id", how="outer")
matches_leagues_teams=matches_leagues_teams.drop("id", axis=1)
matches_leagues_teams=matches_leagues_teams.drop("team_fifa_api_id", axis=1)
matches_leagues_teams=matches_leagues_teams.drop("team_api_id_x", axis=1)
matches_leagues_teams=matches_leagues_teams.drop("team_api_id_y", axis=1)
matches_leagues_teams=matches_leagues_teams.rename(columns={"team_long_name_x":"home_team_long","team_short_name_x":"home_team_short",
                                     "team_long_name_y":"away_team_long","team_short_name_y":"away_team_short"})

print('matches_leagues_teams: \n',
     matches_leagues_teams)
matches_leagues_teams.head()
countries_leagues.head()
players_n_attributes.head()
teams_n_attributes=teams_n_attributes.loc[:1457,:]
teams_n_attributes.head()
matches_leagues_teams.head()
#top scoring/conceding teams in all leagues (by season)

aggr_homescoring_teams=matches_leagues_teams.groupby(["home_team_long"])['home_team_goal'].sum().reset_index()
aggr_awayscoring_teams=matches_leagues_teams.groupby(["away_team_long"])['away_team_goal'].sum().reset_index()
aggr_homeconceding_teams= matches_leagues_teams.groupby(["home_team_long"])['away_team_goal'].sum().reset_index()
aggr_awayconceding_teams= matches_leagues_teams.groupby(["away_team_long"])['home_team_goal'].sum().reset_index()

#top home/away scoring teams
topscoringhome=aggr_homescoring_teams.sort_values(by="home_team_goal",ascending= False)
topscoringaway=aggr_awayscoring_teams.sort_values(by="away_team_goal",ascending= False)


#top home/away conceding teams
topconcedinghome=aggr_homeconceding_teams.sort_values(by="away_team_goal",ascending= False)
topconcedingaway=aggr_awayconceding_teams.sort_values(by="home_team_goal",ascending= False)

#top scoring teams
topscoring_total=aggr_homescoring_teams.merge(aggr_awayscoring_teams,left_on="home_team_long", right_on="away_team_long", how="outer")
topscoring_total['total_goals']=topscoring_total['home_team_goal']+topscoring_total['away_team_goal']
topscoring_total=topscoring_total.sort_values(by='total_goals',ascending = False)
topscoring_total.head()

#top conceding teams
topconceding_total=aggr_homeconceding_teams.merge(aggr_awayconceding_teams,left_on="home_team_long", right_on="away_team_long", how="outer")
topconceding_total['total_goals']=topconceding_total['home_team_goal']+topconceding_total['away_team_goal']
topconceding_total=topconceding_total.sort_values(by='total_goals',ascending = False)
topconceding_total.head()

#least scoring teams
leastscoring_total=topscoring_total.sort_values(by='total_goals',ascending = True)

#least conceding teams
topconceding_total=topconceding_total.sort_values(by='total_goals',ascending = True)

#by season-add "season" in groupby
matches.head()
#merge team with players through matches

#subsetted player attributes
#using only the latest attributes of players
print(player_attributes['date'].tail(300))
player_attributes=player_attributes.sort_values(by="date", ascending=False)
player_attributes=player_attributes.drop_duplicates(subset=['player_api_id'], keep='first')
print(player_attributes.shape)
print(matches['date'].nunique())

player_attrisub=player_attributes[['player_api_id','overall_rating','potential']]
teamssub=teams[['team_api_id','team_long_name']]

#subsetted matches with season,date, home, away teams and their players
matchessub_home=matches[['season','date','home_team_api_id','home_player_1','home_player_2','home_player_3','home_player_4',
                    'home_player_5','home_player_6','home_player_7','home_player_8','home_player_9','home_player_10','home_player_11']]
matchessub_away=matches[['season','date','away_team_api_id','away_player_1','away_player_2','away_player_3','away_player_4',
                    'away_player_5','away_player_6','away_player_7','away_player_8','away_player_9','away_player_10','away_player_11']]                

col_names = {x: y for x, y in zip(matchessub_away, matchessub_home)}
team_rosters = matchessub_home.append(matchessub_away.rename(columns=col_names))
team_rosters=team_rosters.rename(columns={'season':'season','date':'date','home_team_api_id':
                                                                        'team_api_id','home_player_1':'player_1',
                                                                        'home_player_2':'player_2',
                                                                        'home_player_3':'player_3','home_player_4':'player_4',
                    'home_player_5':'player_5','home_player_6':'player_6','home_player_7':'player_7','home_player_8':'player_8',
                                                                        'home_player_9':'player_9','home_player_10':'player_10',
                                                                        'home_player_11':'player_11'})




#team_rosters with attributes of players added
team_rosters_wattr=team_rosters
team_rosters_wattr['ratingsum']=0
team_rosters_wattr['potentialsum']=0

"""for index, value in team_rosters_wattr.iterrows():
    for i in range(1,12):
        team_rosters_wattr.loc[index,'ratingsum']+=player_attrisub.loc[pindex,'overall_rating' lambda pindex=pd.Index]
        team_rosters_wattr=team_rosters_wattr.merge(player_attrisub, left_on="player_"+str(i),right_on="player_api_id", how='left')  
        team_rosters_wattr=team_rosters_wattr.drop('player_api_id', axis=1)
        team_rosters_wattr["overall_rating"] =  team_rosters_wattr["overall_rating"].fillna(0)
        team_rosters_wattr["potential"] =  team_rosters_wattr["potential"].fillna(0)

        print(i)
        print( team_rosters_wattr["ratingsum"])
        #print(team_rosters_wattr['overall_rating'])


        #team_rosters_wattr.loc[:,'ratingsum']+=1
        #team_rosters_wattr.assign(ratingsum=lambda x: x.ratingsum+1)
        team_rosters_wattr.ratingsum+=team_rosters_wattr.overall_rating
        team_rosters_wattr['potentialsum']+=team_rosters_wattr['potential']

        team_rosters_wattr=team_rosters_wattr.drop('potential', axis=1)
        team_rosters_wattr=team_rosters_wattr.drop('overall_rating', axis=1)
    """
    
#print(team_rosters_wattr)


#add match outcome to matches

def get_outcome(hgoal,agoal):
    if hgoal>agoal:
           return 'hwin'
    elif hgoal<agoal:
           return 'awin'
    else:
           return 'draw'

outcome=[get_outcome(hgoal,agoal) for hgoal, agoal in zip(matches['home_team_goal'], matches['away_team_goal'])]

matches['outcome']=outcome
print(matches.outcome)


#betting odds
betcol=['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'SJH', 'SJD', 'SJA', 'VCH', 'VCD', 'VCA', 'GBH', 'GBD', 'GBA', 'BSH', 'BSD', 'BSA']
betting_odds=matches[betcol]

print(betting_odds.min().min())
print(betting_odds.max().max())

s1=betting_odds.loc[0,].copy()
for i in range(1,betting_odds.shape[0]):
    #s1=pd.DataFrame(np.hstack((s1.values, betting_odds.loc[i,].values))  , index=np.hstack((s1.index.values, betting_odds.loc[i,].index.values)))
    s1=pd.concat([s1, betting_odds.loc[i,]])
print(s1) 

fig=plt.figure(figsize=(10,8))
sns.distplot(s1, kde=True, hist=True).set_title('distribution for all match odds')
pd.DataFrame( np.hstack((s1.values, s2.values))  , index=np.hstack((s1.index.values, s2.index.values)))

#add probability(1/odds)
for col in betting_odds.columns:
    betting_odds[col+'_prob']=1/betting_odds[col]
    
#add match outcome
betting_odds = pd.concat([betting_odds,matches['outcome']], axis = 1)

#plot distribution of probability
fig, axs = plt.subplots(nrows=5,ncols=6, figsize=(16,20))
fig.tight_layout()

i=0
j=0

for i in range(5):
    for j in range(6):
        col=betcol[j+6*i]
        sns.distplot(betting_odds[col+'_prob'], kde=True, ax=axs[i,j], hist=True)

    
fig.suptitle('Distribution of betting odds from all bookmakers');
#correlation between odds/probability and match outcome
corr_cols=betting_odds.copy()
for col in betcol:
    corr_cols=corr_cols.drop(col,axis=1)
       
cat_cols=np.array(corr_cols['outcome'])

# integer encode outcome
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(cat_cols)
print(integer_encoded)
# binary encode outcome
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
corr_cols['outcome']=onehot_encoded

#corr df
corr=corr_cols.corr("pearson")
print(corr)

#heatmap on betting odds data
plt.figure(figsize=(32,32))
sns.heatmap(corr,
            vmin=-1,
            cmap='coolwarm',
            annot=True
           )



outcome=corr.loc['outcome',]
#print(outcome)

for col in outcome.index:
    outcome[col]=abs(outcome[col])
    
#ranked correlation of odds to outcome by homew/awayw/draw 
corr_homeodds=outcome.loc[[col for col in corr.columns if col[-6]=='H']].sort_values(ascending=False)


corr_awayodds=outcome.loc[[col for col in corr.columns if col[-6]=='A']].sort_values(ascending=False)


corr_drawodds=outcome.loc[[col for col in corr.columns if col[-6]=='D']].sort_values(ascending=False)



##visualization of above

    
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
fig.suptitle('Bookmakers ranked by odds correlation with outcome')

corr_homeodds.plot.bar(ax=ax1, color='blue')
corr_awayodds.plot.bar(ax=ax2, color='orange')
corr_drawodds.plot.bar(ax=ax3, color='red')
#corr_homeodds_sorted=outcome.sort_values(by='total_goals',ascending = True)

#
corr_oddsaggr=corr_homeodds.copy()
for col in corr_homeodds.index:
    corr_oddsaggr[col[0:-6]+'_aggr']=corr_homeodds[col]+corr_awayodds[col[0:-6]+'A_prob']+corr_drawodds[col[0:-6]+'D_prob']

for col in corr_oddsaggr.index:
    if col[-4:]=='prob':
        corr_oddsaggr=corr_oddsaggr.drop([col])
        
#ranked aggregated correlation of odds with outcome
corr_oddsaggr=corr_oddsaggr.sort_values(axis=0, ascending=False)
print(corr_oddsaggr)

fig=plt.figure(figsize=(10,5))
fig.suptitle('Top bookmakers ranked by aggregated correlation between odds and match outcome ')
corr_oddsaggr.plot.bar()
        
#Bookmakers groupby correlations between them
corr_09pair=[]
corr_08pair=[]
corr_07pair=[]
corr_06pair=[]
corr_05pair=[]
corr_01pair=[]
#corr=corr.drop('outcome')
#corr=corr.drop('outcome', axis=1)

for col in corr.columns:
    for row in corr.index:
        if (abs(corr.loc[row, col])==1):
            pass
        elif (abs(corr.loc[row, col])>=0.9):
            corr_09pair.append([col,row, corr.loc[row,col]])
        elif (abs(corr.loc[row, col])>=0.8):
            corr_08pair.append([col,row, corr.loc[row,col]])
        elif (abs(corr.loc[row, col])>=0.7):
            corr_07pair.append([col,row, corr.loc[row,col]])
        elif (abs(corr.loc[row, col])>=0.6):
            corr_06pair.append([col,row, corr.loc[row,col]])
        elif (abs(corr.loc[row, col])>=0.5):
            corr_05pair.append([col,row, corr.loc[row,col]])
        elif (abs(corr.loc[row, col])<=0.1):
            corr_01pair.append([col,row, corr.loc[row,col]])

for pair in corr_08pair:
    print(pair)
            
for pair in corr_01pair:
    print(pair)