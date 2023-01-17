import numpy as np 
import pandas as pd 
import random
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from subprocess import check_output
## Loading events dataset
df_events = pd.read_csv("../input/football-events/events.csv")
## Loading ginf dataset, which has some important data to merge with event dataset
df_ginf = pd.read_csv("../input/football-events/ginf.csv")
## Selecting only the features I need. `id_odsp` serves as a unique identifier that will be used to 
## union the 2 datasets
df_ginf = df_ginf[['id_odsp', 'date', 'league', 'season', 'country']]
## Join the 2 datasets to add 'date', 'league', 'season', and 'country' information to the main dataset
df_events = df_events.merge(df_ginf, how='left')
## Naming the leagues with their popular names, which will make thinks much clear for us
leagues = {'E0': 'Premier League', 'SP1': 'La Liga',
          'I1': 'Serie A', 'F1': 'League One', 'D1': 'Bundesliga'}

## Apply the mapping
df_events['league'] = df_events['league'].map(leagues)
## Events type 1
event_type_1 = pd.Series([
    'Announcement',
    'Attempt',
    'Corner',
    'Foul',
    'Yellow card',
    'Second yellow card',
    'Red card',
    'Substitution',
    'Free kick won',
    'Offside',
    'Hand ball',
    'Penalty conceded'], index=[[item for item in range(0, 12)]])

## Events type 2
event_type2 = pd.Series(['Key Pass', 'Failed through ball', 'Sending off', 'Own goal'],
                       index=[[item for item in range(12, 16)]])

## Match side
side = pd.Series(['Home', 'Away'], index=[[item for item in range(1, 3)]])

## Shot place
shot_place = pd.Series([
    'Bit too high', 
    'Blocked',
    'Bottom left corner',
    'Bottom right corner',
    'Centre of the goal',
    'High and wide',
    'Hits the bar',
    'Misses to the left',
    'Misses to the right',
    'Too high',
    'Top centre of the goal',
    'Top left corner',
    'Top right corner'
], index=[[item for item in range(1, 14)]])

## Outcome of shot
shot_outcome = pd.Series(['On target', 'Off target', 'Blocked', 'Hit the bar'],
                        index=[[item for item in range(1, 5)]])
## Location of shot
location = pd.Series([
    'Attacking half',
    'Defensive half',
    'Centre of the box',
    'Left wing',
    'Right wing',
    'Difficult angle and long range',
    'Difficult angle on the left',
    'Difficult angle on the right',
    'Left side of the box',
    'Left side of the six yard box',
    'Right side of the box',
    'Right side of the six yard box',
    'Very close range',
    'Penalty spot',
    'Outside the box',
    'Long range',
    'More than 35 yards',
    'More than 40 yards',
    'Not recorded'
], 
index=[[item for item in range(1, 20)]])

## Players' body part
bodypart = pd.Series(['right foot', 'left foot', 'head'], index=[[item for item in range(1, 4)]])

## Assist method
assist_method = pd.Series(['None', 'Pass', 'Cross', 'Headed pass', 'Through ball'],
                         index=[item for item in range(0, 5)])

## Situation
situation = pd.Series(['Open play', 'Set piece', 'Corner', 'Free kick'],
                     index=[item for item in range(1, 5)])
event_type_1
## Utility function to plot bar plots with similar configuration
## this function was took from this website : 
#https://www.kaggle.com/luizhsda/football-exploratory-data-analysis-eda
def plot_barplot(data, x_ticks, x_labels, y_labels, title, color='muted'):
    sns.set_style("whitegrid")   
    ## Set a figure with custom figsize
    plt.figure(figsize=(10, 8)) 
    ## Plottin data
    ax = sns.barplot(x = [j for j in range(0, len(data))], y=data.values, palette=color)
    ## Setting ticks extracted from data indexes
    ax.set_xticks([j for j in range(0, len(data))])
    ## Set labels of the chart
    ax.set_xticklabels(x_ticks, rotation=45)
    ax.set(xlabel = x_labels, ylabel = y_labels, title = title)
    ax.plot();
    plt.tight_layout()
## Count of events occurecies
events_series = df_events['event_type'].value_counts()

## Plotting chart 
plot_barplot(events_series, event_type_1.values,
            "Event type", "Number of events", "Event types")
## Filtering out dataframe to extract attemtps which resulted in goals
df_shot_places = df_events[(df_events['event_type'] == 1) & 
                           (df_events['is_goal'] == 1)]['shot_place'].value_counts()

## Plotting the chart
plot_barplot(df_shot_places, shot_place[[3, 4, 5, 11, 12]], 'Shot places', 'Number of events',
    'Shot places resulting in goals')
## Filtering out dataframe to extract attemtps which resulted in goals
df_shot_places = df_events[(df_events['event_type'] == 1) & 
                           (df_events['is_goal'] == 0)]['shot_place'].value_counts()

## Plotting the chart
plot_barplot(df_shot_places, shot_place, 'Shot places', 'Number of events',
    'Shot places no resulting in goals')
## Copying original dataframe
df_shot_places_ed = df_events.copy()

## Grouping data by shot places
df_shot_places_ed = df_events.groupby('shot_place', as_index=False).count().sort_values('id_event',
                                                   ascending=False).dropna()

## Mapping dataframe index to shot places labels available in the dictionary file
df_shot_places_ed['shot_place'] = df_shot_places_ed['shot_place'].map(shot_place)

## Plotting the chart
plot_barplot(df_shot_places_ed['id_event'], df_shot_places_ed['shot_place'],
             'Shot places',
             'Number of events',
             'Shot places',
             'BuGn_r')
## Grouping attempts by team
grouping_by_offensive = df_events[df_events['league']=='La Liga'][df_events['is_goal']==1].groupby('event_team')

## Sorting the values
grouping_by_offensive = grouping_by_offensive.count().sort_values(by='id_event', ascending=False)[:10]
teams = grouping_by_offensive.index
scores = grouping_by_offensive['id_event']

## Plotting the teams
plot_barplot(scores, teams, 'Teams', 'Number of goals', 'Most offensive teams in La Liga','PRGn_r')
plt.savefig('offensiveteam.jpg', format='jpg', dpi=1000)
## Grouping attempts by team
grouping_by_offensive = df_events[df_events['league']=='La Liga'][df_events['is_goal']==1].groupby('event_team')

## Sorting the values
grouping_by_offensive = grouping_by_offensive.count().sort_values(by='id_event', ascending=True)[:10]
teams = grouping_by_offensive.index
scores = grouping_by_offensive['id_event']

## Plotting the teams
plot_barplot(scores, teams, 'Teams', 'Number of goals', 'Less offensive teams in La Liga', 'PRGn_r')
plt.savefig('lessoffensiveteam.jpg', format='jpg', dpi=1000)
## grouping by player when is goal
grouping_by_offensive_player = df_events[df_events['league']=='La Liga'][df_events['is_goal']==1].groupby('player')

## Couting and sorting the number of goals by player, then pick the top 10
grouping_by_offensive_player = grouping_by_offensive_player.count().sort_values(by='id_event',
                                                                                ascending=False)[:10]
## Extracting player names
players = grouping_by_offensive_player.index
## Extracting values (# of goals)
scores = grouping_by_offensive_player['id_event']

## Plotting the chart
plot_barplot(scores, players, 'Players', 'Number of Goal', 'Most offensive players in La Liga')
plt.savefig('offensiveteamplayer.jpg', format='jpg', dpi=1000)
## Loading events dataset
df_events1 = pd.read_csv("../input/football-events/events.csv")
## Loading ginf dataset, which has some important data to merge with event dataset
df_ginf1 = pd.read_csv("../input/football-events/ginf.csv")
## Selecting only the features I need. `id_odsp` serves as a unique identifier that will be used to 
## union the 2 datasets
## Join the 2 datasets to add 'date', 'league', 'season', and 'country' information to the main dataset
df_events1 = df_events1.merge(df_ginf1, how='left')
df_events1 = df_events1[['id_odsp', 'id_event', 'league', 'season', 'ht', 'at', 'event_team', 'is_goal']]
## Naming the leagues with their popular names.
leagues = {'E0': 'Premier League', 'SP1': 'La Liga',
          'I1': 'Serie A', 'F1': 'League One', 'D1': 'Bundesliga'}
## Apply the mapping
df_events1['league'] = df_events1['league'].map(leagues)

df_events1.info()
import warnings
warnings.filterwarnings('ignore')
Leagues = df_events1['league'].unique()
Seasons = list(df_events1['season'].unique())
print(Leagues)
print(Seasons)
df_events1[df_events1['league']== 'Bundesliga'][df_events1['season'] == 2012]['ht'].unique()
## grouping by player when is goal
goal = df_events[df_events['is_goal']==1]
## Plotting the hist
fig=plt.figure(figsize=(13,10))
plt.hist(goal.time, 100)
plt.xlabel("TIME (min)",fontsize=10)
plt.ylabel("Number of goals",fontsize=10)
plt.title("goal counts vs time",fontsize=15)
x=goal.groupby(by='time')['time'].count().sort_values(ascending=False).index[0]
y=goal.groupby(by='time')['time'].count().sort_values(ascending=False).iloc[0]
x1=goal.groupby(by='time')['time'].count().sort_values(ascending=False).index[1]
y1=goal.groupby(by='time')['time'].count().sort_values(ascending=False).iloc[1]
plt.text(x=x-10,y=y+10,s='time:'+str(x)+',max:'+str(y),fontsize=12,fontdict={'color':'red'})
plt.text(x=x1-10,y=y1+10,s='time:'+str(x1)+',the 2nd max:'+str(y1),fontsize=12,fontdict={'color':'black'})
plt.savefig('goals.jpg', format='jpg', dpi=1000)
plt.show() 
goal = df_events[df_events['event_team']=='Barcelona'][df_events['league']=='La Liga'][df_events['is_goal'] == 1]
shot_outcomes = {1:'On target', 2:'Off target', 3:'Blocked', 4:'Hit the bar'}
assist_methods = {0:np.nan, 1:'Pass', 2:'Cross', 3:'Headed pass', 4:'Through ball'}
situations = {1:'Open play', 2:'Set piece', 3:'Corner', 4:'Free kick'}
goal['shot_outcome'] = goal['shot_outcome'].map(shot_outcomes)
goal['assist_method'] =goal['assist_method'].map(assist_methods)
goal['situation'] = goal['situation'].map(situations)
goal1=goal.copy()
plt.subplot(2,1,1)
plt.figure(figsize=(10,8))
data1=goal1.groupby(by=['situation'])['situation'].count()
colors=["green", "red","yellow", "pink"]
plt.pie(data1,autopct='%1.1f%%',labels=data1.index,startangle=60,explode=(0,0,0,0.1))
plt.axis('equal')
plt.title("Percentage of situations for Barcelona",fontsize=15)
plt.legend(fontsize=12,loc='best')
# plt.show()
data1
goal = df_events[df_events['event_team']=='Barcelona'][df_events['league']=='La Liga']
goal['shot_outcome'] = goal['shot_outcome'].map(shot_outcomes)
goal1=goal.copy()
plt.subplot(2,1,2)
plt.figure(figsize=(10,8))
data2=goal1.groupby(by=['shot_outcome'])['shot_outcome'].count()
colors=["green", "red","yellow", "pink"]
plt.pie(data2,autopct='%1.1f%%',labels=data2.index,startangle=60,explode=(0.1,0,0,0))
plt.axis('equal')
plt.title("Percentage of shot outcome",fontsize=15)
plt.legend(fontsize=12,loc='best')
plt.show()
from matplotlib.gridspec import GridSpec
the_grid = GridSpec(1, 2)
goal = df_events[df_events['event_team']=='Barcelona'][df_events['league']=='La Liga'][df_events['is_goal'] == 1]
goal['situation'] = goal['situation'].map(situations)
goal1=goal.copy()
plt.figure(figsize=(10,8))

data1=goal1.groupby(by=['situation'])['situation'].count()
plt.subplot(the_grid[0, 0], aspect=1)
plt.pie(data1,autopct='%1.1f%%',labels=data1.index,startangle=60,explode=(0,0,0,0.1))
plt.axis('equal')
plt.title("Percentage of goals situations for Barcelona",fontsize=15)
plt.legend(fontsize=12,loc='best')

goals = df_events[df_events['event_team']=='Barcelona'][df_events['league']=='La Liga']
goals['shot_outcome'] = goals['shot_outcome'].map(shot_outcomes)
goals1=goals.copy()
data2=goals1.groupby(by=['shot_outcome'])['shot_outcome'].count()
colors=["green", "red","yellow", "pink"]
plt.subplot(the_grid[0, 1], aspect=1)
plt.pie(data2,autopct='%1.1f%%',labels=data2.index,startangle=60,explode=(0,0,0,0.1))
plt.axis('equal')
plt.title("Percentage of shot outcome for Barcelona",fontsize=15)
plt.legend(fontsize=12,loc='best')
plt.savefig('barca.jpg', format='jpg', dpi=1000)
plt.show()
the_grid = GridSpec(1, 2)
goal = df_events[df_events['event_team']=='Barcelona'][df_events['league']=='La Liga'][df_events['is_goal'] == 1]
goal['situation'] = goal['situation'].map(situations)
goal1=goal.copy()
plt.figure(figsize=(10,8))
# plt.subplot(2,1,1)

data1=goal1.groupby(by=['situation'])['situation'].count()
plt.subplot(the_grid[0, 0], aspect=1)
plt.pie(data1,autopct='%1.1f%%',labels=data1.index,startangle=60,explode=(0,0,0,0.1))
plt.axis('equal')
plt.title("Percentage of goals situations for Barcelona",fontsize=15)
plt.legend(fontsize=12,loc='best')


goal = df_events[df_events['event_team']=='Real Madrid'][df_events['league']=='La Liga'][df_events['is_goal'] == 1]
goal['situation'] = goal['situation'].map(situations)
goal1=goal.copy()
# plt.subplot(2,1,2)
# plt.figure(figsize=(10,8))
data1=goal1.groupby(by=['situation'])['situation'].count()
plt.subplot(the_grid[0, 1], aspect=1)
plt.pie(data1,autopct='%1.1f%%',labels=data1.index,startangle=60,explode=(0,0,0,0.1))
plt.axis('equal')
plt.title("Percentage of goals situations for Real Madrid",fontsize=15)
plt.legend(fontsize=12,loc='best')
plt.savefig('barcavsrealmadrid.jpg', format='jpg', dpi=1000)
plt.show()
the_grid = GridSpec(1, 2)
goal = df_events[df_events['event_team']=='Real Madrid'][df_events['league']=='La Liga'][df_events['is_goal'] == 1]
goal['situation'] = goal['situation'].map(situations)
goal1=goal.copy()
plt.figure(figsize=(10,8))

data1=goal1.groupby(by=['situation'])['situation'].count()
plt.subplot(the_grid[0, 0], aspect=1)
plt.pie(data1,autopct='%1.1f%%',labels=data1.index,startangle=60,explode=(0,0,0,0.1))
plt.axis('equal')
plt.title("Percentage of goals situations for Real Madrid",fontsize=15)
plt.legend(fontsize=12,loc='best')


goals = df_events[df_events['event_team']=='Real Madrid'][df_events['league']=='La Liga']
goals['shot_outcome'] = goals['shot_outcome'].map(shot_outcomes)
goals1=goals.copy()
data2=goals1.groupby(by=['shot_outcome'])['shot_outcome'].count()
colors=["green", "red","yellow", "pink"]
plt.subplot(the_grid[0, 1], aspect=1)
plt.pie(data2,autopct='%1.1f%%',labels=data2.index,startangle=60,explode=(0,0,0,0.1))
plt.axis('equal')
plt.title("Percentage of shot outcome for Real Madrid",fontsize=15)
plt.legend(fontsize=12,loc='best')
plt.savefig('realmadrid.jpg', format='jpg', dpi=1000)
plt.show()
data2
redCards = df_events[df_events['league']=='La Liga'][df_events['event_type'] == 6]['event_team']
## Count of events occurecies
redCards_series = redCards.value_counts().sort_values(ascending=True)[:10]

## Plotting chart 
plot_barplot(redCards_series, redCards_series.index,
            "Event_team", "Number of Red Cards", "Red Cards per team in La Liga", 'gist_earth')
plt.savefig('redcard.jpg', format='jpg', dpi=1000)
yellowCards = df_events[df_events['league']=='La Liga'][df_events['event_type'] == (4 or 5)]['event_team']
## Count of events occurecies
yellowCards_series = yellowCards.value_counts().sort_values(ascending=True)[:10]

## Plotting chart 
plot_barplot(yellowCards_series, yellowCards_series.index,
            "Event_team", "Number of yellow Cards", "Yellow Cards per team", 'gist_earth')
df_events['event_team'].unique()
def Team_strategy(team):  
    goal = df_events[df_events['is_goal']==1][df_events['event_team'] == team]
    plt.hist(goal[goal["situation"]==1]["time"],width=1,bins=100,label="Open play")   
    plt.hist(goal[goal["situation"]==2]["time"],width=1,bins=100,label="Set Piece (excluding direct FreeKick)") 
    plt.hist(goal[goal["situation"]==3]["time"],width=1,bins=100,label="Corners") 
    plt.hist(goal[goal["situation"]==4]["time"],width=1,bins=100,label="Direct Free Kick") 
    plt.xlabel("Minutes")
    plt.ylabel("Number of goals")
    plt.legend()
    plt.title("Number of goals (by situations) against Time during match for {}".format(team),fontname="Times New Roman Bold",fontsize=14,fontweight="bold")
    plt.tight_layout()
# plt.subplot(4,1,1)
team = 'Chelsea'
Team_strategy(team)
# plt.subplot(4,1,2)
team = 'Juventus'
Team_strategy(team)
# plt.subplot(4,1,3)
team = 'Barcelona'
Team_strategy(team)
# plt.subplot(4,1,4)
team = 'Real Madrid'
Team_strategy(team)
# Penalties
penalties=df_events[df_events["location"]==14]
# Check the shot place
for i in range(14):
    if sum(penalties["shot_place"]==i)==0:
        print(i)
top_left=sum(penalties["shot_place"]==12)
bot_left=sum(penalties["shot_place"]==3)
top_right=sum(penalties["shot_place"]==13)
bot_right=sum(penalties["shot_place"]==4)
centre=sum(penalties["shot_place"]==5)+sum(penalties["shot_place"]==11)
missed=sum(penalties["shot_place"]==1)+sum(penalties["shot_place"]==6)+sum(penalties["shot_place"]==7)+sum(penalties["shot_place"]==8)+sum(penalties["shot_place"]==9)+sum(penalties["shot_place"]==10)

labels_pen=["top left","bottom left","centre","top right","bottom right","missed"]
num_pen=[top_left,bot_left,centre,top_right,bot_right,missed]
colors_pen=["red", "aqua","royalblue","yellow","violet","m"]
plt.pie(num_pen,labels=labels_pen,colors=colors_pen,autopct='%1.1f%%',startangle=60,explode=(0,0,0,0,0,0.2))
plt.axis('equal')
plt.title("Percentage of each placement of penalties",fontsize=18,fontweight="bold")
fig=plt.gcf()  
fig.set_size_inches(8,6)
plt.show()
# success rate of penalties
scored_pen=penalties[penalties["is_goal"]==1]
pen_rightfoot=scored_pen[scored_pen["bodypart"]==1].shape[0]
pen_leftfoot=scored_pen[scored_pen["bodypart"]==2].shape[0]

penalty_combi=pd.DataFrame({"right foot":pen_rightfoot,"left foot":pen_leftfoot},index=["Scored"])
penalty_combi.plot(kind="bar")
plt.title("Penalties scored (Right/Left foot)",fontsize=14,fontweight="bold")
penalty_combi
def pen_stats(player):
    player_pen=penalties[penalties["player"]==player]
    right_attempt=player_pen[player_pen["bodypart"]==1]
    right_attempt_scored=right_attempt[right_attempt["is_goal"]==1].shape[0]
    right_attempt_missed=right_attempt[right_attempt["is_goal"]==0].shape[0]
    left_attempt=player_pen[player_pen["bodypart"]==2]
    left_attempt_scored=left_attempt[left_attempt["is_goal"]==1].shape[0]
    left_attempt_missed=left_attempt[left_attempt["is_goal"]==0].shape[0]
    scored=pd.DataFrame({"right foot":right_attempt_scored,"left foot":left_attempt_scored},index=["Scored"])
    missed=pd.DataFrame({"right foot":right_attempt_missed,"left foot":left_attempt_missed},index=["Missed"])
    combi=scored.append(missed)
    return combi
pen_stats("james rodriguez")
pen_stats("lionel messi")
pen_stats("cristiano ronaldo")
pen_stats('mohamed sallah')
## Filtering out events with time <= 15'
first_15 = df_events[df_events['time'] <= 15]
## Filtering out events with time between 75' and 90'
last_15 = df_events[(df_events['time'] >= 75) & (df_events['time'] <= 90)]

## Grouping by teams for the first 15'
top_10_scorer_first_15 = first_15[first_15['is_goal'] == 1].groupby('event_team').count().sort_values(by='id_event', ascending=False)

## Extracting teams from dataframe
teams = top_10_scorer_first_15.index[:10]
## Extracting number of goals 
scores = top_10_scorer_first_15['id_event'][:10]

## Plotting results
sns.set_style("whitegrid")
fig, axs = plt.subplots(ncols=2, figsize=(15, 6))
ax = sns.barplot(x = [j for j in range(0, len(scores))], y=scores.values, ax=axs[0])
ax.set_xticks([j for j in range(0, len(scores))])
ax.set_xticklabels(teams, rotation=45)
ax.set(xlabel = 'Teams', ylabel = 'Number of goals', title = 'Goals scored in the 1st 15 minutes');

## Grouping by last 15' scorers
top_10_scorer_last_15 = last_15[last_15['is_goal'] == 1].groupby('event_team').count().sort_values(by='id_event', ascending=False)[:10]

## Extracting the names of the teams
teams_last_15 = top_10_scorer_last_15.index[:10]
## Extracting the number of goals
scores_last_15 = top_10_scorer_last_15['id_event'][:10]

## Plottin the results
ax = sns.barplot(x = [j for j in range(0, len(scores_last_15))], y=scores_last_15.values, ax=axs[1])
ax.set_xticks([j for j in range(0, len(scores_last_15))])
ax.set_xticklabels(teams_last_15, rotation=45)
ax.set(xlabel = 'Teams', ylabel = 'Number of goals', title = 'Goals scored  in the last 15 minutes');
plt.savefig('Lastminutewinners.jpg', format='jpg', dpi=1000)
fig.tight_layout()
# penalties["player"].unique()
def pen_full_stats(player):
    player_pen=penalties[penalties["player"]==player]
    scored_pen=player_pen[player_pen["is_goal"]==1]
    missed_pen=player_pen[player_pen["is_goal"]==0]
    
    top_left_rightfoot=scored_pen[scored_pen["shot_place"]==12][scored_pen["bodypart"]==1].shape[0]
    top_left_leftfoot=scored_pen[scored_pen["shot_place"]==12][scored_pen["bodypart"]==2].shape[0]
    bot_left_rightfoot=scored_pen[scored_pen["shot_place"]==3][scored_pen["bodypart"]==1].shape[0]
    bot_left_leftfoot=scored_pen[scored_pen["shot_place"]==3][scored_pen["bodypart"]==2].shape[0]
    top_right_rightfoot=scored_pen[scored_pen["shot_place"]==13][scored_pen["bodypart"]==1].shape[0]
    top_right_leftfoot=scored_pen[scored_pen["shot_place"]==13][scored_pen["bodypart"]==2].shape[0]
    bot_right_rightfoot=scored_pen[scored_pen["shot_place"]==4][scored_pen["bodypart"]==1].shape[0]
    bot_right_leftfoot=scored_pen[scored_pen["shot_place"]==4][scored_pen["bodypart"]==2].shape[0]
    centre_rightfoot=scored_pen[scored_pen["shot_place"]==5][scored_pen["bodypart"]==1].shape[0]+scored_pen[scored_pen["shot_place"]==11][scored_pen["bodypart"]==1].shape[0]
    centre_leftfoot=scored_pen[scored_pen["shot_place"]==5][scored_pen["bodypart"]==2].shape[0]+scored_pen[scored_pen["shot_place"]==11][scored_pen["bodypart"]==2].shape[0]
    scored_without_recorded_loc_rightfoot=scored_pen[scored_pen["shot_place"].isnull()][scored_pen["bodypart"]==1].shape[0]
    scored_without_recorded_loc_leftfoot=scored_pen[scored_pen["shot_place"].isnull()][scored_pen["bodypart"]==2].shape[0]
    missed_rightfoot=missed_pen[missed_pen["bodypart"]==1].shape[0]
    missed_leftfoot=missed_pen[missed_pen["bodypart"]==2].shape[0]
    
    right_foot=pd.DataFrame({"Top Left Corner":top_left_rightfoot,"Bottom Left Corner":bot_left_rightfoot,"Top Right Corner":top_right_rightfoot,"Bottom Right Corner":bot_right_rightfoot,"Centre":centre_rightfoot,"Unrecorded placement":scored_without_recorded_loc_rightfoot,"Missed":missed_rightfoot},index=["Right Foot attempt"])
    left_foot=pd.DataFrame({"Top Left Corner":top_left_leftfoot,"Bottom Left Corner":bot_left_leftfoot,"Top Right Corner":top_right_leftfoot,"Bottom Right Corner":bot_right_leftfoot,"Centre":centre_leftfoot,"Unrecorded placement":scored_without_recorded_loc_leftfoot,"Missed":missed_leftfoot},index=["Left Foot attempt"])
    
    fullstats=right_foot.append(left_foot)
    fullstats=fullstats[["Top Right Corner","Bottom Right Corner","Top Left Corner","Bottom Left Corner","Centre","Unrecorded placement","Missed"]]
    return fullstats

import warnings
warnings.filterwarnings('ignore')
pen_full_stats("lionel messi")
pen_full_stats("cristiano ronaldo")
def stats(player):
    player_pen=df_events[df_events["player"]==player]
    right_attempt=player_pen[player_pen["bodypart"]==1]
    right_attempt_scored=right_attempt[right_attempt["is_goal"]==1].shape[0]
    right_attempt_missed=right_attempt[right_attempt["is_goal"]==0].shape[0]
    left_attempt=player_pen[player_pen["bodypart"]==2]
    left_attempt_scored=left_attempt[left_attempt["is_goal"]==1].shape[0]
    left_attempt_missed=left_attempt[left_attempt["is_goal"]==0].shape[0]
    head_attempt=player_pen[player_pen["bodypart"]==3]
    head_attempt_scored=head_attempt[head_attempt["is_goal"]==1].shape[0]
    head_attempt_missed=head_attempt[head_attempt["is_goal"]==0].shape[0]
    scored=pd.DataFrame({"right foot":right_attempt_scored,"left foot":left_attempt_scored, "head": head_attempt_scored},index=["Scored"])
    missed=pd.DataFrame({"right foot":right_attempt_missed,"left foot":left_attempt_missed, "head": head_attempt_missed},index=["Missed"])
    combi=scored.append(missed)
    return combi
stats('lionel messi')
stats("cristiano ronaldo")
def full_stats(player):
    player_pen=df_events[df_events["player"]==player]
    scored_pen=player_pen[player_pen["is_goal"]==1]
    missed_pen=player_pen[player_pen["is_goal"]==0]
    
    top_left_rightfoot=scored_pen[scored_pen["shot_place"]==12][scored_pen["bodypart"]==1].shape[0]
    top_left_leftfoot=scored_pen[scored_pen["shot_place"]==12][scored_pen["bodypart"]==2].shape[0]
    top_left_head = scored_pen[scored_pen["shot_place"]==12][scored_pen["bodypart"]==3].shape[0]
    
    bot_left_rightfoot=scored_pen[scored_pen["shot_place"]==3][scored_pen["bodypart"]==1].shape[0]
    bot_left_leftfoot=scored_pen[scored_pen["shot_place"]==3][scored_pen["bodypart"]==2].shape[0]
    bot_left_head = scored_pen[scored_pen["shot_place"]==3][scored_pen["bodypart"]==3].shape[0]   
    
    top_right_rightfoot=scored_pen[scored_pen["shot_place"]==13][scored_pen["bodypart"]==1].shape[0]
    top_right_leftfoot=scored_pen[scored_pen["shot_place"]==13][scored_pen["bodypart"]==2].shape[0]
    top_right_head = scored_pen[scored_pen["shot_place"]==13][scored_pen["bodypart"]==3].shape[0]
    
    bot_right_rightfoot=scored_pen[scored_pen["shot_place"]==4][scored_pen["bodypart"]==1].shape[0]
    bot_right_leftfoot=scored_pen[scored_pen["shot_place"]==4][scored_pen["bodypart"]==2].shape[0]
    bot_right_head = scored_pen[scored_pen["shot_place"]==4][scored_pen["bodypart"]==3].shape[0]
    
    centre_rightfoot=scored_pen[scored_pen["shot_place"]==5][scored_pen["bodypart"]==1].shape[0]+scored_pen[scored_pen["shot_place"]==11][scored_pen["bodypart"]==1].shape[0]
    centre_leftfoot=scored_pen[scored_pen["shot_place"]==5][scored_pen["bodypart"]==2].shape[0]+scored_pen[scored_pen["shot_place"]==11][scored_pen["bodypart"]==2].shape[0]
    centre_head = scored_pen[scored_pen["shot_place"]==5][scored_pen["bodypart"]==3].shape[0]
    
    scored_without_recorded_loc_rightfoot=scored_pen[scored_pen["shot_place"].isnull()][scored_pen["bodypart"]==1].shape[0]
    scored_without_recorded_loc_leftfoot=scored_pen[scored_pen["shot_place"].isnull()][scored_pen["bodypart"]==2].shape[0]
    scored_without_recorded_loc_head=scored_pen[scored_pen["shot_place"].isnull()][scored_pen["bodypart"]==3].shape[0]

    missed_rightfoot=missed_pen[missed_pen["bodypart"]==1].shape[0]
    missed_leftfoot=missed_pen[missed_pen["bodypart"]==2].shape[0]
    missed_head=missed_pen[missed_pen["bodypart"]==3].shape[0]
    
    right_foot=pd.DataFrame({"Top Left Corner":top_left_rightfoot,"Bottom Left Corner":bot_left_rightfoot,"Top Right Corner":top_right_rightfoot,"Bottom Right Corner":bot_right_rightfoot,"Centre":centre_rightfoot,"Unrecorded placement":scored_without_recorded_loc_rightfoot,"Missed":missed_rightfoot},index=["Right Foot attempt"])
    left_foot=pd.DataFrame({"Top Left Corner":top_left_leftfoot,"Bottom Left Corner":bot_left_leftfoot,"Top Right Corner":top_right_leftfoot,"Bottom Right Corner":bot_right_leftfoot,"Centre":centre_leftfoot,"Unrecorded placement":scored_without_recorded_loc_leftfoot,"Missed":missed_leftfoot},index=["Left Foot attempt"])
    head=pd.DataFrame({"Top Left Corner":top_left_head,"Bottom Left Corner":bot_left_head,"Top Right Corner":top_right_head,"Bottom Right Corner":bot_right_head,"Centre":centre_head,"Unrecorded placement":scored_without_recorded_loc_head,"Missed":missed_head},index=["Head attempt"])

    
    fullstats=right_foot.append(left_foot.append(head))
    fullstats=fullstats[["Top Right Corner","Bottom Right Corner","Top Left Corner","Bottom Left Corner","Centre","Unrecorded placement","Missed"]]
    return fullstats

full_stats('lionel messi')
full_stats("cristiano ronaldo")
grouped_by_player = df_events[df_events['league'] == 'La Liga'][df_events['event_type'] == 1].groupby('event_team').count()
rouped_by_player_goals = df_events[df_events['league'] == 'La Liga'][(df_events['event_type'] == 1) &
                             (df_events['is_goal'] == 1)].groupby('event_team').count()
grouped_by_player_not_goals = df_events[df_events['league'] == 'La Liga'][(df_events['event_type'] == 1) &
                             (df_events['is_goal'] == 0)].groupby('event_team').count()
threshold = grouped_by_player['is_goal'].std()
grouped_by_player_is_goal = df_events[df_events['league'] == 'La Liga'][df_events['is_goal'] == 1].groupby('event_team').count()
grouped_by_player_is_goal_filtered = grouped_by_player_is_goal
grouped_by_players_not_goal_filtered = grouped_by_player_not_goals

## Total number of attemtps
total = grouped_by_players_not_goal_filtered['id_event'] + grouped_by_player_is_goal_filtered['id_event']

## Dividing the total of attempts by the attemtps which ended up in goals
result = total/grouped_by_player_is_goal_filtered['id_event']

## Dropping NaN values
result.dropna(inplace=True)

## Sorting results
sorted_results = result.sort_values(ascending=True)

## Extracting players names
players = sorted_results[:10].index
## Extracting value of effectiveness
effectiveness = sorted_results[:10]

## Plotting results
plot_barplot(effectiveness, players, 'Teams', 
            'Number of shots needed to score a goal',
            'Most effective teams in terms of Shooting Accuracy')
## Creating a dataframe with total of attempts and total goals
result_df = pd.DataFrame({'total': total.dropna(), 'is_goal': grouped_by_player_is_goal_filtered['id_event']})
## Sorting values by total
result_df.sort_values('total', ascending=False, inplace=True)

## Setting style to dark
sns.set(style="darkgrid")

## Creating figure
f, ax = plt.subplots(figsize=(10, 6))

## Plotting chart
sns.set_color_codes("pastel")
sns.barplot(x="total",
            y=result_df.index,
            data=result_df,
            label="# of attempts", color="b")

sns.set_color_codes("muted")
sns.barplot(x='is_goal',
            y=result_df.index, 
            data=result_df,
            label="# of goals", color="b")

ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(ylabel="Teams",
       xlabel="Number of goals x attempts", title='Shooting Accuracy')

each = result_df['is_goal'].values
the_total = result_df['total'].values
x_position = 50

for i in range(len(ax.patches[:30])):
    ax.text(ax.patches[i].get_width() - x_position, ax.patches[i].get_y() +.50, 
            str(round((each[i]/the_total[i])*100, 2))+'%')
    
sns.despine(left=True, bottom=True)
f.tight_layout()
plt.savefig('ShootingAccuracy.jpg', format='jpg', dpi=1000)
