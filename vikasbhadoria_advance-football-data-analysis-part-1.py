# imports used in the project

import zipfile
import numpy as np 
import scipy as sp 
import matplotlib as mpl
import matplotlib.cm as cm 
import matplotlib.pyplot as plt
import pandas as pd 
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')
from collections import Counter

# plots descriptions
#Can be googled to find optimum style.

plt.style.use(['dark_background', 'bmh'])
plt.rc('axes', facecolor='k')
plt.rc('figure', facecolor='k')
plt.rc('figure', figsize=(5,5))
# Read the data from CSV files into pandas data frame

df_events = pd.read_csv("../input/football-events/events.csv")
df_game_info = pd.read_csv("../input/football-events/ginf.csv")
# Encode the data into respective data frames

encoding = pd.read_table('../input/football-events/dictionary.txt', delim_whitespace=False, names=('num','events'))
event_type=encoding[1:13]
event_type_2=encoding[14:18]
side=encoding[19:21]
shot_place=encoding[22:35]
shot_outcome=encoding[36:40]
location=encoding[41:60]
bodypart=encoding[61:64]
assist_method=encoding[65:70]
situition=encoding[71:75]
# Manually convert the dictionary.txt to python dictionaries

event_types = {1:'Attempt', 2:'Corner', 3:'Foul', 4:'Yellow card', 5:'Second yellow card', 6:'Red card', 7:'Substitution', 8:'Free kick won', 9:'Offside', 10:'Hand ball', 11:'Penalty conceded'}
event_types2 = {12:'Key Pass', 13:'Failed through ball', 14:'Sending off', 15:'Own goal'}
sides = {1:'Home', 2:'Away'}
shot_places = {1:'Bit too high', 2:'Blocked', 3:'Bottom left corner', 4:'Bottom right corner', 5:'Centre of the goal', 6:'High and wide', 7:'Hits the bar', 8:'Misses to the left', 9:'Misses to the right', 10:'Too high', 11:'Top centre of the goal', 12:'Top left corner', 13:'Top right corner'}
shot_outcomes = {1:'On target', 2:'Off target', 3:'Blocked', 4:'Hit the bar'}
locations = {1:'Attacking half', 2:'Defensive half', 3:'Centre of the box', 4:'Left wing', 5:'Right wing', 6:'Difficult angle and long range', 7:'Difficult angle on the left', 8:'Difficult angle on the right', 9:'Left side of the box', 10:'Left side of the six yard box', 11:'Right side of the box', 12:'Right side of the six yard box', 13:'Very close range', 14:'Penalty spot', 15:'Outside the box', 16:'Long range', 17:'More than 35 yards', 18:'More than 40 yards', 19:'Not recorded'}
bodyparts = {1:'right foot', 2:'left foot', 3:'head'}
assist_methods = {1:'Pass', 2:'Cross', 3:'Headed pass', 4:'Through ball'}
situations = {1:'Open play', 2:'Set piece', 3:'Corner', 4:'Free kick'}
## Map the leagues' names to their popular names for easier understanding
leagues = {'E0': 'Premier League', 'SP1': 'La Liga',
          'I1': 'Serie A', 'F1': 'League One', 'D1': 'Bundesliga'}

## Map them to events
df_game_info.league = df_game_info.league.map(leagues)
# Map the dictionaries onto the events dataframe

df_events['event_type'] =   df_events['event_type'].map(event_types)
df_events['event_type2'] =  df_events['event_type2'].map(event_types2)
df_events['side'] =         df_events['side'].map(sides)
df_events['shot_place'] =   df_events['shot_place'].map(shot_places)
df_events['shot_outcome']=  df_events['shot_outcome'].map(shot_outcomes)
df_events['location'] =     df_events['location'].map(locations)
df_events['bodypart'] =     df_events['bodypart'].map(bodyparts)
df_events['assist_method']= df_events['assist_method'].map(assist_methods)
df_events['situation'] =    df_events['situation'].map(situations)

# Merge other dataset to have country, league, date and season
df_events = df_events.merge(df_game_info ,how = 'left')
df_game_info.season = df_game_info.season.astype('category')
df_game_info.league = df_game_info.league.astype('category')
df_game_info.country = df_game_info.country.astype('category')
df_game_info.league.unique()
assist_method
# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
missing_values_table(df_events)
# Fill the required features with new class 'unknown'
df_events.shot_place.fillna('UNK', inplace= True)
df_events.player.fillna('UNK', inplace= True)
df_events.shot_outcome.fillna('UNK', inplace= True)
df_events.bodypart.fillna('UNK', inplace= True)
df_events.location.fillna('UNK', inplace= True)
df_events.assist_method.fillna('UNK', inplace= True);
df_events.info()
# Notice that a lot of the floats and ints are in fact categoricals
# We can fix this with Pandas' astype function
categoricals = ['id_odsp', 'event_type', 'event_team', 'opponent', 'shot_place', 'shot_outcome', 'location', 'bodypart', 'assist_method', 'situation', 'side']
d = dict.fromkeys(categoricals,'category')
df_events = df_events.astype(d)
df_events['is_goal'] = df_events['is_goal'].astype('bool') # this is a bool, we can fix that too while we're at it
df_events.info()

df_unique_events = df_events.drop_duplicates() 
# Get the yellow cards data

first_yellow_cards = df_unique_events [df_unique_events ['event_type'] == ('Yellow card')] # select first yellow cards
second_yellow_cards= df_unique_events [df_unique_events ['event_type'] == ('Second yellow card')] # select second yellow cards
red_cards = df_unique_events [df_unique_events['event_type'] == ('Red card')] # select red cards
yellow_cards= df_unique_events [df_unique_events ['event_type'] == ('Yellow card' or 'Second yellow card')]

card_frames = [red_cards, yellow_cards]
all_cards = pd.concat(card_frames)

# Get the yellow cards against time of playing the game
fig = plt.figure(figsize=(14,8))
plt.hist(red_cards.time, 100, color="red")
plt.xlabel("Minute  of the Game")
plt.ylabel("Red Cards")
plt.title("When Red Cards Occur")


# plot the second yellow cards against time of playing the game

fig2 = plt.figure(figsize=(14,8))
plt.hist(first_yellow_cards.time, 100, color="yellow")
plt.xlabel("Minute of the Game")
plt.ylabel("First Yellow Cards")
plt.title("When First Yellow Cards Occur")



# plot the red cards against time of playing the game

fig3 = plt.figure(figsize=(14,8))
plt.hist(second_yellow_cards.time, 100, color="yellow")
plt.xlabel("Minute of the Game")
plt.ylabel("Second Yellow Cards")
plt.title("When Second Yellow Cards Occur")


# Get the yellow cards against time of playing the game
fig4 = plt.figure(figsize=(14,8))                                                            
plt.hist(all_cards.time, 100, color="orange")
plt.xlabel("Minute  of the Game")
plt.ylabel("Cards served")
plt.title("When cards are served")

yellow_league = pd.crosstab(index=yellow_cards.event_type, columns=yellow_cards.league)
yellow_league.plot(kind='bar', figsize=(14,14))

red_league = pd.crosstab(index=red_cards.event_type, columns=red_cards.league)
red_league .plot(kind='bar', figsize=(14,14))
player_red_card = (red_cards[['player', 'league']])
league_one = player_red_card [player_red_card.league == 'League One'].groupby('player').count()
La_Liga = player_red_card [player_red_card.league == 'La Liga'].groupby('player').count()
Bundesliga = player_red_card [player_red_card.league == 'Bundesliga'].groupby('player').count()
Serie_A = player_red_card [player_red_card.league == 'Serie A'].groupby('player').count()
Premier_League = player_red_card [player_red_card.league == 'Premier League'].groupby('player').count()


player_red_card_yellow = (yellow_cards[['player', 'league']])
league_one_yellow = player_red_card_yellow [player_red_card_yellow.league == 'League One'].groupby('player').count()
La_Liga_yellow = player_red_card_yellow [player_red_card_yellow.league == 'La Liga'].groupby('player').count()
Bundesliga_yellow = player_red_card_yellow[player_red_card_yellow.league == 'Bundesliga'].groupby('player').count()
Serie_A_yellow = player_red_card_yellow[player_red_card_yellow.league == 'Serie A'].groupby('player').count()
Premier_League_yellow = player_red_card_yellow[player_red_card_yellow.league == 'Premier League'].groupby('player').count()

league_one.columns=['league_one_red']
La_Liga.columns=['La_Liga_red']
Bundesliga.columns=['Bundesliga_red']
Serie_A.columns=['Serie_A_red']
Premier_League.columns=['Premier_League_red']

league_one_yellow.columns=['league_one_yellow']
La_Liga_yellow.columns=['La_Liga_yellow_']
Bundesliga_yellow.columns=['Bundesliga_yellow']
Serie_A_yellow.columns=['Serie_A_yellow']
Premier_League_yellow.columns=['Premier_League_yellow']

cards_per_league = pd.concat([league_one, La_Liga, Bundesliga, Serie_A,  Premier_League,league_one_yellow, \
                              La_Liga_yellow, Bundesliga_yellow, Serie_A_yellow,  Premier_League_yellow]).fillna(0)


goals=df_unique_events[df_unique_events["is_goal"]==1]

fig4=plt.figure(figsize=(14,8))
plt.hist(goals.time,width=1,bins=100,color="green")   #100 so 1 bar per minute
plt.xlabel("Minutes")
plt.ylabel("Number of goals")
plt.title("Number of goals against Time during match")