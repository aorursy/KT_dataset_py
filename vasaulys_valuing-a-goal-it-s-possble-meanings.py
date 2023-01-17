# Imports -- get these out of the way
# It's considered good Pythonic practice to put imports at top

from random import randint
import os

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

from IPython.display import display


#--> This is a python-converted set of dictionaries from 'dictionary.txt' that I made
from events_dict import *  
# Get Data and local file dictionaries
events = pd.read_csv("events.csv")
games = pd.read_csv("ginf.csv")
shots = events[events['event_type']==1 ]
goals_winning_team = []

# go row by row (don't think a one liner would work)
for I,game in games.iterrows():
    if game['fthg'] > game['ftag']:
        goals_winning_team.append(game['fthg'])
    elif game['fthg'] < game['ftag']:
        goals_winning_team.append(game['ftag'])
        
avg_goals = np.mean(goals_winning_team)
std_goals = np.std(goals_winning_team)
        
print("Average Goals Per Game from Winning Team: %0.3f" % avg_goals)
print("Std. of Goals per game from winning team: %0.3f" % std_goals)
print("%% Deviation: +/- %0.3f%%" % ((std_goals/avg_goals)*100))
# plot this as a histogram
plt.hist(goals_winning_team, bins=10, normed=True)
plt.show()
print("Shots: %d" % shots.shape[0])
print("Goals: %d" % shots[shots['is_goal']==True].shape[0])
print("Shots that convert: %0.3f%%" % (100*(shots[shots['is_goal']==True].shape[0]/shots[shots['event_type']==1].shape[0])))
shots = shots[ shots['location']!=19. ] 
shots.replace({'location': location_dict, 
               'shot_place': shot_place_dict, 
               'bodypart': bodypart_dict, 
               'assist_method': assist_method_dict, 
               'situation': situation_dict}, inplace=True)
shots['uc'] = shots['location'] + ', ' + shots['shot_place'] + ', ' + \
    shots['bodypart'] + ', ' + shots['assist_method'] + ', ' + \
    shots['situation']
unique_combos = shots['uc'].value_counts().to_dict()
unique_combos_that_scored = shots[ shots['is_goal']==True ]['uc'].value_counts().to_dict()

print("Total Unique Combos: %d" % len(unique_combos))

# Filter out keys that did not score
scoring_keys = [k for k in unique_combos.keys() if k in unique_combos_that_scored.keys()]
unique_combos = {key:unique_combos[key] for key in scoring_keys}

print("Total Unique Combos that Scored: %d" % len(unique_combos))

# Filter out keys that don't have at least 100 occurrences
unique_combos = {key:value for key,value in unique_combos.items() if value > 100}
print("Number of Unique Combos: %d" % len(unique_combos))

items = sorted(unique_combos.items(), key=lambda x: x[1], reverse=True)

# Plot the data
plt.bar(range(len(unique_combos)), [i[1] for i in items], align='center')
plt.show()

# Print out most frequent shot type
print("Most Common Shot Taken => %r: %d occurences" % (items[0][0],items[0][1]))
shots_per_uc = unique_combos
goals_per_uc = shots[ shots['is_goal']==True ]['uc'].value_counts().to_dict()

l = [];

for k in shots_per_uc.keys():
    s = shots_per_uc[k]
    g = goals_per_uc[k]
    f = float(g/s)
    
    split_uc = k.split(", ")
    location = split_uc[0]
    shot_place = split_uc[1]
    bodypart = split_uc[2]
    assist_method = split_uc[3]
    situation = split_uc[4]
    
    l.append([location,
              shot_place,
              bodypart,
              assist_method,
              situation,
              s,g,f])


# While the 'uc' column makes typing out easier, display is better if
# we return to the original columnar display
df = pd.DataFrame(l,columns=['location',
                             'shot_place',
                             'bodypart',
                             'assist_method',
                             'situation',
                             'shots',
                             'goals',
                             'percentage_conversion'])

# Finally, we sort the dataframe by the highest conversion rate
df.sort_values(axis=0, by=['percentage_conversion'],
               ascending=False, inplace=True)

# Display the dataframe itself
display(df)
# Rename the DF from before
events = df

# Rebuild the unique column up
events['key'] = events['location']+'/'+events['shot_place']+'/'+events['bodypart']+'/'+events['assist_method']+'/'+events['situation']

other_conversions = pd.Series(events['percentage_conversion'].values,index=events['key']).to_dict()

# Then we'll rank all the other events as zero as there's 
# insufficient information to make a judgement -- this works out in the calculations
shots['key'] = shots['location']+'/'+shots['shot_place']+'/'+shots['bodypart']+'/'+shots['assist_method']+'/'+shots['situation']
conversions = pd.Series([0 for i in range(shots['key'].shape[0])], index=shots['key']).to_dict()

# Then we combine them 
conversions.update(other_conversions)
# We assign value to every contribution
shots['value'] = shots['key'].apply(lambda x: conversions[x])

# Get all players into a list -- players who are not mention (i.e. NaN) are dropped
players = shots['player'].dropna().unique().tolist()
player_contributions_per_game = {}
player_games_number = {}
for player in players:
    A = shots[ shots['player']==player ]
    games_played = A['id_odsp'].unique().tolist()

    contributions = 0
    for game in games_played:
        contributions += A[ A['id_odsp']==game ]['value'].sum()
        
    # We normalize the contribution such that it's _per-game_
    player_contributions_per_game[player] = float(contributions/len(games_played))
    player_games_number[player] = len(games_played);

anonymous_contributions = list(player_contributions_per_game.values())
plt.hist(anonymous_contributions,bins='auto')
plt.show()
A = [x for x in list(player_contributions_per_game.values()) if x > 0.0]

plt.hist(A, bins='auto')
plt.show()
war = np.median(anonymous_contributions)
print("Median Player's Contribution: %f" % war)
print("Mean Player's Contribution: %f" % np.mean(anonymous_contributions))
print("Mean Player's Contribution (zeros dropped): %f" % np.mean(A))
# Let's look at the top players in terms of contributions 
B = [(v,k) for k,v in player_contributions_per_game.items()]
B = sorted(B, key=lambda x: x[0], reverse=True)
l = []

for i in B:
    
    if player_games_number[i[1]] < 10:
        continue

    l.append([i[1],i[0],player_games_number[i[1]]])
    #print("%r: %f, played %d games" % (i[1],i[0],player_games_number[i[1]]));
    
df = pd.DataFrame(l,
                 columns=['player','contribution_per_game','games_played']
                )

# Display the data
display(df);