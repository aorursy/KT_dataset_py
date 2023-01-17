import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv('../input/LeagueofLegends.csv')



data.head()
gold_data = pd.read_csv('../input/goldValues.csv')

gold_data.head()
print(data.shape)

print(data.dtypes)
print('Max game length:', data.gamelength.max())

print('Min game length:', data.gamelength.min())

print('Mean game length:', data.gamelength.mean())

sns.distplot(data.gamelength)
def melt_gold(data):

    # Create the minute columns in array form.

    minutes = ['min_' + str(x) for x in range(1, 82)]

    # Melt the columns.

    melted = pd.melt(data, id_vars=['MatchHistory'], value_vars=minutes,

                     var_name='minute', value_name='gold')

    # Convert the minutes to an integer.

    melted.minute = melted.minute.str.strip('min_').astype(int)

    # Remove rows where gold is  NA.

    melted = melted[melted.gold.notnull()]

    return melted

    

    

    

gold_diff = gold_data[gold_data.NameType == 'golddiff']

gold_diff = melt_gold(gold_diff)

gold_diff.head()
print("Number of Blue Wins", data.bResult.sum())

print("Number of Red Wins", data.rResult.sum())
sns.distplot(gold_diff.groupby('MatchHistory').gold.max(), color='blue', kde=False)

sns.distplot(gold_diff.groupby('MatchHistory').gold.min(), color='red', kde=False)
blue_gold = gold_diff[gold_diff.gold > 0]

red_gold = gold_diff[gold_diff.gold < 0]

plt.plot(red_gold.minute, red_gold.gold, 'r.', blue_gold.minute, blue_gold.gold, 'b.')
color_minute_comparison = pd.DataFrame(data={'blueWins': data.groupby('gamelength').bResult.sum(),

                                             'redWins': data.groupby('gamelength').rResult.sum()})
plt.plot(color_minute_comparison.index, color_minute_comparison.blueWins, 'b-')

plt.plot(color_minute_comparison.index, color_minute_comparison.redWins, 'r-')

plt.axvline(x=data.gamelength.mean(), color='green')

plt.xlabel('Completion Time in Minutes')

plt.ylabel('Number of Wins')
team_comparison = pd.DataFrame(data={'asRed': data.redTeamTag.value_counts(),

                                     'asRedWins': data.groupby('redTeamTag').rResult.sum(),

                                     'asBlue': data.blueTeamTag.value_counts(),

                                     'asBlueWins': data.groupby('blueTeamTag').bResult.sum()})
team_comparison['totalGames'] = team_comparison.asBlue + team_comparison.asRed

team_comparison['blueWinPct'] = team_comparison.asBlueWins / team_comparison.asBlue

team_comparison['redWinPct'] = team_comparison.asRedWins / team_comparison.asRed

team_comparison.head()
plt.plot(team_comparison.asRed, team_comparison.asBlue, 'g.')

plt.xlabel('Number of Games as Red')

plt.ylabel('Number of Games as Blue')
blueWin = team_comparison[team_comparison.redWinPct < team_comparison.blueWinPct]

redWin = team_comparison[team_comparison.redWinPct > team_comparison.blueWinPct]

plt.plot(blueWin.redWinPct, blueWin.blueWinPct, 'b.')

plt.plot(redWin.redWinPct, redWin.blueWinPct, 'r.')

plt.xlabel('Red Win Percentage')

plt.ylabel('Blue Win Percentage')