# initialise

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set_style('darkgrid')
# read player stats data

players = pd.read_csv('../input/basketball_players.csv', low_memory = False)

players.head()
# read awards data

awards = pd.read_csv('../input/basketball_awards_players.csv')

awards.head()
# breakdown of number of observations by league

players['lgID'].value_counts()
# filter to only show rows corresponding to the NBA

players = players[players['lgID'] == 'NBA']

awards = awards[awards['lgID'] == 'NBA']



# remove columns not relevant to research question

players = players.drop(['stint','tmID','lgID','GS','note'], axis=1)

awards = awards.drop(['lgID','note', 'pos'], axis=1)



# remove variables corresponding to postseason stats

cols = [c for c in players.columns if c[:4] != 'Post']

players = players[cols]



# remove players with zero games played in any given year

players = players[players['GP'] != 0]



# remove observations with zero field goals and zero free throws attempted

players = players[players['fgAttempted'] != 0]

players = players[players['ftAttempted'] != 0]
# checking for sum of stats by year that equal zero

stats = players.columns[3:]   # identify all stats from players data frame

print(stats)

yearStats = players.groupby('year')[stats].sum() # sum stats by year

yearStats.head(10)
# visualising sum of stats by year

def plot_stats_by_year(statistic):

    fig, ax = plt.subplots(figsize = (8,6))

    ax.plot(yearStats.index, yearStats[statistic])

    ax.set_xlabel('year')

    ax.set_ylabel(statistic)

    plt.show()



# plot all stats over the years

for s in stats:

    if 'rebounds' in s.lower():

        continue    # ignore rebounds (see below)

    elif 'Attempted' in s:

        continue    # ignore fg, ft, three attempts (see below)

    elif 'Made' in s:

        continue    # ignore fg, ft, three made (see below)

    else:

        plot_stats_by_year(s)

        

# plot rebounds over the years

fig, ax = plt.subplots(figsize = (8,6))

ax.plot(yearStats.index, yearStats.rebounds, color = 'blue', label = 'total')

ax.plot(yearStats.index, yearStats.dRebounds, color = 'green', label = 'defensive')

ax.plot(yearStats.index, yearStats.oRebounds, color = 'red', label = 'offensive')

ax.set_xlabel('year')

ax.set_ylabel('rebounds')

plt.legend()

plt.show()



# plot field goal and three point attempts over the years

fig, ax = plt.subplots(figsize = (8,6))

ax.plot(yearStats.index, yearStats.fgAttempted, color = 'blue', label = 'total')

ax.plot(yearStats.index, yearStats.threeAttempted, color = 'red', label = '3pt')

ax.set_xlabel('year')

ax.set_ylabel('field goals attempted')

plt.legend()

plt.show()
# remove years with incomplete stats

yearStats = yearStats[(yearStats.T != 0).all()] # remove rows with zeros

yearStats.head(10)
# filter players and awards datasets to only include years with complete stats

players = players[players['year'].isin(yearStats.index)]

awards = awards[awards['year'].isin(yearStats.index)]
# determining total number of games played each year, bar plot

totalGames = players.groupby('year')['GP'].max()

print(totalGames[totalGames != totalGames.max()])

fig, ax = plt.subplots(figsize = (8,6))

sns.barplot(totalGames.index, totalGames, color = '#756bb1')

ax.set_ylabel('games played')

ax.set_xticklabels(totalGames.index, rotation = 'vertical')

plt.show()
# combining rows for players who played for more than 1 team in the same year

players = players.groupby(['playerID', 'year']).sum().reset_index()

players.sort_values('GP', ascending = False).head(10)
# playerID willish03 had 106 games in 2010, so his stats were checked

players.loc[players['playerID'] == 'willish03',:]
# remove row with IDs 12542 and 12543 due to discrepancy

players = players.drop([12542, 12543])

print(players.loc[players['playerID'] == 'willish03',:], "\n")
# compute average stats per game

playersPG = players.iloc[:,0:3] # select playerID, year and GP

for s in stats:

    # for each stat listed in stats, compute the avg per game

    playersPG[s] = np.divide(players[s], players['GP']).round(2)

# compute field goal, free throw and three point fg percentages

with np.errstate(divide='ignore',  invalid='ignore'):    

    playersPG['fgPct'] = (np.divide(players['fgMade'], players['fgAttempted'])*100).round(2)

    playersPG['ftPct'] = (np.divide(players['ftMade'], players['ftAttempted'])*100).round(2)

    playersPG['threePct'] = (np.divide(players['threeMade'], players['threeAttempted'])*100).round(2)



# compute eFG% = (fgMade + 0.5*(threeMade))/fgAttempted

# source: https://www.basketball-reference.com/about/glossary.html

playersPG['efgPct'] = (np.divide(players['fgMade'] + 0.5*players['threeMade'], players['fgAttempted'])*100).round(2)

playersPG.head(10)
# summary statistics of player stats, excluding playerID and year

playersPG.iloc[:,2:].describe().round(2)
# define functions for player per game stats plots



def plot_hist(series):

    fig, ax = plt.subplots(figsize = (8,6))

    sns.distplot(playersPG[series], kde = False, color = 'blue', hist_kws = {'alpha': 0.8, 'edgecolor': 'black'})

    if series == 'year':

        ax.set_xlabel(series)

    elif 'Pct' in series:

        ax.set_xlabel(series)

    else:

        ax.set_xlabel(series + ' per game')

    plt.show()



def plot_pg_stat(s1, s2, annualMean = False, regLine = False, logx = False, logy = False, alpha = 0.2):

    # annualMean specifies if the avg stat per year will be added to the plot

    # regLIne specifies if a linear regression line is to be plotted

    # logx and logy specify if whether or not to take the log of those variables

    # alpha = 0.2 makes the dense scatterplots easier to see, can be altered

    temp = pd.DataFrame(playersPG[[s1, s2]])

    if logx:

        temp[s1] = np.log(temp[s1])

    if logy:

        temp[s2] = np.log(temp[s2])

    fig, ax = plt.subplots(figsize = (8,6))

    if regLine:

        sns.regplot(s1, s2, data = temp, ci = None, color = '#ff7f00', scatter_kws = {'alpha': alpha, 'color': 'blue'})

    else:

        ax.scatter(temp[s1], temp[s2], alpha = alpha, color = 'blue')

    if s1 == 'year' and annualMean:

        ax.plot(sorted(temp[s1].unique()), temp.groupby(s1)[s2].mean(), color = 'red')

    if logx:

        ax.set_xlabel('log (' + s1 + ')', fontsize = 16)

    elif s1 == 'year':

        ax.set_xlabel(s1, fontsize = 16)

    elif s1 == 'GP':

        ax.set_xlabel('games played', fontsize = 16)

    elif 'Pct' in s1:

        ax.set_xlabel(s1, fontsize = 16)

    else:

        ax.set_xlabel(s1 + ' per game', fontsize = 16)

    if logy:

        ax.set_ylabel('log (' + s2 + ')', fontsize = 16)

    elif 'Pct' in s2:

        ax.set_ylabel(s2, fontsize = 16)

    else:

        ax.set_ylabel(s2 + ' per game', fontsize = 16)

    plt.show()



# plots of selected stats

plot_hist('points')

plot_hist('assists')

plot_hist('rebounds')

plot_pg_stat('year', 'minutes', annualMean = True)

plot_pg_stat('year', 'points', annualMean = True)

plot_pg_stat('year', 'threePct', annualMean = True)

plot_pg_stat('minutes', 'points', regLine = True)

plot_pg_stat('minutes', 'points', logx = True, logy = True)

plot_pg_stat('turnovers', 'assists', regLine = True)

plot_pg_stat('fgPct', 'points')

plot_pg_stat('GP', 'fgPct')
# select only All-NBA Team awards and join with player stats

allNBA = awards[awards['award'].str.contains('All-NBA')]

playersMerged = pd.merge(playersPG, allNBA, how='left', on=['playerID','year'])

playersMerged.rename(columns={'award': 'allNBA'}, inplace=True)



# select only players awarded regular season MVP and join with player stats

seasonMVP = awards[awards['award'] == 'Most Valuable Player']

playersMerged = pd.merge(playersMerged, seasonMVP, how='left', on=['playerID','year'])

playersMerged.rename(columns={'award': 'MVP'}, inplace=True)



# select only players awarded Defensive Player of the Year

seasonDPY = awards[awards['award'] == 'Defensive Player of the Year']

playersMerged = pd.merge(playersMerged, seasonDPY, how='left', on=['playerID','year'])

playersMerged.rename(columns={'award': 'DPY'}, inplace=True)



playersMerged.head(10)
# determine total number of players with awards in the dataset

print(playersMerged['allNBA'].value_counts(), '\n')

print(playersMerged['MVP'].value_counts(), '\n')

print(playersMerged['DPY'].value_counts())



# convert awards columns to categorial (1: player received award, 0: no award)

playersMerged['allNBA'] = playersMerged['allNBA'].notnull().astype(int)

playersMerged['MVP'] = playersMerged['MVP'].notnull().astype(int)

playersMerged['DPY'] = playersMerged['DPY'].notnull().astype(int)



# number of players with each award in each year (check to ensure correct number of awards per year)

playersMerged.groupby('year')['allNBA', 'MVP', 'DPY'].sum()
# visualise correlation matrix

corMatrix = playersMerged.iloc[:,2:].corr()

fig, ax = plt.subplots(figsize = (8,6))

sns.heatmap(corMatrix, ax = ax, vmin = -1, vmax = 1, mask = np.zeros_like(corMatrix, dtype = np.bool),

            cmap = sns.diverging_palette(220, 20, as_cmap = True), square = True)

#heatmap code source: https://stackoverflow.com/a/42977946/8452935

plt.show()
# plotting player stats grouped by players with awards and players without awards

def stats_awards_boxplot(var, award = "allNBA"):

    # var is the variable of interest

    # award is the name of the award to categorise by (either allNBA, MVP or DPY)

    # award defaults to allNBA if not specified

    # all variables take strings as inputs

    fig, ax = plt.subplots(figsize = (8,6))

    sns.boxplot(x = award, y = var, data = playersMerged, palette = "Set1")

    ax.set_xlabel(award, fontsize = 16)

    if 'Pct' in var:

        ax.set_ylabel(var, fontsize = 16)

    elif var == 'GP':

        ax.set_ylabel('games played', fontsize = 16)

    else:

        ax.set_ylabel(var + ' per game', fontsize = 16)

    plt.show()



for s in stats:

    if 'rebounds' in s.lower():

        continue    # ignore rebounds (see below)

    elif 'Attempted' in s:

        continue    # ignore fg, ft, three attempts (see below)

    elif 'Made' in s:

        continue    # ignore fg, ft, three made (see below)

    else:

        stats_awards_boxplot(s)



stats_awards_boxplot('fgPct')

stats_awards_boxplot('ftPct')

stats_awards_boxplot('threePct')

stats_awards_boxplot('rebounds')




# prepare training data for modelling

train_allNBA = playersMerged.drop(['playerID', 'year', 'MVP', 'DPY'], axis = 1)