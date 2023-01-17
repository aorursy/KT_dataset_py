import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
seeds = pd.read_csv('../input/TourneySeeds.csv')
seeds.head(3)
games = pd.read_csv('../input/TourneyCompactResults.csv')
games.head(3)
allgames = games.merge(seeds, how='inner', left_on=['Wteam', 'Season'], right_on=['Team', 'Season'])
allgames.head(3)
allgames = allgames.merge(seeds, how='inner', left_on=['Lteam', 'Season'], right_on=['Team', 'Season'])
allgames.head(3)
#####split strings to get the seed



allgames['Winning Seed'] = allgames['Seed_x'].str[1:]
allgames['Losing Seed'] = allgames['Seed_y'].str[1:]
allgames.tail(3)
def change(ch):
    if len(ch) > 2:
        return ch[:2]
    else:
        return ch


allgames['Winner'] = allgames['Winning Seed'].apply(change)
allgames['Loser'] = allgames['Losing Seed'].apply(change)
allgames['Winner'].sort_values().tail(30)
allgames['Winner'] =  allgames['Winner'].astype('int')
allgames['Loser'] =  allgames['Loser'].astype('int')

allgames.info()

allgames.groupby('Winner')['Wteam'].count()

plt.hist(allgames['Winner'], color='m', alpha=0.65, edgecolor='black')

plt.style.use('ggplot')
allgames['game number'] = allgames.groupby(['Season', 'Winner'])['Winner'].transform('rank')
#get rid of the play in games
allgames[allgames['Winner'] == 16].head(10)
allgames = allgames[(allgames['Winner'] != 16) & (allgames['Loser'] != 16)]

allgames['game number'] = allgames.sort_values(['Season', 'Daynum']).groupby(['Season']).cumcount()+1
allgames.sort_values(['Season', 'Daynum']).head(70)
def rounds(g):
    if g <= 32:
        return 1
    elif g <= 48:
        return 2
    elif g <= 56:
        return 3
    elif g <= 60:
        return 4
    else:
        return 5
allgames['Round'] = allgames['game number'].apply(rounds)
allgames.head(10)
wrounds = allgames.groupby(['Round', 'Winner']).agg({'Wteam' : 'count'})
wrounds.reset_index(inplace=True)
wrounds.head(20)
lrounds = allgames.groupby(['Round', 'Loser']).agg({'Lteam' : 'count'})

lrounds.reset_index(inplace=True)
lrounds.head()
wincounts = pd.merge(wrounds, lrounds, how='outer', left_on=['Round', 'Winner'], right_on=['Round', 'Loser'])

wincounts.head(20)
wincounts['Lteam'] = wincounts['Lteam'].fillna(value = 0)
wincounts['games'] = wincounts['Wteam'] + wincounts['Lteam']
wincounts.head(20)
wincounts['Win Percentage'] = wincounts['Wteam'] / wincounts['games'] 
wincounts.head(20)
wincounts['Seed Number'] = wincounts['Winner']
## omit the 5th round
wincounts = wincounts[wincounts['Round'] != 5]
g = sns.FacetGrid(wincounts, col='Round', hue='Round', palette = 'GnBu_d')

g = g.map(plt.bar, 'Seed Number', "Win Percentage")

plt.subplots_adjust(top=0.8)
g.fig.suptitle('March Madness Team Seed and Likelihood of Winning by Round', fontsize=26) 
plt.savefig("March Madness Rounds.png")
allgames['Difference'] = allgames['Winner'] - allgames['Loser']

def upse(diff):
    if diff > 2:
        return 1
    else:
        return 0

allgames['Upset'] =  allgames['Difference'].apply(upse)
allgames.head()
###concat variable for each upset matchup for each roound

allgames['Matchup'] = allgames[['Winner', 'Loser']].min(axis=1).astype(str)  + 'vs' + allgames[['Winner', 'Loser']].max(axis=1).astype(str)
allgames.head()
##aggregate number of games and upsets
upsetdfsum = allgames.groupby(['Matchup']).agg({"Upset" : "sum"}).reset_index()
upsetdfsum.head()
upsetdfcount = allgames.groupby(['Matchup']).agg({"Upset" : "count"}).reset_index()
upsetdfcount.head()
upsetdf = pd.merge(upsetdfcount, upsetdfsum, how='inner', on='Matchup')
upsetdf.tail()

upsetdf.info()

upsetdf = upsetdf[upsetdf['Upset_y'] != 0]
upsetdf['Upset Percentage'] = upsetdf['Upset_y'] / upsetdf['Upset_x']
upsetdf

#upsetdf['Upset_x'] = upsetdf['Upset_x'].rename('Number of Occurances', axis='columns')
upsetdf.rename(columns={'Upset_x': 'Number of Occurances'}, inplace=True)
upsetdf.drop('Upset_y', axis=1, inplace=True)
upsetdf.head()
#export
upsetdf.to_csv("March Madness Upsets List.csv")