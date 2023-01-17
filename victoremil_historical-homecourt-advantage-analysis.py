%matplotlib inline
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
path = "../input/RegularSeasonCompactResults.csv"
df_rs = pd.read_csv(path, usecols=['Season', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc'])
df_rs.head()

### Throw away neutral games

df_rs = df_rs[df_rs['WLoc']!='N']
df_rs['Diff'] = df_rs['WScore'] - df_rs['LScore']
df_rs.head()
n_games = len(df_rs)
print(n_games)
home_mask = df_rs.WLoc=='H'
away_mask = df_rs.WLoc=='A'

home_wins = len(df_rs.loc[home_mask, 'WScore'])
away_wins = len(df_rs.loc[away_mask, 'WScore'])
home_wp = home_wins/(home_wins+away_wins)
print("Home Win-Loss: {}-{} (Winning %: {})".format(home_wins, away_wins, round(home_wp,3)))
games_by_season = df_rs[['Season','WLoc']].groupby('Season').agg('count')
games_by_season.columns = ['Games']
wins_by_season = df_rs[home_mask][['Season','WLoc']].groupby('Season').agg('count')
wins_by_season.columns = ['Wins']
#wins_by_season.rename({'WLoc':'Wins'}, axis=1, inplace=True)

df_home = pd.DataFrame(wins_by_season['Wins']/games_by_season['Games'], columns=['WinP']).reset_index()
df_home.head()
sns.regplot(x="Season",y='WinP',data=df_home, color='g', scatter='False')
sns.set_style("darkgrid")
plt.title("Win % Since 1985")
plt.ylabel("Home Win %")

sns.regplot(x="Season",y='WinP',data=df_home.tail(10), color='g')
plt.title("Home Win % Over Last 10 Seasons")
plt.ylabel('Win %')
df_rs.head()
home_ptdiff = ((df_rs.loc[home_mask,'WScore'].sum() + df_rs.loc[away_mask,'LScore'].sum()) - #(Total home score - Total away score)/n_games 
 (df_rs.loc[home_mask,'LScore'].sum() + df_rs.loc[away_mask,'WScore'].sum())) / n_games
print("Historical Average Homecourt Advantage (points): ", home_ptdiff)

home_scores = df_rs.loc[home_mask, ['Season', 'WScore']].groupby('Season').agg('sum').values + df_rs.loc[away_mask, ['Season', 'LScore']].groupby('Season').agg('sum').values
away_scores = df_rs.loc[home_mask, ['Season', 'LScore']].groupby('Season').agg('sum').values + df_rs.loc[away_mask, ['Season', 'WScore']].groupby('Season').agg('sum').values
df_ptdiff = pd.DataFrame((home_scores-away_scores)/games_by_season)
#df_ptdiff.rename({'Games':'PtDiff'}, axis=1, inplace=True)
df_ptdiff.columns = ['PtDiff']
df_ptdiff = df_ptdiff.reset_index()
df_home['PtDiff'] = df_ptdiff.PtDiff
df_home.head()
sns.regplot(x="Season",y='PtDiff',data=df_home, color='g', scatter='False')
sns.set_style("darkgrid")
plt.title("Home Point Differential since 1985")
plt.ylabel("Home Team Point Differential")

sns.regplot(x="Season",y='PtDiff',data=df_home.tail(10), color='g', scatter='False')
sns.set_style("darkgrid")
plt.title("Home Point Differential Since 2007")
plt.ylabel("Home Team Point Differential")

df_home['WinPDev'] = df_home['WinP'] - home_wp
df_home['PtDiffDev'] =  df_home['PtDiff'] - home_ptdiff
df_home.head()
sns.regplot(x="Season",y='WinPDev',data=df_home, color='g', scatter='False')
sns.set_style("darkgrid")
plt.title("Home Win % Deviation since 1985")
plt.ylabel("Home Win % Deviation")
sns.regplot(x="Season",y='WinPDev',data=df_home.tail(10), color='g', scatter='False')
sns.set_style("darkgrid")
plt.title("Home Win % Deviation Since 2007")
plt.ylabel("Win % Deviation")
sns.regplot(x="Season",y='PtDiffDev',data=df_home, color='g', scatter='False')
sns.set_style("darkgrid")
plt.title("Home Point Differential Deviation Since 1985")
plt.ylabel("Home Point Differential Deviation")
sns.regplot(x="Season",y='PtDiffDev',data=df_home.tail(10), color='g', scatter='False')
sns.set_style("darkgrid")
plt.title("Home Point Differential Deviation Since 2007")
plt.ylabel("Home Point Differential Deviation")
### Save homecourt advantage measurements to a csv
df_home.to_csv('homecourt.csv')
