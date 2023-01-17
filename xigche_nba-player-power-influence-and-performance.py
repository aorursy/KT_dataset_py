import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
color = sns.color_palette()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
%matplotlib inline
attendance_valuation_elo_df = pd.read_csv("../input/nba_2017_att_val_elo.csv");
attendance_valuation_elo_df.head()
salary_df = pd.read_csv("../input/nba_2017_salary.csv");
salary_df.head()
pie_df = pd.read_csv("../input/nba_2017_pie.csv");
pie_df.head()
plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv");
plus_minus_df.head()
br_stats_df = pd.read_csv("../input/nba_2017_br.csv");
br_stats_df.head()

plus_minus_df.rename(columns={"NAME":"PLAYER", "WINS": "WINS_RPM"}, inplace=True)
players = []
for player in plus_minus_df["PLAYER"]:
    plyr, _ = player.split(",")
    players.append(plyr)
plus_minus_df.drop(["PLAYER"], inplace=True, axis=1)
plus_minus_df["PLAYER"] = players
plus_minus_df.head()
nba_players_df = br_stats_df.copy()
nba_players_df.rename(columns={'Player': 'PLAYER','Pos':'POSITION', 'Tm': "TEAM", 'Age': 'AGE', "PS/G": "POINTS"}, inplace=True)
nba_players_df.drop(["G", "GS", "TEAM"], inplace=True, axis=1)
nba_players_df = nba_players_df.merge(plus_minus_df, how="inner", on="PLAYER")
nba_players_df.head()
pie_df_subset = pie_df[["PLAYER", "PIE", "PACE", "W"]].copy()
nba_players_df = nba_players_df.merge(pie_df_subset, how="inner", on="PLAYER")
nba_players_df.head()
salary_df.rename(columns={'NAME': 'PLAYER'}, inplace=True)
salary_df["SALARY_MILLIONS"] = round(salary_df["SALARY"]/1000000, 2)
salary_df.drop(["POSITION","TEAM", "SALARY"], inplace=True, axis=1)
salary_df.head()
diff = list(set(nba_players_df["PLAYER"].values.tolist()) - set(salary_df["PLAYER"].values.tolist()))
len(diff)

nba_players_with_salary_df = nba_players_df.merge(salary_df); 
plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY)")
corr = nba_players_with_salary_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.lmplot(x="SALARY_MILLIONS", y="WINS_RPM", data=nba_players_with_salary_df)
results = smf.ols('WINS_RPM~SALARY_MILLIONS', data=nba_players_with_salary_df).fit()

print(results.summary())

nba_players_with_salary_prediction_df = nba_players_with_salary_df.copy()

nba_players_with_salary_prediction_df["predicted"] = results.predict()
nba_players_with_salary_prediction_df
sns.lmplot(x="predicted", y="WINS_RPM", data=nba_players_with_salary_prediction_df)
wiki_df = pd.read_csv("../input/nba_2017_player_wikipedia.csv");wiki_df.head()

wiki_df.rename(columns={'names': 'PLAYER', "pageviews": "PAGEVIEWS"}, inplace=True)
median_wiki_df = wiki_df.groupby("PLAYER").median()
median_wiki_df.head()
median_wiki_df_small = median_wiki_df[["PAGEVIEWS"]]
median_wiki_df_small = median_wiki_df_small.reset_index()
nba_players_with_salary_wiki_df = nba_players_with_salary_df.merge(median_wiki_df_small)
twitter_df = pd.read_csv("../input/nba_2017_twitter_players.csv");twitter_df.head()
nba_players_with_salary_wiki_twitter_df = nba_players_with_salary_wiki_df.merge(twitter_df)
nba_players_with_salary_wiki_twitter_df.head()

plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY & TWITTER & WIKIPEDIA)")
corr = nba_players_with_salary_wiki_twitter_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
!pip -q install -U yellowbrick
nba_players_with_salary_wiki_twitter_df['TWITTER_FAVORITE_COUNT'] = nba_players_with_salary_wiki_twitter_df['TWITTER_FAVORITE_COUNT'].fillna(np.mean(nba_players_with_salary_wiki_twitter_df['TWITTER_FAVORITE_COUNT']))
nba_players_with_salary_wiki_twitter_df['TWITTER_RETWEET_COUNT'] = nba_players_with_salary_wiki_twitter_df['TWITTER_FAVORITE_COUNT'].fillna(np.mean(nba_players_with_salary_wiki_twitter_df['TWITTER_RETWEET_COUNT']))
numerical_df = nba_players_with_salary_wiki_twitter_df.loc[:,["WINS_RPM", "SALARY_MILLIONS", "PAGEVIEWS", "TWITTER_FAVORITE_COUNT","TWITTER_RETWEET_COUNT"]]
numerical_df.head()
from yellowbrick.features import Rank2D

visualizer = Rank2D(algorithm="pearson")
visualizer.fit_transform(numerical_df)
visualizer.poof()
sns.lmplot(x="PAGEVIEWS", y="WINS_RPM", data=nba_players_with_salary_wiki_twitter_df)
results = smf.ols('WINS_RPM ~PAGEVIEWS+TWITTER_FAVORITE_COUNT+TWITTER_RETWEET_COUNT', data=nba_players_with_salary_wiki_twitter_df).fit()
print(results.summary())
nba_players_with_salary_wiki_twitter_prediction_df = nba_players_with_salary_wiki_twitter_df.copy()

nba_players_with_salary_wiki_twitter_prediction_df["predicted"] = results.predict()
nba_players_with_salary_wiki_twitter_prediction_df
sns.lmplot(x="predicted", y="SALARY_MILLIONS", data=nba_players_with_salary_wiki_twitter_prediction_df)
results = smf.ols('WINS_RPM ~PAGEVIEWS+TWITTER_FAVORITE_COUNT+TWITTER_RETWEET_COUNT+SALARY_MILLIONS', data=nba_players_with_salary_wiki_twitter_df).fit()
print(results.summary())
nba_players_with_salary_wiki_twitter_prediction_df = nba_players_with_salary_wiki_twitter_df.copy()

nba_players_with_salary_wiki_twitter_prediction_df["predicted"] = results.predict()
nba_players_with_salary_wiki_twitter_prediction_df
sns.lmplot(x="predicted", y="SALARY_MILLIONS", data=nba_players_with_salary_wiki_twitter_prediction_df)