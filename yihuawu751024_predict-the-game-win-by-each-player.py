import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
color = sns.color_palette()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
salary_df = pd.read_csv("../input/nba_2017_salary.csv");salary_df.head()

pie_df = pd.read_csv("../input/nba_2017_pie.csv");pie_df.head()
plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv");plus_minus_df.head()
br_stats_df = pd.read_csv("../input/nba_2017_br.csv");br_stats_df.head()
wiki_df = pd.read_csv("../input/nba_2017_player_wikipedia.csv");wiki_df.head()
wiki_df.rename(columns={'names': 'PLAYER', "pageviews": "PAGEVIEWS"}, inplace=True)
median_wiki_df = wiki_df.groupby("PLAYER").median()
median_wiki_df_small = median_wiki_df[["PAGEVIEWS"]]
median_wiki_df_small = median_wiki_df_small.reset_index()
twitter_df = pd.read_csv("../input/nba_2017_twitter_players.csv");twitter_df.head()

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
nba_players_with_salary_wiki_df = nba_players_with_salary_df.merge(median_wiki_df_small)
nba_players_with_salary_wiki_twitter_df = nba_players_with_salary_wiki_df.merge(twitter_df)
nba_players_with_salary_wiki_twitter_df.head()

plt.subplots(figsize=(20,20))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season")
corr = nba_players_with_salary_wiki_twitter_df.corr()
sns.heatmap(corr, 
           linewidths=0.1,
           annot=True, 
          annot_kws={'fontsize':8 })
#Filter out features which have more than 50% correlation
cor =corr.loc['W',:]
cor=cor[abs(cor)>0.4][:]
cor=cor.drop('W')
imp_feature=cor.index
print(imp_feature)
#EDA by plots to look into the distribution

sns.lmplot(x="Rk", y="W", data= df)
sns.lmplot(x="MP", y="W", data= df)
sns.lmplot(x="FG", y="W", data= df)
sns.lmplot(x="eFG%", y="W", data= df)
sns.lmplot(x="POINTS", y="W", data= df)
sns.lmplot(x="GP", y="W", data= df)
sns.lmplot(x="MPG", y="W", data= df)
sns.lmplot(x="RPM", y="W", data= df)
sns.lmplot(x="WINS_RPM", y="W",data= df)
#Predict Salary and find score
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
df=nba_players_with_salary_wiki_twitter_df
lr=linear_model.LinearRegression()
predict=cross_val_predict(lr,df[imp_feature],df['W'],cv=10)
predicted_score = cross_val_score(lr, df[imp_feature], df['W'], cv=10,scoring='r2')
predicted_score.mean()
lasso=linear_model.Lasso()
predict=cross_val_predict(lasso,df[imp_feature],df['W'],cv=10)
predicted_score = cross_val_score(lasso, df[imp_feature], df['W'], cv=10,scoring='r2')
predicted_score.mean()
lasso.fit(df[imp_feature], df['W'])
df['Nrml_predict']=lasso.predict(df[imp_feature])
df.head()
