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
from plotnine import *

#Import data
attendance_valuation_elo_df = pd.read_csv("../input/nba_2017_att_val_elo.csv")
salary_df = pd.read_csv("../input/nba_2017_salary.csv")
pie_df = pd.read_csv("../input/nba_2017_pie.csv")
plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv")
br_stats_df = pd.read_csv("../input/nba_2017_br.csv")

#Convert player names into the same format
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
pie_df_subset = pie_df[["PLAYER", "PIE", "PACE", "W"]].copy()
nba_players_df = nba_players_df.merge(pie_df_subset, how="inner", on="PLAYER")
salary_df.rename(columns={'NAME': 'PLAYER'}, inplace=True)
salary_df["SALARY_MILLIONS"] = round(salary_df["SALARY"]/1000000, 2)
salary_df.drop(["POSITION","TEAM", "SALARY"], inplace=True, axis=1)
nba_players_with_salary_df = nba_players_df.merge(salary_df)
nba_players_with_salary_df.head()
#Correlation matrix heat map
plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY)")
corr = nba_players_with_salary_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
           cmap="Greens",
           annot =True)
#Plot the distribution of salary
sns.kdeplot(nba_players_with_salary_df["SALARY_MILLIONS"], color="lightcoral", shade=True)
#Position vs. Salary
plt.subplots(figsize= (10, 5))
sns.countplot(x= "POSITION", data =nba_players_with_salary_df)

fig = plt.subplots(figsize= (10, 5))
ax = sns.boxplot(y= "SALARY_MILLIONS" , x="POSITION", data = nba_players_with_salary_df, orient="Vertical", width= 0.9)
mean_team_salary_df = nba_players_with_salary_df[["TEAM", "SALARY_MILLIONS"]].groupby("TEAM").mean().sort_values(by="SALARY_MILLIONS", ascending=False) 
mean_team_salary_df
plt.subplots(figsize=(15,15))
df = nba_players_with_salary_df
sns.boxplot( x=df["SALARY_MILLIONS"], y=df["TEAM"] )
#See whether team valuation and conference play a role in average team salary
new_salary_df = pd.read_csv("../input/nba_2017_salary.csv")
mean_salary_df = new_salary_df.groupby(["TEAM"], as_index= False).mean().merge(attendance_valuation_elo_df, how="inner", on="TEAM")
ax = sns.lmplot(x= "VALUE_MILLIONS", y= "SALARY", data= mean_salary_df, hue= "CONF", size = 10)
ax.set(xlabel='Mean Salary of a Team', ylabel='Team Valuation', title="Mean Salary vs Team Valuation:  2016-2017 Season")
# See the range of age
print(set(nba_players_with_salary_df["AGE"]))
sal_One = nba_players_with_salary_df.loc[nba_players_with_salary_df.AGE<=25,"SALARY_MILLIONS"]
sal_Two = nba_players_with_salary_df.loc[(nba_players_with_salary_df.AGE>25) & (nba_players_with_salary_df.AGE <=30), "SALARY_MILLIONS"]
sal_Three = nba_players_with_salary_df.loc[(nba_players_with_salary_df.AGE>30) & (nba_players_with_salary_df.AGE <=35), "SALARY_MILLIONS"]
sal_Four = nba_players_with_salary_df.loc[(nba_players_with_salary_df.AGE>35) & (nba_players_with_salary_df.AGE <=40 ), "SALARY_MILLIONS"]
  
plt.figure(figsize=(15,8))
sns.kdeplot(sal_One, color="darkturquoise", shade=False)
sns.kdeplot(sal_Two, color="lightcoral", shade=False)
sns.kdeplot(sal_Three, color="forestgreen", shade=False)
sns.kdeplot(sal_Four, color="dimgray", shade=False)
plt.legend(['One', 'Two', 'Three', 'Four'])
plt.title('Density Plot of Salary by Age Group')
plt.show()
# Classify age into 4 age groups
def age_group(x):   
    if x <= 25:
        ag = "One"
    elif (x > 25) & (x <= 30):
        ag = "Two"
    elif (x > 30) & (x <= 35):
        ag = "Three"
    else:
        ag = "Four"
    return ag  
nba_players_with_salary_df['Age Group'] = nba_players_with_salary_df.AGE.apply(age_group)
nba_players_with_salary_df.head()

fig = plt.subplots(figsize= (10, 5))
ax = sns.boxplot(y= "SALARY_MILLIONS" , x="Age Group", data = nba_players_with_salary_df, orient="Vertical", width= 0.9)
sns.swarmplot(x="AGE", y="SALARY_MILLIONS", hue="Age Group" ,data=nba_players_with_salary_df)

sns.pairplot(nba_players_with_salary_df, x_vars=['WINS_RPM','POINTS','MP',"FG%", "2P%", "3P%", "FT%", "eFG%"], y_vars='SALARY_MILLIONS', size=7, aspect=0.8,kind = 'reg')
plt.savefig("pairplot.jpg")

# Classify salary into 4 age groups
def salary_group(x):   
    if x <= 10:
        sg = "Low"
    elif (x > 10) & (x <= 20):
        sg = "Medium"
    elif (x > 20) & (x <= 30):
        sg = "High"
    else:
        sg = "Extreme High"
    return sg  
nba_players_with_salary_df['Salary Group'] = nba_players_with_salary_df.SALARY_MILLIONS.apply(salary_group)
nba_players_with_salary_df.head()
mean_performance_df = nba_players_with_salary_df[["FG%", "2P%", "3P%", "eFG%","Salary Group"]].groupby("Salary Group").mean()
mean_performance_df.head()
sns.pairplot(nba_players_with_salary_df, x_vars=['FG','2P','3P','FT'], y_vars=['FG%','2P%','3P%','FT%','MP'], size=7, aspect=0.8,kind = 'reg')
plt.savefig("pairplot.jpg")

results = smf.ols('SALARY_MILLIONS ~POINTS+WINS_RPM+MP', data=nba_players_with_salary_df).fit()

print(results.summary())

wiki_df = pd.read_csv("../input/nba_2017_player_wikipedia.csv");wiki_df.head()

wiki_df.rename(columns={'names': 'PLAYER', "pageviews": "PAGEVIEWS"}, inplace=True)

median_wiki_df = wiki_df.groupby("PLAYER").median()
median_wiki_df.head()

median_wiki_df_small = median_wiki_df[["PAGEVIEWS"]]
median_wiki_df_small.head()
median_wiki_df_small = median_wiki_df_small.reset_index()
median_wiki_df_small.head()
nba_players_with_salary_wiki_df = nba_players_with_salary_df.merge(median_wiki_df_small)
nba_players_with_salary_wiki_df.head()
twitter_df = pd.read_csv("../input/nba_2017_twitter_players.csv");twitter_df.head()

nba_players_with_salary_wiki_twitter_df = nba_players_with_salary_wiki_df.merge(twitter_df)

nba_players_with_salary_wiki_twitter_df.head()

plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY & TWITTER & WIKIPEDIA)")
corr = nba_players_with_salary_wiki_twitter_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
           annot =True)
sns.pairplot(nba_players_with_salary_wiki_twitter_df, x_vars=['FG%','2P%','3P%'], y_vars=['PAGEVIEWS', 'TWITTER_FAVORITE_COUNT'], size=7, aspect=0.8,kind = 'reg')
plt.savefig("pairplot.jpg")
plt.scatter(nba_players_with_salary_wiki_twitter_df['SALARY_MILLIONS'], nba_players_with_salary_wiki_twitter_df['PAGEVIEWS'], color="blue", label="Pageviews vs. Salary")
plt.scatter(nba_players_with_salary_wiki_twitter_df['SALARY_MILLIONS'], nba_players_with_salary_wiki_twitter_df['TWITTER_FAVORITE_COUNT'], color="orange", label="Twitter vs. Salary")
results = smf.ols('SALARY_MILLIONS ~PAGEVIEWS', data=nba_players_with_salary_wiki_twitter_df).fit()
print(results.summary())
results = smf.ols('SALARY_MILLIONS ~TWITTER_FAVORITE_COUNT', data=nba_players_with_salary_wiki_twitter_df).fit()
print(results.summary())
mean_nba_players_with_salary_wiki_twitter_df = nba_players_with_salary_wiki_twitter_df[["TEAM", "SALARY_MILLIONS","PAGEVIEWS", "TWITTER_FAVORITE_COUNT"]].groupby("TEAM").mean().sort_values(by="SALARY_MILLIONS", ascending=False) 
mean_nba_players_with_salary_wiki_twitter_df
plt.scatter(mean_nba_players_with_salary_wiki_twitter_df['SALARY_MILLIONS'], mean_nba_players_with_salary_wiki_twitter_df['PAGEVIEWS'], color="green", label="Pageviews vs. Team Salary")
plt.scatter(mean_nba_players_with_salary_wiki_twitter_df['SALARY_MILLIONS'], mean_nba_players_with_salary_wiki_twitter_df['TWITTER_FAVORITE_COUNT'], color="red", label="Twitter vs. Team Salary")
#Regression: Average Team Salary vs. Average Team Twitter Favorites
results = smf.ols('SALARY_MILLIONS ~TWITTER_FAVORITE_COUNT', data=mean_nba_players_with_salary_wiki_twitter_df).fit()
print(results.summary())
#Regression: Average Team Salary vs. Average Team Wiki Pageviews
results = smf.ols('SALARY_MILLIONS ~PAGEVIEWS', data=mean_nba_players_with_salary_wiki_twitter_df).fit()
print(results.summary())