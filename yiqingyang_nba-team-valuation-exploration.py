import pandas as pd

import statsmodels.api as sm

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

from ggplot import *

%matplotlib inline
attendance_df = pd.read_csv("../input/nba_2017_attendance.csv");attendance_df.head()
endorsement_df = pd.read_csv("../input/nba_2017_endorsements.csv");endorsement_df.head()
valuations_df = pd.read_csv("../input/nba_2017_team_valuations.csv");valuations_df.head()
salary_df = pd.read_csv("../input/nba_2017_salary.csv");salary_df.head()
pie_df = pd.read_csv("../input/nba_2017_pie.csv");pie_df.head()
plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv");plus_minus_df.head()
br_stats_df = pd.read_csv("../input/nba_2017_br.csv");br_stats_df.head()
elo_df = pd.read_csv("../input/nba_2017_elo.csv");elo_df.head()
attendance_valuation_df = attendance_df.merge(valuations_df, how="inner", on="TEAM")

attendance_valuation_elo_df = attendance_valuation_df.merge(elo_df, how="inner", on="TEAM")

attendance_valuation_elo_df.head()
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
#standardize the data first

sc = StandardScaler()

data = sc.fit_transform(br_stats_df.iloc[:,5:].dropna())



#fit into k-means cluster

kmeans = KMeans(n_clusters=7)

kmeans.fit(data)



player_kmeans4 = br_stats_df.copy().dropna()

player_kmeans4["cluster"] = kmeans.labels_

player_kmeans4.head()
plt.scatter(player_kmeans4.iloc[:,-2], player_kmeans4.iloc[:,-3], c=kmeans.labels_,  s=50, cmap='viridis')
g = sns.FacetGrid(player_kmeans4, col="cluster")

g = g.map(plt.scatter, "PS/G", "3P%", marker="o")
p = ggplot(player_kmeans4,aes(x="PS/G", y="2P", color="Pos")) + geom_point(size=200) +facet_grid("cluster")

p + xlab("POINTS/GAME") + ylab("2P") + ggtitle("NBA Players 2016-2017")
player_salary = br_stats_df.merge(salary_df,how = "left",left_on="Player",right_on = "NAME")

player_salary.head(3)
plt.figure(figsize =(8,10))

sns.boxplot(y="TEAM", x="SALARY", data = player_salary)
player_salary.loc[player_salary["POSITION"]==' PG',"POSITION"] = 'PG'

player_salary.loc[player_salary["POSITION"]==' SF',"POSITION"] = 'SF'



plt.figure(figsize =(8,10))

sns.boxplot(y="POSITION", x="SALARY", data = player_salary)
plt.figure(figsize=(15,10))

sns.heatmap(player_salary.corr(),cmap='coolwarm',annot=True)
sc = StandardScaler()

data = sc.fit_transform(player_salary.dropna().select_dtypes(include=['float64','int64']))

data = pd.DataFrame(data,index = player_salary.dropna().index, 

                    columns = player_salary.select_dtypes(include=['float64','int64']).columns)

plt.figure(figsize=(15,10))

sns.heatmap(data.corr(),cmap='coolwarm',annot=True)
sns.lmplot(x="SALARY", y="PS/G", data=player_salary)
player_salary[['Player','2P','2PA','2P%','3P','3PA','3P%']].sort_values(

    by=['3P'],ascending = False).drop_duplicates().head(5)
sns.lmplot(x="2P%", y="3P%", data=player_salary)

plt.ylim(0,0.6)

plt.xlim(0,0.8)
import statsmodels.api as sm
train_x = attendance_valuation_elo_df.drop(["ELO","TEAM","CONF"],axis=1)

train_y = attendance_valuation_elo_df["ELO"]



x = sm.add_constant(train_x, has_constant='add')

est = sm.OLS(train_y, x)

est = est.fit()

print(est.summary()) 
#predict

pred_y= est.predict(x)

pred_y