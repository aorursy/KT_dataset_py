

import pandas as pd

import statsmodels.api as sm

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline
attendance_df = pd.read_csv("../input/nba_2017_attendance.csv");attendance_df.head()
endorsement_df = pd.read_csv("../input/nba_2017_endorsements.csv");endorsement_df.head()

valuations_df = pd.read_csv("../input/nba_2017_team_valuations.csv");valuations_df.head()

pie_df = pd.read_csv("../input/nba_2017_pie.csv");pie_df.head()
plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv");plus_minus_df.head()

br_stats_df = pd.read_csv("../input/nba_2017_br.csv");br_stats_df.head()

elo_df = pd.read_csv("../input/nba_2017_elo.csv");elo_df.head()

attendance_valuation_df = attendance_df.merge(valuations_df, how="inner", on="TEAM")
attendance_valuation_df.head()
nba_2017_twitter_players_df = pd.read_csv("../input/nba_2017_twitter_players.csv");nba_2017_twitter_players_df.head(10)
salary_df = pd.read_csv("../input/nba_2017_salary.csv");salary_df.head()
#switch the name to player in salary

salary_df.rename(columns={'NAME': 'PLAYER'}, inplace=True); salary_df.head()
salary_twitter_df = nba_2017_twitter_players_df.merge(salary_df, how="inner", on="PLAYER"); salary_twitter_df.head()
sns.kdeplot(salary_twitter_df["TWITTER_FAVORITE_COUNT"], color="mediumpurple", shade=True)

plt.show()
sns.lmplot(x="TWITTER_FAVORITE_COUNT", y="SALARY", data=salary_twitter_df)

plt.show()
plt.figure(figsize=(8,15))

sns.boxplot(x="SALARY", y="TEAM",data=salary_twitter_df, orient="h")

plt.show()
plt.figure(figsize=(8,15))

sns.boxplot(x="TWITTER_FAVORITE_COUNT", y="TEAM",data=salary_twitter_df, orient="h")

plt.show()
corr = salary_twitter_df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"));sns.pairplot(salary_twitter_df, hue="TEAM")
#ME

#joint distribution

with sns.axes_style('white'):

    sns.jointplot("PCT", "VALUE_MILLIONS", data=attendance_valuation_df, kind='hex')


with sns.axes_style('white'):

    sns.jointplot("AVG", "VALUE_MILLIONS", data=attendance_valuation_df, kind='hex')
corr = attendance_valuation_df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
valuations = attendance_valuation_df.pivot("TEAM", "AVG", "VALUE_MILLIONS")
plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("NBA Team AVG Attendance vs Valuation in Millions:  2016-2017 Season")

sns.heatmap(valuations,linewidths=.5, annot=True, fmt='g')
results = smf.ols('VALUE_MILLIONS ~AVG', data=attendance_valuation_df).fit()

print(results.summary())

sns.residplot(y="VALUE_MILLIONS", x="AVG", data=attendance_valuation_df)

attendance_valuation_elo_df = attendance_valuation_df.merge(elo_df, how="inner", on="TEAM")

attendance_valuation_elo_df.head()

corr_elo = attendance_valuation_elo_df.corr()

plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("NBA Team Correlation Heatmap:  2016-2017 Season (ELO, AVG Attendance, VALUATION IN MILLIONS)")

sns.heatmap(corr_elo, 

            xticklabels=corr_elo.columns.values,

            yticklabels=corr_elo.columns.values)
corr_elo



ax = sns.lmplot(x="ELO", y="AVG", data=attendance_valuation_elo_df, hue="CONF", size=12)

ax.set(xlabel='ELO Score', ylabel='Average Attendence Per Game', title="NBA Team AVG Attendance vs ELO Ranking:  2016-2017 Season")

attendance_valuation_elo_df.groupby("CONF")["ELO"].median()

attendance_valuation_elo_df.groupby("CONF")["AVG"].median()

results = smf.ols('AVG ~ELO', data=attendance_valuation_elo_df).fit()

print(results.summary())

from sklearn.cluster import KMeans

k_means = KMeans(n_clusters=3)

cluster_source = attendance_valuation_elo_df.loc[:,["AVG", "ELO", "VALUE_MILLIONS"]]

kmeans = k_means.fit(cluster_source)

attendance_valuation_elo_df['cluster'] = kmeans.labels_

ax = sns.lmplot(x="ELO", y="AVG", data=attendance_valuation_elo_df,hue="cluster", size=12, fit_reg=False)

ax.set(xlabel='ELO Score', ylabel='Average Attendence Per Game', title="NBA Team AVG Attendance vs ELO Ranking Clustered on ELO, AVG, VALUE_MILLIONS:  2016-2017 Season")
kmeans.__dict__

kmeans.cluster_centers_

cluster_1 = attendance_valuation_elo_df["cluster"] == 1

attendance_valuation_elo_df[cluster_1]
