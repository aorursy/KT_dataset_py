# Load packages

import pandas as pd

import statsmodels.api as sm

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline
# import the dataset

data = pd.read_csv("../input/nba_2017_players_with_salary_wiki_twitter.csv",index_col = 0);

data.sample(5)
data.info()
data.describe()
# Correlation heatmap

plt.subplots(figsize=(25,25))

ax = plt.axes()

ax.set_title("NBA 2016-2017 Season Player Correlation Heatmap")

corr = data.corr()

sns.heatmap(corr)
sns.distplot(data['SALARY_MILLIONS']);
sns.boxplot( x="POSITION",y= "SALARY_MILLIONS" , data = data, orient="Vertical")
sns.boxplot(x= "AGE" , y="SALARY_MILLIONS", data = data, orient="Vertical");
sns.lmplot(x="SALARY_MILLIONS", y="WINS_RPM", data=data)
sns.distplot(data['PAGEVIEWS']);
sns.lmplot(x="SALARY_MILLIONS", y="PAGEVIEWS", data=data)
sns.lmplot(x="SALARY_MILLIONS", y="TWITTER_FAVORITE_COUNT", data=data)
results = smf.ols('SALARY_MILLIONS ~POINTS+WINS_RPM+PAGEVIEWS', data=data).fit()
results.summary()