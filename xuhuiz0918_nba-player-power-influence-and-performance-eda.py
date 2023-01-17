# Load packages

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.formula.api as smf

import statsmodels.api as sm

color = sns.color_palette()

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

%matplotlib inline
# Load dataset

player_social_influence = pd.read_csv("../input/nba_2017_players_with_salary_wiki_twitter.csv", index_col = 0); player_social_influence.head() 
# Correlation heatmap

plt.subplots(figsize=(25,25))

ax = plt.axes()

ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season")

corr = player_social_influence.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
# Relationship between SALARY_MILLIONS and PAGEVIEWS

sns.lmplot(x="PAGEVIEWS", y="SALARY_MILLIONS", data=player_social_influence)
# Relationship between SALARY_MILLIONS and TWITTER_FAVORITE_COUNT

sns.lmplot(x="TWITTER_FAVORITE_COUNT", y="SALARY_MILLIONS", data=player_social_influence)
# Relationship between SALARY_MILLIONS and TWITTER_RETWEET_COUNT

sns.lmplot(x="TWITTER_RETWEET_COUNT", y="SALARY_MILLIONS", data=player_social_influence)
# Relationship between SALARY_MILLIONS and WINS_RPM

sns.lmplot(x="WINS_RPM", y="SALARY_MILLIONS", data=player_social_influence)
# Relationship between SALARY_MILLIONS and Social Influence

results = smf.ols(formula='SALARY_MILLIONS ~ PAGEVIEWS + TWITTER_FAVORITE_COUNT + TWITTER_RETWEET_COUNT', data=player_social_influence).fit()

print(results.summary())
# Relationship between Player's performance and WINS_RPM

result = smf.ols(formula='WINS_RPM ~ POINTS + TRB + AST + STL + BLK + TOV + PF', data=player_social_influence).fit()

print(result.summary())
# Add a column called Social_Influence: Add PAGEVIEWS, TWITTER_FAVORITE_COUNT, TWITTER_RETWEET_COUNT

player_social_influence['Social_Influence'] = player_social_influence['PAGEVIEWS'] + player_social_influence['TWITTER_FAVORITE_COUNT'] + player_social_influence['TWITTER_RETWEET_COUNT']

player_social_influence.head()
# Relationship between SALARY_MILLIONS and Social Influence

sns.lmplot(x="Social_Influence", y="SALARY_MILLIONS", data=player_social_influence)
# Load ggplot package

from ggplot import *
# Relationship between Salary and Age for five positions

p = ggplot(player_social_influence,aes(x="AGE", y="SALARY_MILLIONS", color="POSITION")) + geom_point(size=200)

p + xlab("Age") + ylab("Salary") + ggtitle("NBA Players 2016-2017: Age vs Salary")
# Order Player by Salary and Social Influence descendingly: List top 5

player_social_influence.sort_values(by=['SALARY_MILLIONS', 'Social_Influence'], ascending=[False, False])

player_social_influence.head()