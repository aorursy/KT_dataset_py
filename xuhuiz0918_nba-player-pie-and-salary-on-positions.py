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

from sklearn.model_selection import train_test_split
# Load dataset

player = pd.read_csv("../input/nba_2017_nba_players_with_salary.csv", index_col = 0)

player.head()
# Correlation heatmap

plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season")

corr = player.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
# Split dataset into train data and test data

x_train ,x_test = train_test_split(player,test_size=0.3)

x_train.head()

x_test.head()
# Load ggplot package

from ggplot import *
# Relationship between Salary and Age for five positions

p = ggplot(player,aes(x='PIE', y='SALARY_MILLIONS')) + geom_point(size=150, color = 'blue') + stat_smooth(color = 'red', se=False, span=0.2) + facet_grid('POSITION')

p + xlab("PIE") + ylab("Salary") + ggtitle("NBA Players 2016-2017: WINS_RPM vs Salary")