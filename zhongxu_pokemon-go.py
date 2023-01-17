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

pokemon = pd.read_csv("../input/Pokemon.csv", index_col = 0)

pokemon.head()
# Load ggplot package

from ggplot import *
p = ggplot(pokemon,aes(x='Attack', y='Defense')) + geom_point(size=15, color = 'blue') + stat_smooth(color = 'red', se=False, span=0.2) + facet_grid('Legendary')

p + xlab("Attack") + ylab("Defense") + ggtitle("Pokemon Attack vs Defense by legendary or not")
# Correlation heatmap

plt.subplots(figsize=(15,12))

ax = plt.axes()

ax.set_title("Pokemon Statistics Correlation Heatmap:")

corr = pokemon.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
# Split 

train ,test = train_test_split(pokemon,test_size=0.3)

train.head()

test.head()