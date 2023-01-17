# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# read and skim the data
data = pd.read_csv('../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')
data.head()
# No missing value check is required
# Find the table size:
print("(row, column): ", data.shape)
# See if there is repeated record:
print("\nThe dataset has duplicated game record: ", data.duplicated().any())
# See the variables available:
print("\nThe column names: ")
print(data.columns.values)
# After adjusting the columns:
data['blueWardsPlacedDiff'] = data.apply(lambda row: int(row['blueWardsPlaced']-row['redWardsPlaced']),axis=1)
data['blueWardsDestoryedDiff'] = data.apply(lambda row: int(row['blueWardsDestroyed']-row['redWardsDestroyed']),axis=1)
data['blueCSDiff'] = data.apply(lambda row: int(row['blueTotalMinionsKilled']-row['redTotalMinionsKilled']),axis=1)
data['blueTotalJungleMinionsKilledDiff'] = data.apply(lambda row: int(row['blueTotalJungleMinionsKilled']-row['redTotalJungleMinionsKilled']),axis=1)
data['blueKillsDiff'] = data.apply(lambda row: int(row['blueKills']-row['redKills']),axis=1)
data['blueDeathsDiff'] = data.apply(lambda row: int(row['blueDeaths']-row['redDeaths']),axis=1)
data['blueAssistsDiff'] = data.apply(lambda row: int(row['blueAssists']-row['redAssists']),axis=1)
data['blueDragonsDiff'] = data.apply(lambda row: int(row['blueDragons']-row['redDragons']),axis=1)
data['blueHeraldsDiff'] = data.apply(lambda row: int(row['blueHeralds']-row['redHeralds']),axis=1)
data['blueTowersDestroyedDiff'] = data.apply(lambda row: int(row['blueTowersDestroyed']-row['redTowersDestroyed']),axis=1)
data['blueAvgLevelDiff'] = data.apply(lambda row: row['blueAvgLevel']-row['redAvgLevel'],axis=1)

data_1 = data.drop(['blueGoldPerMin','redGoldPerMin','blueCSPerMin','redCSPerMin','blueEliteMonsters',
                   'redEliteMonsters','blueTotalGold','redTotalGold','blueTotalMinionsKilled',
                   'redTotalMinionsKilled','redGoldDiff','redExperienceDiff', 'blueTotalJungleMinionsKilled', 
                   'redTotalJungleMinionsKilled','blueWardsPlaced','redWardsPlaced','blueWardsDestroyed',
                   'redWardsDestroyed','redFirstBlood','blueKills','blueDeaths','blueAssists','blueDragons',
                   'blueHeralds','blueTowersDestroyed','blueAvgLevel','blueTotalExperience','redKills','redDeaths',
                   'redAssists','redDragons','redHeralds','redTowersDestroyed','redAvgLevel','redTotalExperience'],axis=1)
print(data_1.columns.values)
data_1.head()
# General discription of the table
data_1[data_1.columns[1:]].describe()
# Visualize and compare the ratio of blueWins: redWins and blueFirstBlood: redFirstBlood
blueWins = pd.DataFrame({'Side':['red','blue'], 'Winning':data_1['blueWins'].value_counts(),'FirstBlood':data_1['blueFirstBlood'].value_counts()})
blueWins.plot(y='Winning',kind='pie',colors=['tomato','lightskyblue'], figsize=(5,5),labels=['red','blue'],autopct='%1.2f%%')
blueWins.plot(y='FirstBlood',kind='pie',colors=['tomato','lightskyblue'], figsize=(5,5),labels=['red','blue'],autopct='%1.2f%%')
# find the correlation between variables
corr = data_1[data_1.columns[1:]].corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(220, 10, n=20),
    square=True)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')
print(corr)
# Use gold and experience diff to predict winning team
x = data_1[['blueGoldDiff','blueExperienceDiff']]
y = data_1[['blueWins']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1) # 80% training and 20% test
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy")
# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(x_test)
# Model Accuracy
print("Accuracy of DT using gold and xp diff to predict winning team:\n",metrics.accuracy_score(y_test, y_pred))
# Use all variables other than gold and experience difference to predict winning team
x = data_1[data_1.columns.difference(['blueGoldDiff','blueExperienceDiff','blueWins','gameId'])]
y = data_1[['blueWins']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1) # 80% training and 20% test
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy")
# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(x_test)
# Model Accuracy
print("Accuracy of DT using gold and xp diff to predict winning team:\n",metrics.accuracy_score(y_test, y_pred))
