# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
### Carregar Dados

train_df = pd.read_csv("../input/matchesheader.csv")



### Visualizar Dados

train_df.head()
names_col= list(train_df.columns.values)

new_names_col = map(lambda name : name.strip(),names_col)

train_df.columns = new_names_col
### Visualizar colunas

list(train_df.columns.values)
### Checando por valores nulos

train_df.isna().sum()
###Tranformar team_name em numero para facilitar processamento: *1 para team_1_name, 0 para team_2_name*

from sklearn.preprocessing import LabelEncoder

labelEncoder = LabelEncoder()

train_df.team_1_name = labelEncoder.fit_transform(train_df.team_1_name)

train_df.team_2_name = labelEncoder.fit_transform(train_df.team_2_name)

train_df.winning_team = labelEncoder.fit_transform(train_df.winning_team)

train_df.head()
train_df = train_df.drop(["id","queue_id","team_1_win","team_2_win","winning_team","season_id"],axis=1)

train_df.head()
data = train_df.game_duration

def gameTimeType(time):

    if time <= 1200:

      return 1

    elif time <= 1500:

      return 2

    elif time <= 1800:

      return 3

    elif time <= 2100:

      return 4

    else:

      return 5

        

new_types= [gameTimeType(x) for x in data]

new_types

train_df.game_duration = new_types
data = train_df.game_duration

fig, ax = plt.subplots()

data.value_counts().plot(ax=ax, kind='bar')
data = []

for x in range(1, 5):

    data.append(train_df["team_1_player_{}_champion_id".format(x)])

for x in range(1, 5):

    data.append(train_df["team_2_player_{}_champion_id".format(x)])





fig, ax = plt.subplots()

data = pd.concat(data)

data.value_counts().plot(ax=ax, kind='bar',figsize=(40, 30),fontsize=16,)
data = []

for x in range(1, 5):

    data.append(train_df["team_1_player_{}_spell_1".format(x)])

for x in range(1, 5):

    data.append(train_df["team_1_player_{}_spell_2".format(x)])

for x in range(1, 5):

    data.append(train_df["team_2_player_{}_spell_1".format(x)])

for x in range(1, 5):

    data.append(train_df["team_2_player_{}_spell_2".format(x)])



fig, ax = plt.subplots()

data = pd.concat(data)

data.value_counts().plot(ax=ax, kind='bar')
classe = train_df['game_duration']

atributos = train_df.drop('game_duration', axis=1)

atributos.head()
from sklearn.model_selection import train_test_split

atributos_train, atributos_test, class_train, class_test = train_test_split(atributos, classe, test_size = 0.25 )



atributos_train.describe()
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=3, random_state =0, min_samples_leaf=3, min_samples_split=3  )

model = dtree.fit(atributos_train, class_train)
from sklearn.metrics import accuracy_score

classe_pred = model.predict(atributos_test)

acc = accuracy_score(class_test, classe_pred)

print("My Decision Tree acc is {}".format(acc))