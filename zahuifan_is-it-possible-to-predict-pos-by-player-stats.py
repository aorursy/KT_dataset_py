# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import learning_curve, validation_curve, GridSearchCV

from sklearn.preprocessing import LabelEncoder

from xgboost import plot_importance, plot_tree

import plotly.graph_objects as go

import plotly.express as px

import plotly as pl

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score

import xgboost as xgb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
raw_data = pd.read_csv(r'/kaggle/input/nba-players-stats/Seasons_Stats.csv')

raw_data.head()



player_data = pd.read_csv(r'/kaggle/input/nba-players-stats/Players.csv')

player_data = player_data[['height', 'weight', 'Player']]

player_data.head()



raw_data = pd.merge(raw_data, player_data, left_on=[

    'Player'], right_on=['Player']).copy()



data = raw_data.dropna(subset=['Pos']).copy()



# 丢弃Pos异常值

data.drop(index=data.query(

    'Pos not in ["C","PF","SF","SG","PG"]').index, inplace=True)



data.drop(['Unnamed: 0', 'Tm', 'Year', 'Player'], axis=1, inplace=True)



# 编码

pos_le = LabelEncoder()

pos_le.fit(data['Pos'].values)

data['Pos'] = pos_le.transform(data['Pos'])



# 训练集

X = data.drop(['Pos'], axis=1)

y = data['Pos']

Xtrain, Xtest, ytrain, ytest = train_test_split(

    X, y, random_state=42, test_size=0.4)



Xtrain.shape
model = xgb.XGBClassifier(max_depth=12, subsample=1, n_estimators=200)

                          # tree_method='gpu_hist', gpu_id=0)

model.fit(Xtrain, ytrain)

y_predict = model.predict(Xtest)



# 评估

train_score = model.score(Xtrain, ytrain)

print(f'train_score: {train_score}')



val_score = model.score(Xtest, ytest)

print(f'val_score: {val_score}')
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

plot_importance(model, ax=ax)
import plotly.graph_objects as go

fig_data = raw_data.drop(index=raw_data.query(

    'Pos not in ["C","PF","SF","SG","PG"]').index)

pos_group = fig_data.groupby('Pos')

fig = go.Figure(

    data=[go.Box(y=fig_data.query(f'Pos == "{pos}"')['weight'], name=f'{pos}')

          for pos in pos_group['weight'].median().sort_values().index],

    layout={'title': 'Weight'})

fig.show()



fig = go.Figure(

    data=[go.Box(y=fig_data.query(f'Pos == "{pos}"')['AST%'], name=f'{pos}')

          for pos in pos_group['AST%'].median().sort_values().index],

    layout={'title': 'AST%'})

fig.show()



fig = go.Figure(

    data=[go.Box(y=fig_data.query(f'Pos == "{pos}"')['FT%'], name=f'{pos}')

          for pos in pos_group['FT%'].median().sort_values().index],

    layout={'title': 'FT%'})

fig.show()
mat = confusion_matrix(ytest, y_predict)

labels = pos_le.inverse_transform(ytest.sort_values().unique())

sns.set(font_scale=0.8)

fig, ax = plt.subplots(1, 1, figsize=(7, 7))

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,

            ax=ax, xticklabels=labels, yticklabels=labels, annot_kws={"size": 8})

plt.xlabel('true label')

plt.ylabel('predicted label')