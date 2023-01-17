# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from xgboost import plot_importance

from sklearn.metrics import r2_score

import plotly.graph_objects as go

import plotly.express as px

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
data = pd.read_csv(

    '/kaggle/input/nba-players-stats/Seasons_Stats.csv')

data.head()
data = data.dropna(subset=['Age'])



player_le = LabelEncoder()

player_le.fit(data['Player'].values)

data['Player'] = player_le.transform(data['Player'])



data = data.drop(['Pos', 'Tm', 'blanl',

                  'blank2', 'Unnamed: 0', 'Year'], axis=1)

    

data.head()
X = data.drop(['Age'], axis=1)

y = data['Age']

Xtrain, Xtest, ytrain, ytest = train_test_split(

    X, y, random_state=42, test_size=0.4)

Xtrain.shape
model = xgb.XGBRegressor(max_depth=10, subsample=1,

                         n_estimators=1000, learning_rate=0.03, gamma=0.1)

model.fit(Xtrain, ytrain)

y_predict = model.predict(Xtest)



print('R2:', r2_score(ytest, y_predict))
line_data = pd.DataFrame()

line_data['test'] = ytest

line_data = line_data.reset_index().drop(['index'], axis=1)

line_data['predict'] = y_predict



fig = go.Figure()

fig.add_trace(go.Scatter(x=line_data.index, y=line_data['test'],

                         mode='lines',

                         name='test'))

fig.add_trace(go.Scatter(x=line_data.index, y=line_data['predict'],

                         mode='lines',

                         name='predict'))

fig.show()



fig = go.Figure()

line_data50 = line_data[0:50]

fig.add_trace(go.Scatter(x=line_data50.index, y=line_data50['test'],

                         mode='lines',

                         name='test'))

fig.add_trace(go.Scatter(x=line_data50.index, y=line_data50['predict'],

                         mode='lines',

                         name='predict'))

fig.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

plot_importance(model, ax=ax)
minAge = data['Age'].min()

maxAge = data['Age'].max()

fig = go.Figure(

    data=[go.Box(y=data.query(f'Age == {age}')['G'], name=f'{age}岁',

                )

          for age in range(int(minAge), int(maxAge)+1)],

    layout={'title': '比赛数量'})

fig.show()
minAge = data['Age'].min()

maxAge = data['Age'].max()

fig = go.Figure(

    data=[go.Box(y=data.query(f'Age == {age}')['FTr'], name=f'{age}岁',

                )

          for age in range(int(minAge), int(maxAge)+1)],

    layout={'title': '罚球率'})

fig.show()
minAge = data['Age'].min()

maxAge = data['Age'].max()

ftr_data = data.query('FTr < 1')

fig = go.Figure(

    data=[go.Box(y=ftr_data.query(f'Age == {age}')['FTr'], name=f'{age}岁',

                )

          for age in range(int(minAge), int(maxAge)+1)],

    layout={'title': '罚球率'})

fig.show()
minAge = data['Age'].min()

maxAge = data['Age'].max()

fig = go.Figure(

    data=[go.Box(y=data.query(f'Age == {age}')['PER'], name=f'{age}岁',

                )

          for age in range(int(minAge), int(maxAge)+1)],

    layout={'title': '霍林格球员效率值'})

fig.show()
minAge = data['Age'].min()

maxAge = data['Age'].max()

per_data = data.query('PER <= 30 & PER >= -30')

fig = go.Figure(

    data=[go.Box(y=per_data.query(f'Age == {age}')['PER'], name=f'{age}岁',

                )

          for age in range(int(minAge), int(maxAge)+1)],

    layout={'title': '霍林格球员效率值'})

fig.show()