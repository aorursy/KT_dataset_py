# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier as GBC

# Input data files are available in the "../input/" directory.
# Any results you write to the current directory are saved as output.
def plot_hist_discrete(data, column):
    plt.figure(figsize=(16, 8))
    bins = data[column].nunique()
    plt.hist([data[data['Survived'] == 1.0][column].dropna(), data[data['Survived'] == 0.0][column].dropna()], color=['g','r'], 
             alpha=0.5, bins=bins, label=['alive', 'dead'])
    plt.xticks(data[column].drop_duplicates().dropna())
    plt.title(column)
    plt.legend()
    plt.grid()
    plt.show()
    
def plot_hist_numeric(data, column):
    plt.figure(figsize=(16, 8))
    sns.distplot(data[data['Survived'] == 1.0][column].dropna(), color = 'green', label='alive')
    sns.distplot(data[data['Survived'] == 0.0][column].dropna(), color = 'red', label='dead')
    plt.legend()
    plt.grid()
    plt.show()
np.random.seed(42)
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train
# Class affects survival chances 
plot_hist_discrete(train, 'Pclass')
# Sex affects survival chances
plot_hist_discrete(train, 'Sex')
# Age slightly affects survival chances
plot_hist_numeric(train, 'Age')
# SibSp doesn't affect survival chances
plot_hist_discrete(train, 'SibSp')
# Parch slightly affects survival chances
plot_hist_discrete(train, 'Parch')
# Fare affects survival chances
plot_hist_numeric(train, 'Fare')
# Embarkation slightly affects survival chances
plot_hist_discrete(train, 'Embarked')
train_processed = train.drop(columns=['PassengerId', 'Name', 'SibSp', 'Ticket', 'Cabin'])
train_processed['Sex'].replace({'male': 0, 'female': 1}, inplace=True)
train_processed['Embarked'].replace({'C': 0, 'S': 1, 'Q': 2}, inplace=True)

X_tr, y_tr = train_processed.drop(columns='Survived'), train_processed.loc[:, 'Survived']
test_processed = test.drop(columns=['Name', 'SibSp', 'Ticket', 'Cabin'])
test_processed['Sex'].replace({'male': 0, 'female': 1}, inplace=True)
test_processed['Embarked'].replace({'C': 0, 'S': 1, 'Q': 2}, inplace=True)

id_ts, X_ts = test_processed.loc[:, 'PassengerId'], test_processed.drop(columns=['PassengerId'])
X_tr_plus = X_tr.copy()
X_ts_plus = X_ts.copy()

for col in (col for col in X_tr.columns if X_tr[col].isnull().any()):
    X_tr_plus[col + '_was_missing'] = X_tr_plus[col].isnull()
    
for col in (col for col in X_ts.columns if X_ts[col].isnull().any()):
    X_ts_plus[col + '_was_missing'] = X_ts_plus[col].isnull()  
    
X_tr_plus.fillna({'Age': X_tr['Age'].mean(), 'Embarked': 3}, inplace=True)
X_ts_plus.fillna({'Age': X_ts['Age'].mean(), 'Fare': X_ts['Fare'].mean()}, inplace=True)
model = GBC()
model.fit(X_tr_plus, y_tr)
ans = model.predict(X_ts_plus)
answer = pd.concat([id_ts, pd.Series(ans, name='Survived')], axis=1)
answer.to_csv('answer.csv')