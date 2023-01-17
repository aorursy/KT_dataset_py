import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



test_cols = ['PassengerId', 'Fare', 'Sex', 'Pclass']

train_cols = test_cols + ['Survived']

df_train = pd.read_csv('../input/train.csv', dtype={'Survived': int})[train_cols].dropna()

df_test = pd.read_csv('../input/test.csv')[test_cols].dropna()
df_train['Fare'].describe()
df_train['Sex'].unique()
df_train['Pclass'].unique()
df_train['FareBin'] = pd.cut(df_train['Fare'], [0, 10, 20, 30, 40])
group_cols = ['FareBin', 'Sex', 'Pclass']

df_survive = (df_train[group_cols + ['Survived']].dropna()

                                                 .groupby(group_cols)['Survived']

                                                 .mean()

                                                 .apply(lambda x: x > 0.5))
df_test['FareBin'] = pd.cut(df_test['Fare'], [0, 10, 20, 30, 40])



def predict(x):

    try:

        return df_survive[x['FareBin'], x['Sex'], x['Pclass']]

    except KeyError:

        return False



df_test['Survived'] = df_test.apply(predict, axis=1).astype(int)

df_test[['PassengerId', 'Survived']].to_csv('predict.csv', index=False)