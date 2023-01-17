import pandas as pd
train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv')
train['NflId'] = train['NflId'].astype('|S')



train = pd.concat([train.drop(['NflId'], axis=1), pd.get_dummies(train['NflId'], prefix='NflId')], axis=1)

dummy_col = train.columns
train.shape