import numpy as np

import pandas as pd

import lightgbm as lgb

import copy



def convert_table(df_input):

    df = copy.deepcopy(df_input)

    df.loc[:, 'bin_3':'bin_4'] = df.loc[:, 'bin_3':'bin_4'].applymap(lambda x: 'FTNY'.find(x) % 2)

    df = pd.get_dummies(df, columns=['nom_{}'.format(i) for i in range(5)])

    df.loc[:, 'nom_5':'nom_9'] = df.loc[:, 'nom_5':'nom_9'].applymap(lambda x: int(x, 16))

    df['ord_1'] = df['ord_1'].map(lambda x: ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'].index(x))

    df['ord_2'] = df['ord_2'].map(lambda x: ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot'].index(x)) 

    df['ord_3'] = df['ord_3'].map(lambda x: (ord(x) - ord('a')))

    df['ord_4'] = df['ord_4'].map(lambda x: (ord(x) - ord('A')))

    df['ord_6'] = df['ord_5'].map(lambda x: (ord(x[1]) - ord('A')))

    df['ord_5'] = df['ord_5'].map(lambda x: (ord(x[0]) - ord('A')))

    return df



df_train = convert_table(pd.read_csv('../input/cat-in-the-dat/train.csv'))

df_test = convert_table(pd.read_csv('../input/cat-in-the-dat/test.csv'))

X_train, T_train, X_test = (df_train.drop(['id', 'target'], axis=1), df_train['target'], df_test.drop('id', axis=1))



params = {'objective':'binary', 'metric':'binary_logloss'}

model = lgb.train(params, lgb.Dataset(X_train, T_train))

pd.concat([df_test['id'], pd.Series(model.predict(X_test), name='target')], axis=1).to_csv('submission.csv', index=False)