# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_test = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')

submission = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/sample_submission_V2.csv')

df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')
id_list = ["Id"]

df.drop(id_list, axis=1, inplace=True)

df_test.drop(id_list, axis=1, inplace=True)
drop_match_type = df[df["matchType"]=="crashfpp"]

drop_match_type = drop_match_type.index

df.drop(index=drop_match_type, inplace=True)
drop_list = ["maxPlace", "killPoints"]

df.drop(drop_list, axis=1, inplace=True)

df_test.drop(drop_list, axis=1, inplace=True)
import category_encoders as ce



list_cols = ['groupId','matchId']



ce_oe = ce.OrdinalEncoder(cols=list_cols,handle_unknown='impute')

df = ce_oe.fit_transform(df, inplace=True)
ce_oe = ce.OrdinalEncoder(cols=list_cols,handle_unknown='impute')

df_test = ce_oe.fit_transform(df_test, inplace=True)
#Variables types Categorical

embarked = pd.concat([df["matchType"], df_test["matchType"]])



embarked_ohe = pd.get_dummies(embarked)



embarked_ohe_train = embarked_ohe[:4440679]

embarked_ohe_test = embarked_ohe[4440679:]



df = pd.concat([df, embarked_ohe_train], axis=1)

df_test = pd.concat([df_test, embarked_ohe_test], axis=1)
df["solo_game"]=0

df.loc[(df["solo"]==1) | (df["solo-fpp"]==1),"solo_game"]=1

df_test["solo_game"]=0

df_test.loc[(df_test["solo"]==1) | (df_test["solo-fpp"]==1),"solo_game"]=1
list_cols = ['matchType']



ce_oe = ce.OrdinalEncoder(cols=list_cols,handle_unknown='impute')

df = ce_oe.fit_transform(df, inplace=True)
list_cols = ['matchType']



# 序数をカテゴリに付与して変換

ce_oe = ce.OrdinalEncoder(cols=list_cols,handle_unknown='impute')

df_test = ce_oe.fit_transform(df_test, inplace=True)
match_list = ["squad-fpp","duo-fpp","solo-fpp","squad","duo","solo"]



df.drop(match_list, axis=1, inplace=True)

df_test.drop(match_list, axis=1, inplace=True)
match_list = ["normal-squad-fpp"]



df.drop(match_list, axis=1, inplace=True)

df_test.drop(match_list, axis=1, inplace=True)
drop_list = ['headshotKills','longestKill','teamKills','vehicleDestroys']



df.drop(drop_list, axis=1, inplace=True)

df_test.drop(drop_list, axis=1, inplace=True)
drop_list = ['groupId','matchId','roadKills']



df.drop(drop_list, axis=1, inplace=True)

df_test.drop(drop_list, axis=1, inplace=True)
from sklearn.model_selection import train_test_split, GridSearchCV



#LightGBMライブラリ

import lightgbm as lgb



train_set, test_set = train_test_split(df, test_size=0.3, random_state=4)
#訓練データを説明変数データ(X_train)と目的変数データ(y_train)に分割

X_train = train_set.drop('winPlacePerc', axis=1)

y_train = train_set['winPlacePerc']

 

#モデル評価用データを説明変数データ(X_train)と目的変数データ(y_train)に分割

X_test = test_set.drop('winPlacePerc', axis=1)

y_test = test_set['winPlacePerc']
lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_test, y_test)
params = {'metric': {'rmse'},

          'max_depth' : 10}
gbm = lgb.train(params,

                lgb_train,

                valid_sets=lgb_eval,

                num_boost_round=10000,

                early_stopping_rounds=100,

                verbose_eval=50)
predicted = gbm.predict(X_test)
Z_test = df_test.iloc[:, 0:]
Z_test = df_test.iloc[:, 0:]

light_pred = gbm.predict(Z_test)

light_pred.shape
light_pred = np.clip(light_pred, 0.00000000000000000, 1.00000000000000000)
submission['winPlacePerc'] = light_pred

submission.to_csv('submission.csv', index=False)