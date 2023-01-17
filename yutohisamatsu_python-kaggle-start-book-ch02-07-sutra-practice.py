# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd





train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')



data = pd.concat([train, test], sort=False)



data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

data['Age'].fillna(data['Age'].median(), inplace=True)

data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

data['IsAlone'] = 0

data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
data.head()
delete_columns = ['Name', 'PassengerId', 'Ticket', 'Cabin']

data.drop(delete_columns, axis=1, inplace=True)



train = data[:len(train)]

test = data[len(train):]



y_train = train['Survived']

X_train = train.drop('Survived', axis=1)

X_test = test.drop('Survived', axis=1)
X_train.head()
from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)
categorical_features = ['Embarked', 'Pclass', 'Sex']
params = {

    'objective':'binary',

    'max_bin': 300,

    'learning_rate': 0.05,

    'num_leaves': 40

}
import lightgbm as lgb



lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)



lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_features)



model = lgb.train(params, lgb_train, 

                      valid_sets = [lgb_train, lgb_eval],

                      verbose_eval = 10,

                      num_boost_round = 1000,

                      early_stopping_rounds = 10

                 )



y_pred = model.predict(X_test, num_iteration = model.best_iteration)
y_pred[:10]
y_pred = (y_pred > 0.5).astype(int)

y_pred[:10]
sub = pd.read_csv('../input/titanic/gender_submission.csv')



sub['Survived'] = y_pred

sub.to_csv('submission_lightgbm_holdout.csv', index=False)



sub.head()
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')



data = pd.concat([train, test], sort=False)



data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

data['Age'].fillna(data['Age'].median(), inplace=True)

data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

data['IsAlone'] = 0

data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1



delete_columns = ['Name', 'PassengerId', 'Ticket', 'Cabin']

data.drop(delete_columns, axis=1, inplace=True)



train = data[:len(train)]

test = data[len(train):]



y_train = train['Survived']

X_train = train.drop('Survived', axis=1)

X_test = test.drop('Survived', axis=1)



X_train.head()
# 交差検証のためのデータ分割を実装するためのKFold

from sklearn.model_selection import KFold



# 各分割でのX_testに対する予測値を格納するリスト

y_preds = []



# 各分割で学習したモデルを格納するリスト

models = []



# 各分割での検証用データセット(X_val)に対する予測値を格納するnumpy.ndarray

# trainのoof 各分割でのoofに対する予測値を収納

oof_train = np.zeros((len(X_train),))



# 分割方法の設定

# n_splits(分割数) データセットを5つに分ける shuffle=Trueで、データセットを分割前にシャッフル

cv = KFold(n_splits=5, shuffle=True, random_state=0)



categorical_features = ['Embarked', 'Pclass', 'Sex']



params = {

    'objective': 'binary',

    'max_bin': 300,

    'learning_rate': 0.05,

    'num_leaves': 40

}



# 設定した分割方法に基づいて、各分割での学習用・検証用データセットに対応するindexを取得

# enumerate　要素と同時に各分割のindex(fold_id)も取得

# 今回、fold_idは利用していないが、モデルを個別に保存する場合などにファイル名として使う

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):

    # 各fold_idで取得したindexに基づいて、データセットを分割

    # X_tr,X_val = pandas.DataFrame, y_tr,y_val = numpy.ndarray indexの指定方法が両者で異なる点に注意

    X_tr = X_train.loc[train_index, :]

    X_val = X_train.loc[valid_index, :]

    y_tr = y_train.loc[train_index]

    y_val = y_train.loc[valid_index]

    

    lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)

    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=categorical_features)

    

    model = lgb.train(params, lgb_train,

                          valid_sets = [lgb_train, lgb_eval],

                          verbose_eval = 10,

                          num_boost_round = 1000,

                          early_stopping_rounds = 10

                     )

    

    # 各分割での検証用データセット(X_val)に対する予測値を格納

    # oof_train(numpy.ndarray)

    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)

    

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    # 各分割でのX_testに対する予測値を、先に作成したy_predsというリストに格納

    y_preds.append(y_pred)

    # 各分割で学習したモデルを先に作成したmodelsというリストに格納

    models.append(model)
# to_csv()は、numpy.ndarrayでは使えないので、先にpandas.DataFrameに変換

pd.DataFrame(oof_train).to_csv('oof_train_kfold.csv', index=False)



scores = [

    m.best_score['valid_1']['binary_logloss'] for m in models

]



score = sum(scores) / len(scores)



print('----- CV scores -----')

print(scores)

print(score)
# 正解率を計算するためのaccuracy_score

from sklearn.metrics import accuracy_score



y_pred_oof = (oof_train > 0.5).astype(int)

accuracy_score(y_train, y_pred_oof)
# y_predsは、各分割でのデータを保存しているので、CV数と一致

len(y_preds)
# fold_idが0の場合の予測値は、以下のよう

y_preds[0][:10]
# y_predsに保存されている各分割での予測値を平均した値を、最終的なsubmitに利用する

y_sub = sum(y_preds) / len(y_preds)

y_sub = (y_sub > 0.5).astype(int)

y_sub[:10]
sub['Survived'] = y_sub

sub.to_csv('submission_lightgbm_kfold.csv', index=False)



sub.head()
from sklearn.model_selection import KFold



cv = KFold(n_splits=5, shuffle=True, random_state=0)



for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):

    X_tr = X_train.loc[train_index, :]

    X_val = X_train.loc[valid_index, :]

    y_tr = y_train.loc[train_index]

    y_val = y_train.loc[valid_index]

    

    # 「f文字列」・・・最初にfをつけ、変数は{}で括ることで文字列を中に取りいれられる

    print(f'fold_id: {fold_id}')

    print(f'y_tr y==1 rate: {sum(y_tr)/len(y_tr)}')

    print(f'y_val y==1 rate: {sum(y_val)/len(y_val)}')
from sklearn.model_selection import StratifiedKFold



cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)



# cv.split(X_train, y_train) StratifiedKFoldではy_trainをもとにした分割を実行

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):

    X_tr = X_train.loc[train_index, :]

    X_val = X_train.loc[valid_index, :]

    y_tr = y_train[train_index]

    y_val = y_train[valid_index]

    

    print(f'fold_id: {fold_id}')

    print(f'y_tr y==1 rate: {sum(y_tr)/len(y_tr)}')

    print(f'y_val y==1 rate: {sum(y_val)/len(y_val)}')
from sklearn.model_selection import StratifiedKFold





y_preds = []

models = []

oof_train = np.zeros((len(X_train),))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)



categorical_features = ['Embarked', 'Pclass', 'Sex']



params = {

    'objective': 'binary',

    'max_bin': 300,

    'learning_rate': 0.05,

    'num_leaves': 40

}



for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):

    X_tr = X_train.loc[train_index, :]

    X_val = X_train.loc[valid_index, :]

    y_tr = y_train[train_index]

    y_val = y_train[valid_index]



    lgb_train = lgb.Dataset(X_tr, y_tr,

                                             categorical_feature=categorical_features)

    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train,

                                            categorical_feature=categorical_features)



    model = lgb.train(params, lgb_train,

                                   valid_sets=[lgb_train, lgb_eval],

                                   verbose_eval=10,

                                   num_boost_round=1000,

                                   early_stopping_rounds=10)



    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)



    y_preds.append(y_pred)

    models.append(model)
pd.DataFrame(oof_train).to_csv('oof_train_skfold.csv', index=False)

print(oof_train[:10])



scores = [

    m.best_score['valid_1']['binary_logloss'] for m in models

]

score = sum(scores) / len(scores)

print('===CV scores===')

print(scores)

print(score)
from sklearn.metrics import accuracy_score





y_pred_oof = (oof_train > 0.5).astype(int)

accuracy_score(y_train, y_pred_oof)
y_sub = sum(y_preds) / len(y_preds)

y_sub = (y_sub > 0.5).astype(int)

y_sub[:10]
sub['Survived'] = y_sub

sub.to_csv('submission_lightgbm_skfold.csv', index=False)



sub.head()