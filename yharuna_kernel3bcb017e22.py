# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

from sklearn.linear_model import LogisticRegression

import sklearn.preprocessing as sp

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

import xgboost as xgb

from sklearn.model_selection import GridSearchCV



%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
train.Sex = train.Sex.astype('category')

train.Ticket = train.Ticket.astype('category')

train.Cabin = train.Cabin.astype('category')

train.Embarked = train.Embarked.astype('category')

test.Sex = test.Sex.astype('category')

test.Ticket = test.Ticket.astype('category')

test.Cabin = test.Cabin.astype('category')

test.Embarked = test.Embarked.astype('category')
# 両方のセットへ「is_train」のカラムを追加

# 1 = trainのデータ、0 = testデータ

train['is_train'] = 1

test['is_train'] = 0

 

# trainのprice(価格）以外のデータをtestと連結

train_test_combine = pd.concat([train.drop(['Survived'], axis=1),test],axis=0)

 

# 念のためデータの中身を表示させましょう

train_test_combine.head()
train_test_combine.Sex = train_test_combine.Sex.astype('category')

train_test_combine.Ticket = train_test_combine.Ticket.astype('category')

train_test_combine.Cabin = train_test_combine.Cabin.astype('category')

train_test_combine.Embarked = train_test_combine.Embarked.astype('category')



train_test_combine.Sex = train_test_combine.Sex.cat.codes

train_test_combine.Ticket = train_test_combine.Ticket.cat.codes

train_test_combine.Cabin = train_test_combine.Cabin.cat.codes

train_test_combine.Embarked = train_test_combine.Embarked.cat.codes

train_test_combine = train_test_combine.drop('Name',axis=1)

train_test_combine.head()
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)

imr = imr.fit(train_test_combine.values)

imputed_data = pd.DataFrame(imr.transform(train_test_combine.values))

imputed_data.isnull().sum()

imputed_data.columns = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','is_train']

imputed_data.head()
# 「is_train」のフラグでcombineからtestとtrainへ切り分ける

df_test = imputed_data.loc[imputed_data['is_train'] == 0]

df_train = imputed_data.loc[imputed_data['is_train'] == 1]

 

# 「is_train」をtrainとtestのデータフレームから落とす

df_test = df_test.drop(['is_train'], axis=1)

df_train = df_train.drop(['is_train'], axis=1)

 

# サイズの確認をしておきましょう

df_test.shape, df_train.shape
# df_trainへprice（価格）を戻す

df_train['Survived'] = train.Survived

 

# df_trainを表示して確認

df_train.head()
# x ＝ price以外の全ての値、y = price（ターゲット）で切り分ける

x_train, y_train = df_train.drop(['Survived'], axis=1), df_train.Survived



# モデルの作成

param_grid = {'max_depth': [2, 3, 4, 5, 6, 7],

              'min_samples_leaf': [1, 3, 5, 7, 10],

              'n_estimators':[3, 5, 7, 10, 15]}



m = GradientBoostingClassifier(random_state=0)

grid_search = GridSearchCV(m, param_grid, iid=True, cv=5, return_train_score=True)



# GridSearchCVは最良パラメータの探索だけでなく、それを使った学習メソッドも持っています

grid_search.fit(x_train, y_train)

print('best params: {}'.format(grid_search.best_params_))



# m = xgb.XGBClassifier()

# m.fit(x_train, y_train)



# スコアを表示

grid_search.score(x_train, y_train)
M = GradientBoostingClassifier(max_depth=5, min_samples_leaf=5, n_estimators=15, random_state=0)

M.fit(x_train, y_train)

# 作成したランダムフォレストのモデル「m」に「df_test」を入れて予測する

preds = M.predict(df_test)

 

# # 予測値 predsをnp.exp()で処理

#np.exp(preds)

 

# Numpy配列からpandasシリーズへ変換

preds = pd.Series(preds)



# テストデータのIDと予測値を連結

submit = pd.concat([gender_submission.PassengerId, preds], axis=1)

 

# カラム名をメルカリの提出指定の名前をつける

submit.columns = ['PassengerId', 'Survived']

 

# 提出ファイルとしてCSVへ書き出し

submit.to_csv('submit.csv', index=False)

 