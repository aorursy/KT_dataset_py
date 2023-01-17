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
# 学習データとテストデータの読み込み

train = pd.read_csv('/kaggle/input/titanic/train.csv') # 学習データ

test = pd.read_csv('/kaggle/input/titanic/test.csv') # テストデータ

# 学習データの中身チェック

train.head()
# テストデータの中身チェック

test.head()
# 学習データを特徴量と目的変数に分ける

train_x = train.drop(['Survived'], axis=1) # trainのSurvived列を削除する（※axis=1で列を示す）

train_y = train['Survived']
# テストデータは特徴量のみなのでそのまま使う

test_x = test.copy() # testデータをコピーしたリストをtest_xにセットする
from sklearn.preprocessing import LabelEncoder



# PassengerId列を削除

train_x = train_x.drop(['PassengerId'], axis=1)

test_x = test_x.drop(['PassengerId'], axis=1)

# Name, Ticket, Cabin列を削除

train_x = train_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)

test_x = test_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)



# それぞれのカテゴリ変数にlabel Encodingを適用する

for c in ['Sex', 'Embarked']:

    # le.fit()で、学習データに基づいてどう変換するかを決める。le.trainsform()する前のお約束。

    le = LabelEncoder()

    le.fit(train_x[c].fillna('NA'))

    

    # 学習データ、テストデータを変換する

    train_x[c] = le.transform(train_x[c].fillna('NA'))

    test_x[c] = le.transform(test_x[c].fillna('NA'))
train_x.head()
test_x.head()
from xgboost import XGBClassifier



# モデルの作成

model = XGBClassifier(n_estimator = 20, random_state=71)



# 特徴量(train_x)と目的変数(train_y)を与えての学習

model.fit(train_x, train_y)



# predict_proba()でテストデータの予測値を確率で出力する

# ちなみにラベルで出したい場合はpredict()でイケる

pred = model.predict_proba(test_x)[:, 1]



# テストデータの予測値を二値に変換する

# 50%超えているなら1それ以外は0って感じにする

pred_label = np.where(pred > 0.5, 1, 0)

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived':pred_label})

submission.to_csv('submission_first.csv', index=False)
import os

for dirname, _, filenames in os.walk('/kaggle/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))