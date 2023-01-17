# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

pd.options.display.precision = 7

# Any results you write to the current directory are saved as output.
# 読み込み
train = pd.read_csv("../input/train_V2.csv")
test = pd.read_csv("../input/test_V2.csv")

display(train.head())
display(test.head())
display(train.shape)
display(train.info())
display(train.describe())
#各カテゴリ変数のカテゴリ数
display('matchId category count={}'.format(train['matchId'].nunique()))
display('groupId category count={}'.format(train['groupId'].nunique()))
display('matchType category count={}'.format(train['matchType'].nunique()))


matchId = train['matchId'].value_counts()
groupId = train['groupId'].value_counts()
matchType = train['matchType'].value_counts()
# matchId種別出現頻度
plt.hist(matchId, bins=20)
plt.title("matchId")
plt.show()
# 欠損行カウント
train.isnull().sum()
train[train['winPlacePerc'].isnull()]
train = train.dropna()
matchType_categories = pd.get_dummies(train[['matchType']])
matchType_categories.head()

test_matchType_categories = pd.get_dummies(test[['matchType']])

test_Id = test['Id']

# groupId,matchIdはカテゴリ数が多く扱いに困るため削除
train_drop = train.drop(['Id','groupId','matchId','matchType'], axis=1)
train_drop = pd.merge(train_drop, matchType_categories, left_index=True, right_index=True)


test_drop = test.drop(['Id','groupId','matchId','matchType'], axis=1)
test_drop = pd.merge(test_drop, test_matchType_categories, left_index=True, right_index=True)
y_train = train_drop['winPlacePerc']
X_train = train_drop.drop(['winPlacePerc'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.25, random_state=1)
##import numpy as np
##from sklearn.ensemble import RandomForestRegressor
##from sklearn.model_selection import train_test_split

# Average CV score on the training set was:-0.0075825755105452825
##mod = RandomForestRegressor(bootstrap=True, max_features=0.6500000000000001, min_samples_leaf=11, min_samples_split=3, n_estimators=100, n_jobs=4)

##scoring = {"mae": "neg_mean_absolute_error", "mse": "neg_mean_squared_error"}

#kf = KFold(n_splits=3, random_state=1234, shuffle=True)
#scores = cross_validate(mod, X_train, y_train, cv=kf, scoring=scoring)
#display(scores)
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# Average CV score on the training set was:-0.058177168575242334
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        StackingEstimator(estimator=LassoLarsCV(normalize=False))
    ),
    GradientBoostingRegressor(alpha=0.9, learning_rate=0.1, loss="huber", max_depth=8, max_features=0.55, min_samples_leaf=16, min_samples_split=12, n_estimators=100, subsample=1.0)
)


#mod.fit(training_features, training_target)
#results = exported_pipeline.predict(testing_features)

X_train.shape
exported_pipeline.fit(X_train,y_train)

test_predict = exported_pipeline.predict(test_drop)

submission = pd.DataFrame({
    "Id": test_Id,
    "winPlacePerc": test_predict
})
submission.to_csv('submission2.csv', index=False)

