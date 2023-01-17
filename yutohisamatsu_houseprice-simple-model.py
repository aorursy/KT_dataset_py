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
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

train.head()
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

test.head()
all_data = pd.concat((train, test)).reset_index(drop = True)
# log transform the target:

train['SalePrice'] = np.log1p(train['SalePrice'])

y = train['SalePrice']
# objectを除外 = 数値変数→欠損を中央値で

for feature in all_data.select_dtypes(exclude=['object', 'category']).columns:

        all_data[feature].fillna(all_data[feature].median(), inplace = True)



# objectを抽出 = カテゴリ変数→欠損を最頻値で

for feature in all_data.select_dtypes(include=['object']).columns: 

        all_data[feature].fillna(all_data[feature].value_counts().idxmax(), inplace = True)
all_data = pd.get_dummies(all_data)

all_data.shape
all_data.drop('SalePrice', axis=1, inplace=True)
train = all_data[:len(train)]

test = all_data[len(train):]

y = y
print(train.shape)

print(test.shape)
from sklearn.model_selection import train_test_split, KFold

train_x, test_x, train_y, test_y = train_test_split(train, y, test_size=0.3, random_state=0)
from sklearn.linear_model import ElasticNet

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.metrics import mean_squared_error



# Define Model

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=3))
ENet.fit(train, y)
def predict_cv(model, train_x, train_y, test_x):

    preds = []

    preds_test = []

    va_indexes = []

    kf = KFold(n_splits=4, shuffle=True, random_state=6785)

    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する

    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)): # enumerate 配列要素とインデックスも同時に取得する

        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]

        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        model.fit(tr_x, tr_y)

        tr_pred = model.predict(tr_x)

        pred = model.predict(va_x)

        preds.append(pred)

        pred_test = model.predict(test_x)

        preds_test.append(pred_test)

        va_indexes.append(va_idx)

        print('  score Train : {:.6f}' .format(np.sqrt(mean_squared_error(tr_y, tr_pred))), 

              '  score Valid : {:.6f}' .format(np.sqrt(mean_squared_error(va_y, pred)))) 

    # バリデーションデータに対する予測値を連結し、その後元の順番に並べなおす

    va_indexes = np.concatenate(va_indexes)

    preds = np.concatenate(preds, axis=0)

    order = np.argsort(va_indexes)

    pred_train = pd.DataFrame(preds[order])

    # テストデータに対する予測値の平均をとる

    preds_test = pd.DataFrame(np.mean(preds_test, axis=0))

    print('Score : {:.6f}' .format(np.sqrt(mean_squared_error(train_y, pred_train))))

    return pred_train, preds_test, model
# Cross Validation

predict_cv(ENet, train_x, train_y, test_x)
# テストデータ 推測

ENet_pred = np.expm1(ENet.predict(test))
sub = pd.DataFrame()

sub['Id'] = test['Id']

sub['SalePrice'] = ENet_pred

sub.to_csv('submission.csv',index=False)