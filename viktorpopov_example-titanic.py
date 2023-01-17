# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import pandas as pd



%matplotlib inline 

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test  = pd.read_csv('/kaggle/input/titanic/test.csv')

sub_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train_target = train.Survived

test_id = test.PassengerId
train = train.drop(['Survived', 'PassengerId','Name', 'Ticket', 'Cabin'],axis=1)

test  = test.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
y = train_target
def ReplaceCatNan(data):

    for column in data.columns:

        if data[column].dtypes == object:

            data[column] = data[column].fillna(data[column].mode()[0])



ReplaceCatNan(train)

ReplaceCatNan(test)
cat_feat = list(train.dtypes[train.dtypes == object].index)

#закодируем пропущенные значений строкой, факт пропущенного значения тоже может нести в себе информацию

#train[cat_feat] = train[cat_feat].fillna('nan')

#отфильтруем непрерывные признаки

num_feat = [f for f in train if f not in (cat_feat)]# + ['Id', 'SalePrice'])]
cat_nunique = train[cat_feat].nunique()

print(cat_nunique)

cat_feat = list(cat_nunique[cat_nunique < 30].index)
dummy_data = pd.get_dummies(train[cat_feat], columns=cat_feat)



dummy_cols = list(set(dummy_data))



dummy_data = dummy_data[dummy_cols]





train = pd.concat([train[num_feat].fillna(-999),

                     dummy_data], axis=1)

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.3)
import xgboost as xgb
params = {'n_estimators': 300,

          'learning_rate': 0.01,

          'max_depth': 3,

          'min_child_weight': 1,

          'subsample': 1,

          'colsample_bytree': 1,

          'objective': 'reg:linear',

          'n_jobs': 4}

clf_xgb = xgb.XGBClassifier(**params)



clf_xgb.fit(X_train, y_train, eval_metric='auc', eval_set=[(X_train, y_train), (X_test, y_test)])
cat_feat = list(test.dtypes[test.dtypes == object].index)

#закодируем пропущенные значений строкой, факт пропущенного значения тоже может нести в себе информацию

#test[cat_feat] = test[cat_feat].fillna('nan')

#отфильтруем непрерывные признаки

num_feat = [f for f in test if f not in (cat_feat)]# + ['Id', 'SalePrice'])]
cat_nunique = test[cat_feat].nunique()

print(cat_nunique)

cat_feat = list(cat_nunique[cat_nunique < 30].index)
dummy_data = pd.get_dummies(test[cat_feat], columns=cat_feat)



dummy_cols = list(set(dummy_data))



dummy_data = dummy_data[dummy_cols]





test = pd.concat([test[num_feat].fillna(-999),

                     dummy_data], axis=1)

predict = clf_xgb.predict(test)#[:,1]
submission = pd.DataFrame({

    "PassengerId" : test_id,

    "Survived" : predict

})

submission.to_csv('pred_submission.csv', index=False)
submission = pd.read_csv('pred_submission.csv')

submission