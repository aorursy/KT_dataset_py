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
import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import tqdm

import seaborn as sns

from sklearn.metrics import mean_absolute_error



import xgboost as xgb



%matplotlib inline

%config InlineBackend.figure_format = 'retina'



pd.options.display.max_rows = 100
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

sample_submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
y_test = sample_submission.SalePrice
train.info()
print(train.shape)

print(test.shape)
plt.figure(figsize=(15,10))

sns.distplot(train.SalePrice)

plt.show()
# Defer target in advance

train_target = train.SalePrice
# Remove unnecessary columns in train and test

train = train.drop(['Id', 'SalePrice'], axis=1)

test = test.drop(['Id'],axis=1)
# Link them together for general processing

data = pd.concat([train, test])
cat_feat = list(data.dtypes[data.dtypes == object].index)
# encode the missing values with a string, the fact of the missing value can also carry information

data[cat_feat] = data[cat_feat].fillna('nan')

# filter out continuous features

num_feat = [f for f in data if f not in (cat_feat + ['Id', 'SalePrice'])]
cat_nunique = data[cat_feat].nunique()

print(cat_nunique)

cat_feat = list(cat_nunique[cat_nunique < 30].index)
dummy_data = pd.get_dummies(data[cat_feat], columns=cat_feat)



dummy_cols = list(set(dummy_data))



dummy_data = dummy_data[dummy_cols]





data = pd.concat([data[num_feat].fillna(-999),

                     dummy_data], axis=1)

data.head()
train = data.iloc[:train.shape[0], :]

test = data.iloc[train.shape[0]:, :]
X_train = train

y_train = train_target

X_test = test

y_test = sample_submission.SalePrice
params = {'n_estimators': 130,

          'learning_rate': 0.01,

          'max_depth': 3,

          'min_child_weight': 1,

          'subsample': 1,

          'colsample_bytree': 1,

          'objective': 'reg:linear',

          'n_jobs': 4}

clf_xgb = xgb.XGBRegressor(**params)



clf_xgb.fit(X_train, y_train, eval_metric='mae', eval_set=[(X_train, y_train), (X_test, y_test)])
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
# function to be MINIMIZED

def score(params):

    params['max_depth'] = int(params['max_depth'])

    params['n_jobs'] = -1

    print("Training with params : ", params)

    clf = xgb.XGBRegressor(**params)

    clf.fit(X_train, y_train)

    y_pred_xgb_test = clf.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_xgb_test)

    result = {'loss': mae, 'status': STATUS_OK}

    print('TEST ROC AUC: {0:.4f}'.format(mae))

    return result





space = {

            'max_depth' : hp.choice('max_depth', range(1, 15)),

            'n_estimators': hp.choice('n_estimators', range(100, 1000)),

            'eta': hp.quniform('eta', 0.025, 0.5, 0.10),

            'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),

            'min_child_weight': hp.quniform('min_child_weight', 1, 6, 0.5),

            'subsample': hp.quniform('subsample', 0.5, 1, 0.05),

            'gamma': hp.quniform('gamma', 0.5, 1, 0.05),

            'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),

            'eval_metric': 'mae',

            'objective': 'reg:linear',

            # Increase this number if you have more cores. Otherwise, remove it and it will default

            # to the maxium number.

            'nthread': 4,

            'booster': 'gbtree',

            'tree_method': 'exact',

            'silent': 1

        }



trials = Trials()



best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=20)
best
trials.best_trial
clf_xgb = xgb.XGBRegressor(**best)



clf_xgb.fit(X_train, y_train, eval_metric='mae', eval_set=[(X_train, y_train), (X_test, y_test)])
predict = clf_xgb.predict(X_test)
submission = pd.DataFrame({

        "Id": sample_submission["Id"],

        "SalePrice": predict

    })

submission.to_csv('submission.csv', index=False)
submission