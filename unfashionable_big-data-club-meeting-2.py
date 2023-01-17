# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.tsv",sep="\t")

train.head()
train['target'] = np.log1p(train['price'])
train.drop('price',axis = 1,inplace=True)
train['category_name'] = pd.factorize(train['category_name'])[0]
train['name'] = pd.factorize(train['name'])[0]

train['brand_name'] = pd.factorize(train['brand_name'])[0]

train['item_description'] = pd.factorize(train['item_description'])[0]
train.head()
train.isnull().sum()
train.drop('train_id',axis=1,inplace=True)
train.shape
test = pd.read_csv("../input/test.tsv",sep="\t")

test['category_name'] = pd.factorize(test['category_name'])[0]

test['name'] = pd.factorize(test['name'])[0]

test['brand_name'] = pd.factorize(test['brand_name'])[0]

test['item_description'] = pd.factorize(test['item_description'])[0]
from sklearn import *
train.columns
x_train, x_valid, y_train, y_valid = model_selection.train_test_split(train[train.columns[:-1]], train['target'], test_size=0.25)
import xgboost as xgb

dtrain = xgb.DMatrix(x_train, y_train)

dvalid  = xgb.DMatrix(x_valid,  y_valid)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

params = {}

params['eta'] = 0.75

params['max_depth'] = 5

params['seed'] = 99

params['tree_method'] = 'hist'

model = xgb.train(params, dtrain, 20, watchlist, verbose_eval=10, early_stopping_rounds=20)
test.head()
test['price'] = model.predict(xgb.DMatrix(test[test.columns[1:]]))
test['price'].head()
test['target'] = np.expm1(test['price'])
test.head()
sample_submission_file = pd.read_csv("../input/sample_submission.csv")

sample_submission_file.head()
test[['test_id','target']].to_csv("submission_test.csv",index=False)