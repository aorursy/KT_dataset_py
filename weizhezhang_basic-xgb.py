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
import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



from tqdm import tqdm

import xgboost as xgb



data_path = '../input/'

train = pd.read_csv(data_path + 'train.csv')

test = pd.read_csv(data_path + 'test.csv')
#print(train.head())

#print(test.head())




X = np.array(train.drop(['target'], axis=1))

y = train['target'].values



X_test = np.array(test.drop(['id'], axis=1))

ids = test['id'].values



X_train, X_valid, y_train, y_valid = train_test_split(X, y, \

    test_size=0.2, random_state=0)



d_train = xgb.DMatrix(X_train, label=y_train)

d_valid = xgb.DMatrix(X_valid, label=y_valid) 

d_test = xgb.DMatrix(X_test)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]

del X_train, X_valid, y_train, y_valid



# Train model, evaluate and make predictions

params = {}

params['objective'] = 'binary:logistic'

params['eta'] = 0.75

params['max_depth'] = 2

params['silent'] = 1

params['eval_metric'] = 'auc'



model = xgb.train(params, d_train, 100, watchlist, early_stopping_rounds=20, \

    maximize=True, verbose_eval=5)

model.save_model("./model_file_name")

model = xgb.Booster(params)

model.load_model("./model_file_name")

p_test = model.predict(d_test)



# Prepare submission

print(len(ids), len(p_test))

subm = pd.DataFrame()

subm['id'] = ids

subm['target'] = p_test

subm.to_csv('./submission.csv', index=False)