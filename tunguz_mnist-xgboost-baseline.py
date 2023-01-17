# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import xgboost as xgb

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

sample_submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

col = ['pixel%d'%i for i in range(784)]
X_train, X_val, Y_train, Y_val = train_test_split(train[col], train['label'], test_size=0.1)

dtrain = xgb.DMatrix(X_train, label=Y_train)

dval = xgb.DMatrix(X_val, label=Y_val)
watchlist = [(dval, 'eval'), (dtrain, 'train')]



xgb_params = {

    "objective" : "multi:softmax",

    "eval_metric" : "mlogloss",

    "num_class" : 10,

    "max_depth" : 12,

    "eta" : 0.05,

    "subsample" : 0.9,

    "colsample_bytree" : 0.9,

}



xgb_clf = xgb.train(xgb_params, dtrain, 2000, 

                    watchlist, 

                    early_stopping_rounds=20, 

                    verbose_eval=20)
res_prob = xgb_clf.predict_proba(xgb.DMatrix(test[col]))

res = xgb_clf.predict( xgb.DMatrix(test[col]) ).astype(int)
sample_submission['Label'] = res

sample_submission.to_csv('submission.csv', index=False)

np.save('res_prob', res_prob)