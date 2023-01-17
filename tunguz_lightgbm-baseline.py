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
import pandas as pd

import lightgbm as lgb

from sklearn.model_selection import train_test_split



df = pd.read_csv('../input/Kannada-MNIST/train.csv')

col = ['pixel%d'%i for i in range(784)]



lgb_params = {

    "objective" : "multiclass",

    "metric" : "multi_logloss",

    "num_class" : 10,

    "max_depth" : 12,

    "num_leaves" : 15,

    "learning_rate" : 0.05,

    "bagging_fraction" : 0.9,

    "feature_fraction" : 0.9,

    "lambda_l1" : 0.01,

    "lambda_l2" : 0.0,

}



X_train, X_test, Y_train, Y_test = train_test_split(df[col], df['label'], test_size=0.1)



lgtrain = lgb.Dataset(X_train, label=Y_train)

lgtest = lgb.Dataset(X_test, label=Y_test)

lgb_clf = lgb.train(lgb_params, lgtrain, 1500, 

                    valid_sets=[lgtrain, lgtest], 

                    early_stopping_rounds=10, 

                    verbose_eval=20)



df = pd.read_csv('../input/Kannada-MNIST/test.csv')

res = lgb_clf.predict( df[col] ).argmax(axis=1)



df = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

df['label'] = res

df.to_csv('submission.csv', index=False)