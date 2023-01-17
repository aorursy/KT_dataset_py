# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import datetime
from datetime import datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")
train.head()
train.isnull().sum()
train.describe()
train.dtypes.sample(10)
train['date']  = train.datetime.apply(lambda x: x.split()[0])
train['hour'] = train.datetime.apply(lambda x: x.split()[1].split(':')[0])
train['weekday'] = train.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').weekday())
train['month'] = train.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').month)
train.dtypes.sample(15)
from sklearn.model_selection import train_test_split
train = train.drop(['datetime'], axis = 1)
#train = train.drop(['date'], axis = 1)
train = pd.get_dummies(train)
y = train["count"]
X = train.drop(['count'], axis=1)
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)


X.head()
from xgboost import XGBRegressor
my_model = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
my_model.fit(train_X, train_y, verbose=False)

print(my_model.predict(test_X))
my_model.score(test_X,test_y)