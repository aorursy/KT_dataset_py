# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output
train = pd.read_csv("../input/train.csv")

import matplotlib.pyplot as plt
%matplotlib inline
pd.set_option('display.precision', 5)
print(train.shape)
train_y=train.TARGET
train_x = train.iloc[:,:-1]
train_y.value_counts() / float(train_y.size)
print(train_y.value_counts())
train_X = train.drop(['ID','TARGET'],axis=1)
test_X = test.drop(['ID'],axis=1)
train_Y = train.TARGET
ids = test.ID
from sklearn.linear_model import LogisticRegressionCV

log = LogisticRegressionCV(cv=5)

log.fit(train_X,train_Y)

Y_pred_log = log.predict(test_X)

