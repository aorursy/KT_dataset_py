# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

print('hello')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
sample = pd.read_csv('../input/train_sample.csv')
sample.drop(labels=['attributed_time'], inplace=True, axis=1)

X = pd.get_dummies(sample.drop(labels=['is_attributed'], axis=1), sparse=True).values
y = sample.is_attributed.values

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.33)


clf = LogisticRegression()
clf.fit(X_train, y_train)
