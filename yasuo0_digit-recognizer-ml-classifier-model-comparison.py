# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
data = pd.read_csv("../input/train.csv")

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report
train1,validate = train_test_split(data,test_size=0.3,random_state=100) 

train_x = train1.drop('label', axis=1)
train_y = train1['label']

test_x = validate.drop('label', axis=1)
test_y = validate['label']

# Any results you write to the current directory are saved as output.
## Decision Tree
model = DecisionTreeClassifier(random_state=100,max_depth = 10)
model.fit(train_x, train_y)
test_pred = model.predict(test_x)
accuracy_dt = accuracy_score(test_y,test_pred)
accuracy_dt

## Random Forest
model_rf = RandomForestClassifier(random_state=100,n_estimators = 500)
model_rf.fit(train_x, train_y)

test_pred = model_rf.predict(test_x)
accuracy_rf = accuracy_score(test_y,test_pred)
accuracy_rf
## AdaBoost

model_ab  = AdaBoostClassifier(random_state=100)
model_ab.fit(train_x,train_y)
test_pred = model_ab.predict(test_x)
accuracy_ab = accuracy_score(test_y,test_pred)
accuracy_ab
test = pd.read_csv("../input/test.csv")
test_digit_recog = model_rf.predict(test)

df_pred_digit = pd.DataFrame(test_digit_recog, columns=['Label'])
df_pred_digit['ImageId'] = test.index + 1
df_pred_digit.head(10)
df_pred_digit[['ImageId', 'Label']].to_csv('assignment.csv', index=False)