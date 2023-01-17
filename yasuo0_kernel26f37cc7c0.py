# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/heart.csv")
data.head(5)
#data.isnull().sum()
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
train, test = train_test_split(data, test_size = 0.3, random_state = 100)

train_y = train['target']

test_y = test['target']



train_x = train.drop('target', axis = 1)

test_x = test.drop('target', axis = 1)
data['target'].value_counts()
# RandomForest

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=100,n_estimators=300,max_depth=60)

model.fit(train_x, train_y)



test_pred = model.predict(test_x)

df_pred = pd.DataFrame({'actual' : test_y,

                       'predicted' : test_pred})

df_pred['pred_status'] = df_pred['actual'] == df_pred['predicted']

df_pred['pred_status'].sum() / df_pred.shape[0] * 100
from sklearn.ensemble import AdaBoostClassifier
#AdaBoost

model = AdaBoostClassifier(random_state=100,n_estimators=600,learning_rate=0.6)

model.fit(train_x, train_y)



test_pred = model.predict(test_x)

df_pred = pd.DataFrame({'actual' : test_y,

                       'predicted' : test_pred})

#df_pred['pred_status'] = df_pred['actual'] == df_pred['predicted']

#df_pred['pred_status'].sum() / df_pred.shape[0] * 100

accuracy_score(test_y, test_pred)
from sklearn.neighbors import KNeighborsClassifier
#KNN Classifier

model_knn = KNeighborsClassifier(n_neighbors=4)



model_knn.fit(train_x, train_y)



test_pred_knn = model_knn.predict(test_x)



df_pred_knn = pd.DataFrame({'actual': test_y,

                        'predicted': test_pred_knn})



#df_pred_knn['pred_status'] = df_pred_knn['actual'] == df_pred_knn['predicted']



#df_pred['pred_status'].sum() / df_pred.shape[0] * 100

accuracy_score(test_y, test_pred)
# Decision Tree

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=100)

model.fit(train_x, train_y)



test_pred = model.predict(test_x)

df_pred = pd.DataFrame({'actual' : test_y,

                       'predicted' : test_pred})

#df_pred['pred_status'] = df_pred['actual'] == df_pred['predicted']

#acc = df_pred['pred_status'].sum() / df_pred.shape[0] * 100

#acc

accuracy_score(test_y, test_pred)