# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head(1)
# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
first_digit = train.iloc[3].drop('label').values.reshape(28,28)
plt.imshow(first_digit)
#Spliting train data into data_train, data_test(validate) data
data_train, data_test = train_test_split(train, test_size=0.3, random_state=100)

data_train_x = data_train.drop('label', axis=1)
data_train_y = data_train['label']

data_test_x = data_test.drop('label', axis=1)
data_test_y = data_test['label']
#Decision Tree Model:
model_dt = DecisionTreeClassifier(random_state=100)
model_dt.fit(data_train_x, data_train_y)

test_pred_dt = model_dt.predict(data_test_x)
df_pred_dt = pd.DataFrame({'actual': data_test_y, 'predicted': test_pred_dt})
df_pred_dt['pred_status'] = df_pred_dt['actual'] == df_pred_dt['predicted']
#df_pred_dt.head()
acc_dt = df_pred_dt['pred_status'].sum() / df_pred_dt.shape[0] * 100
print('Accuracy:%d' % acc_dt)
#Random Forest Model:
model_rf = RandomForestClassifier(n_estimators=300, random_state=100)
model_rf.fit(data_train_x, data_train_y)

test_pred_rf = model_rf.predict(data_test_x)
df_pred_rf = pd.DataFrame({'actual': data_test_y, 'predicted': test_pred_rf})
df_pred_rf['pred_status'] = df_pred_rf['actual'] == df_pred_rf['predicted']
#df_pred_rf.head()
acc_rf = df_pred_rf['pred_status'].sum() / df_pred_rf.shape[0] * 100
print('Accuracy:%.2f' % acc_rf)
#AdaBoost Model:
model_ada = AdaBoostClassifier(random_state=100)
model_ada.fit(data_train_x, data_train_y)

test_pred_ada = model_ada.predict(data_test_x)
df_pred_ada = pd.DataFrame({'actual': data_test_y, 'predicted': test_pred_ada})
df_pred_ada['pred_status'] = df_pred_ada['actual'] == df_pred_ada['predicted']
#df_pred_ada.head()
acc_ada = df_pred_ada['pred_status'].sum() / df_pred_ada.shape[0] * 100
print('Accuracy:%.2f' % acc_ada)
#K-Nearest Neighbour Model:
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(data_train_x, data_train_y)

test_pred_knn = model_knn.predict(data_test_x)
df_pred_knn = pd.DataFrame({'actual': data_test_y, 'predicted': test_pred_knn})
df_pred_knn['pred_status'] = df_pred_knn['actual'] == df_pred_knn['predicted']
df_pred_knn.head()
acc_knn = df_pred_knn['pred_status'].sum() / df_pred_knn.shape[0] * 100
print('Accuracy:%.2f' % acc_knn)
#Final Model Submission:
train_x = train.drop('label', axis=1)
train_y = train['label']

model_final = KNeighborsClassifier(n_neighbors=5)
model_final.fit(train_x, train_y)

test_pred_final = model_final.predict(test)
submission = pd.DataFrame({'ImageId': list(range(1, len(test_pred_final)+1)), 'Label': test_pred_final})
submission.to_csv('Final_Submission.csv', index=False)
