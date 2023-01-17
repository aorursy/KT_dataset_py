import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head(1)

first_digit = train.iloc[3].drop('label').values.reshape(28, 28)
plt.imshow(first_digit)

# Splitting the train data further into train and test sets
data_train, data_test = train_test_split(train, test_size = 0.3, random_state = 100)

data_train_x = data_train.drop('label', axis = 1)
data_train_y = data_train['label']

data_test_x = data_test.drop('label', axis = 1)
data_test_y = data_test['label']
model_DT = DecisionTreeClassifier(random_state = 100)
model_DT.fit(data_train_x, data_train_y)

test_pred_DT = model_DT.predict(data_test_x)
pred_DT_df = pd.DataFrame({'actual': data_test_y, 'predicted': test_pred_DT})
pred_DT_df['pred_status'] = pred_DT_df['actual'] == pred_DT_df['predicted']

accuracy_DT = pred_DT_df['pred_status'].sum() / pred_DT_df.shape[0] * 100
print('Accuracy:%.2f' % accuracy_DT)
model_RF = RandomForestClassifier(random_state = 100)
model_RF.fit(data_train_x, data_train_y)

test_pred_RF = model_RF.predict(data_test_x)
pred_RF_df = pd.DataFrame({'actual': data_test_y, 'predicted': test_pred_RF})
pred_RF_df['pred_status'] = pred_RF_df['actual'] == pred_RF_df['predicted']

accuracy_RF = pred_RF_df['pred_status'].sum() / pred_RF_df.shape[0] * 100
print('Accuracy:%.2f' % accuracy_RF)
model_AB = AdaBoostClassifier(random_state = 100)
model_AB.fit(data_train_x, data_train_y)

test_pred_AB = model_AB.predict(data_test_x)
pred_AB_df = pd.DataFrame({'actual': data_test_y, 'predicted': test_pred_AB})
pred_AB_df['pred_status'] = pred_AB_df['actual'] == pred_AB_df['predicted']

accuracy_AB = pred_AB_df['pred_status'].sum() / pred_AB_df.shape[0] * 100
print('Accuracy:%.2f' % accuracy_AB)
model_KNN = KNeighborsClassifier(n_neighbors = 5)
model_KNN.fit(data_train_x, data_train_y)

test_pred_KNN = model_KNN.predict(data_test_x)
pred_KNN_df = pd.DataFrame({'actual': data_test_y, 'predicted': test_pred_KNN})
pred_KNN_df['pred_status'] = pred_KNN_df['actual'] == pred_KNN_df['predicted']

accuracy_KNN = pred_KNN_df['pred_status'].sum() / pred_KNN_df.shape[0] * 100
print('Accuracy:%.2f' % accuracy_KNN)
train_x = train.drop('label', axis = 1)
train_y = train['label']

model_submission = KNeighborsClassifier(n_neighbors = 5)
model_submission.fit(train_x, train_y)

test_pred_submission = model_submission.predict(test)
submission = pd.DataFrame({'ImageId': list(range(1, len(test_pred_submission) + 1)), 'Label': test_pred_submission})
submission.to_csv('Submission.csv', index = False)
