# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print(train.shape, test.shape)


# Any results you write to the current directory are saved as output.

train.head()


from sklearn.model_selection import train_test_split
train_1  ,train_2  = train_test_split(train,train_size = 0.3,random_state =100)
train_1_x  = train_1.drop('label',axis=1)
train_2_x = train_2.drop('label', axis=1)
train_1_y = train_1['label']
train_2_y  =train_2['label']
train_1_x.shape

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=100)
model.fit(train_1_x,train_1_y )
test_pred = model.predict(train_2_x)
df_pred = pd.DataFrame({'actual': train_2_y,
                        'predicted': test_pred})
df_pred['pred_status'] = df_pred['actual'] == df_pred['predicted']
df_pred['pred_status'].sum() / df_pred.shape[0] * 100

train_1_y = train_1['label']
train_2_y = train_2['label']
train_1_x = train_1.drop('label', axis=1)
train_2_x = train_2.drop('label', axis=1)

from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(random_state=100)
model_rf.fit(train_1_x, train_1_y)
train_2_pred = model_rf.predict(train_2_x)
df_pred = pd.DataFrame({'actual': train_2_y,
                        'predicted': train_2_pred})
df_pred['pred_status'] = df_pred['actual'] == df_pred['predicted']
df_pred['pred_status'].sum() / df_pred.shape[0] 
from sklearn import neighbors
train_1_x=train_1.drop('label',axis=1)
train_1_y=train_1['label']
train_2_x=train_2.drop('label',axis=1)
train_2_y=train_2['label']
model_knn=neighbors.KNeighborsClassifier(n_neighbors=5)
model_knn.fit(train_1_x,train_1_y)
train_2_pred=model_knn.predict(train_2_x)
df_pred = pd.DataFrame({'actual': train_2_y,
                        'predicted': train_2_pred})
df_pred['pred_status'] = df_pred['actual'] == df_pred['predicted']
df_pred['pred_status'].sum() / df_pred.shape[0] * 100
from sklearn.ensemble import AdaBoostClassifier
train_1_x=train_1.drop('label',axis=1)
train_1_y=train_1['label']
train_2_x=train_2.drop('label',axis=1)
train_2_y=train_2['label']
model_ab=AdaBoostClassifier(random_state=100)
model_ab.fit(train_1_x,train_1_y)
train_2_pred=model_ab.predict(train_2_x)
df_pred = pd.DataFrame({'actual': train_2_y,
                        'predicted': train_2_pred})
df_pred['pred_status'] = df_pred['actual'] == df_pred['predicted']
df_pred['pred_status'].sum() / df_pred.shape[0] * 100
#rint(train_2_y,train_2_pred)
test_pred = model_knn.predict(test)
df_test_pred  =pd.DataFrame(test_pred, columns=['Label'])
df_test_pred['ImageId']=test.index+1

df_test_pred[['ImageId','Label']].to_csv('sample_submission.csv',index=False)


