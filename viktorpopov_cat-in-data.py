# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
submission = pd.read_csv("../input/cat-in-the-dat/sample_submission.csv")
test = pd.read_csv("../input/cat-in-the-dat/test.csv")
train = pd.read_csv("../input/cat-in-the-dat/train.csv")
train.head(10).T
print(test.shape)
print(train.shape)
train['bin_3'] = train['bin_3'].replace(to_replace=['T', 'F'], value=['1', '0']).astype(int)
train['bin_4'] = train['bin_4'].replace(to_replace=['Y', 'N'], value=['1', '0']).astype(int)

test['bin_3'] = test['bin_3'].replace(to_replace=['T', 'F'], value=['1', '0']).astype(int)
test['bin_4'] = test['bin_4'].replace(to_replace=['Y', 'N'], value=['1', '0']).astype(int)
train.head(3)
nom_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

for i in nom_cols:
    print('The nimber of unique values in {} columns is : {}'.format(i, train[i].nunique()))
ord_col = [ 'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5'] 

for i in ord_col:
    print('The number of unique in {} columns is: {}'.format(i, train[i].nunique()))
target = train['target']
train_id = train['id']
test_id = test['id']

train.drop(['target', 'id'], axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

print(train.shape)
print(test.shape)
sns.countplot(target)
plt.title("labels counts")
plt.show()

data = pd.concat([train, test])
columns = [i for i in data.columns]
dummies = pd.get_dummies(data,columns=columns, drop_first=True,sparse=True)
train = dummies.iloc[:train.shape[0], :]
test = dummies.iloc[train.shape[0]:, :]
train = train.sparse.to_coo().tocsr()
test = test.sparse.to_coo().tocsr()
X_train,X_test,y_train,y_test=train_test_split(train,target,random_state=42,test_size=0.3)
import xgboost as xgb
params = {'n_estimators': 500,
          'learning_rate': 0.1,
          'max_depth': 5,
          'min_child_weight': 1,
          'subsample': 1,
          'colsample_bytree': 1,
          'n_jobs': -1}
clf_xgb = xgb.XGBClassifier(**params)

clf_xgb.fit(X_train, y_train, eval_metric='auc', eval_set=[(X_train, y_train), (X_test, y_test)])
predict = clf_xgb.predict(test)
submission = pd.DataFrame({'id':test_id,'target':predict})
submission.to_csv('submission.csv', index=False)
sub = pd.read_csv('submission.csv')
sub