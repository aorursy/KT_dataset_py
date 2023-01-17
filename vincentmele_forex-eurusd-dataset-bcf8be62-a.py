import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import  train_test_split
from sklearn import metrics

# dataset01_eurusd4h.csv has 4479 rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('../input/dataset01_eurusd4h.csv', delimiter=',')
df1.dataframeName = 'dataset01_eurusd4h.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
#split
train, test = train_test_split(df1, test_size=0.2, shuffle=False, stratify = None )
train = train.values
test = test.values
#slice
X_train = train[:,:-1]
X_test = test[:,:-1]
y_train = train[:,-1]
y_test = test[:,-1]
import xgboost as xgb
from skopt import BayesSearchCV

from sklearn.ensemble import RandomForestClassifier
#########################################################################################
#Train

# Linear Regression
#clf = linear_model.LogisticRegression(max_iter=100).fit(X_train, y_train)


clf = RandomForestClassifier(random_state=0, n_estimators=750).fit(X_train, y_train)
clf.score(X_test, y_test)
#########################################################################################
#Evaluate
predictions = clf.predict(X_test)
print ('AUC:', metrics.roc_auc_score(y_test, predictions))
print ('Precision:', metrics.precision_score(y_test, predictions))
print ('Recall:', metrics.recall_score(y_test, predictions))
