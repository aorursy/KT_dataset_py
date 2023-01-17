import numpy as np

import pandas as pd



from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import GridSearchCV , train_test_split , cross_val_score



from sklearn.linear_model import LogisticRegression

import os

import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv' , index_col = 'id')

test = pd.read_csv('../input/test.csv'  , index_col = 'id')

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

print(os.listdir("../input/"))
data = pd.read_csv("../input/train.csv")

X = data.iloc[:,2:].values

y = data.iloc[:,1:2].values
np.shape(test_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify = y)


clf_gini = DecisionTreeClassifier(max_depth=1, random_state = 100)

clf_gini.fit(X, y)

scores = cross_val_score(clf_gini, X, y, cv=5)

scores
clf = LogisticRegression(solver='liblinear',C= 0.1, max_iter=10000)

clf.fit(X, y)

scores = cross_val_score(clf, X, y, cv=5)

scores
test =  pd.read_csv("../input/test.csv")

test_data =test.iloc[:,1:]
test_data
ans= clf_gini.predict_proba(test_data.values)

submit = pd.read_csv('../input/sample_submission.csv')

submit['target'] = ans[:,1]

submit.to_csv('submit.csv', index = False)