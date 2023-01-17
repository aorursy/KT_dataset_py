import pandas as pd

import pydot

from sklearn import tree

from time import time

import numpy as np

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.model_selection import train_test_split
train_data = pd.read_csv("train.csv",parse_dates = ['Dates'])



train_data.head()
test_data = pd.read_csv('test.csv', parse_dates=['Dates'])

test_data.head()
from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.metrics import log_loss

from sklearn.linear_model import LogisticRegression



labeled_data = preprocessing.LabelEncoder()

crime = labeled_data.fit_transform(train_data.Category)

 

#Get binarized weekdays, districts, and hours.

days = pd.get_dummies(train_data.DayOfWeek)

district = pd.get_dummies(train_data.PdDistrict)

hour = pd.get_dummies(train_data.Dates.dt.hour) 

 

#Build new array

train_data_new = pd.concat([hour, days, district], axis=1)

train_data_new['crime']=crime

 

#Repeat for test data

days = pd.get_dummies(test_data.DayOfWeek)

district = pd.get_dummies(test_data.PdDistrict)



hour = pd.get_dummies(test_data.Dates.dt.hour) 

 

test_data_new = pd.concat([hour, days, district], axis=1)

features = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',

 'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',

 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
train_X ,validation_X,train_y, validation_y = train_test_split(train_data_new[features],train_data_new['crime']  ,train_size = 0.7, test_size = 0.3)

from sklearn import metrics

dt_clf = DecisionTreeClassifier()

dt_clf.fit(train_X,train_y)

predicted = np.array(dt_clf.predict_proba(validation_X))

log_loss(validation_y, predicted) 
from sklearn.model_selection import cross_val_score

from sklearn import metrics

for depth in range(1,10):

    tree_clf = DecisionTreeClassifier(max_depth=depth, random_state=0)

    if tree_clf.fit(train_X,train_y).tree_.max_depth < depth:

        break

    score = np.mean(cross_val_score(tree_clf, train_X, train_y,scoring=metrics.make_scorer(log_loss, needs_proba=True, labels=train_y), cv=3, n_jobs=1))

    print("Depth: %i Accuracy: %.3f" % (depth,score))


dt_clf = DecisionTreeClassifier(max_depth=7)

dt_clf.fit(train_X,train_y)

predicted = np.array(dt_clf.predict_proba(validation_X))

log_loss(validation_y, predicted) 
dt_clf = DecisionTreeClassifier(max_depth=7)

dt_clf.fit(train_data_new[features],train_data_new['crime'])

predicted = np.array(dt_clf.predict_proba(test_data_new[features]))

result=pd.DataFrame(predicted, columns=labeled_data.classes_)

result.to_csv('testOutput.csv', index = True, index_label = 'Id' )