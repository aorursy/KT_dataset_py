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
train = pd.read_csv('../input/train.csv', low_memory = False)
train.head()
print(np.mean(train.default == True))
ZIP_labels = np.unique(train.ZIP)

print(ZIP_labels)
train.groupby(by = 'ZIP').default.mean()
min_year_vector = (train.year == min(np.unique(train.year)))

print(sum((train.default == True) & min_year_vector)/sum(min_year_vector))
np.corrcoef(train.age, train.income)
y_train = train.default

X_train = pd.get_dummies(train[['loan_size', 'payment_timing', 'education', 'occupation', 'income','job_stability', 'ZIP', 'rent']])



X_train.head()
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators = 100, random_state = 42, oob_score = True)

model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score



in_sample_acc = accuracy_score(model.predict(X_train), y_train)

print(in_sample_acc)
print(model.oob_score_)
test = pd.read_csv('../input/test.csv')

y_test = test.default

X_test = pd.get_dummies(test[['loan_size', 'payment_timing', 'education', 'occupation', 'income','job_stability', 'ZIP', 'rent']])

oos_accuracy = accuracy_score(model.predict(X_test), y_test)

print(oos_accuracy)
predict_proba = model.predict_proba(X_test)

non_minority_vec = (test.minority == 0)*1

non_minority_default_probability = np.dot(predict_proba[:, 1], non_minority_vec)/sum(non_minority_vec)

print(non_minority_default_probability)
minority_vec = (test.minority == 1)*1

minority_default_probability = np.dot(predict_proba[:, 1], minority_vec)/sum(minority_vec)

print(minority_default_probability)
X_test.columns
print(np.dot(model.predict(X_test), (X_test.sex == 0))/sum(X_test.sex == 0))
np.dot(model.predict(X_test), np.ones((160000, 1)))
import matplotlib.pyplot as plt



a = np.multiply (predict_proba[:, 1]),(X_test.sex == 0)

print(a)