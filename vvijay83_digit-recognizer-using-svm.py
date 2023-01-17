# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

from sklearn.preprocessing import scale

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA
import pandas as pd

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
print(train.head())

print(train.shape)

print(test.head())

print(test.shape)
# Looking at the labels they are balanced properly.

print(train.label.value_counts())

print(round(100*(train.label.value_counts()/train.label.shape[0])), 2)

sns.countplot(x='label', data=train)
# There are no null values in the training dataset

train.isnull().sum().to_frame(name='counts').query('counts > 0')
X = train.iloc[:, 1:]

y = train.iloc[:, :1]



X = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=4)
#Linear Model with default parameters

model_linear = SVC(kernel='linear')

model_linear.fit(X_train, y_train)

y_pred_linear = model_linear.predict(X_test)

print("Accuracy Score for SVM Linear is:  ", metrics.accuracy_score(y_test, y_pred_linear))
#SVM RBF Model with default parameters

non_linear_model = SVC(kernel='rbf')

non_linear_model.fit(X_train, y_train)

y_pred_rbf = non_linear_model.predict(X_test)

print("Accuracy Score for SVM RBF is:  ", metrics.accuracy_score(y_test, y_pred_rbf))
#Non Linear Model with best C & Gamma Parameter

non_linear_model_1 = SVC(kernel='rbf', C=10, gamma=0.001)

non_linear_model_1.fit(X_train, y_train)

y_pred_rbf_1 = non_linear_model_1.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred_rbf_1))
test_pred = test

test_pred = scale(test_pred)

y_pred = non_linear_model_1.predict(test_pred)
y_pred = pd.DataFrame(y_pred)

test['Label'] = y_pred

test['ImageId'] = test.index +1

submission = test[['ImageId','Label']]

submission.to_csv('submission.csv', index=False)