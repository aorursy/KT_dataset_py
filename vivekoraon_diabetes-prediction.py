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

%matplotlib inline
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.isnull().sum()
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
X = df.drop('Outcome',axis = 1)

y = df['Outcome']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)
LR = LogisticRegression().fit(X_train,y_train)
predict_lr = LR.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predict_lr))

print('\n')

print(confusion_matrix(y_test,predict_lr))
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train,y_train)

predict_knn = knn.predict(X_test)

print(classification_report(y_test,predict_knn))

print('\n')

print(confusion_matrix(y_test,predict_knn))
svc = SVC().fit(X_train,y_train)

predict_svc = svc.predict(X_test)

print(classification_report(y_test,predict_svc))

print('\n')

print(confusion_matrix(y_test,predict_svc))
dtc = DecisionTreeClassifier().fit(X_train,y_train)

predict_dtc = dtc.predict(X_test)

print(classification_report(y_test,predict_dtc))

print('\n')

print(confusion_matrix(y_test,predict_dtc))
rfc = RandomForestClassifier(n_estimators = 100).fit(X_train,y_train)

predict_rfc = rfc.predict(X_test)

print(classification_report(y_test,predict_rfc))

print('\n')

print(confusion_matrix(y_test,predict_rfc))