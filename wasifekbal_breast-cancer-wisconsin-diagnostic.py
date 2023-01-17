# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df
df.shape
df.drop(columns='Unnamed: 32', inplace=True)
X = df.iloc[:,2:].values
X.shape
y=df['diagnosis']
y.shape
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=13,n_jobs=-1)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)
results = []

for i in range(1,25):

    knn = KNeighborsClassifier(n_neighbors=i,n_jobs=-1)

    knn.fit(X_train,y_train)

    y_pred=knn.predict(X_test)

    results.append(accuracy_score(y_pred,y_test))
max_accuracy = max(results)

results.index(max_accuracy)+1
print("highest accuracy = ",max_accuracy)

print("n_neighbors = ",results.index(max_accuracy)+1)
from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier(max_depth=4)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
accuracy_score(y_pred,y_test)
param={

    "criterion":["gini","entropy"],

    "max_depth":[1,2,3,4,5,None]

}
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(clf,param_grid=param,cv=10,n_jobs=-1)

grid.fit(X_train,y_train)

y_pred = grid.predict(X_test)
grid.best_estimator_
grid.best_score_
accuracy_score(y_pred,y_test)