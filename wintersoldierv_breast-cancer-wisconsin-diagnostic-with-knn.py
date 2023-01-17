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
df=pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
df.shape
df
df['diagnosis'].value_counts()
df.info()
df.drop(columns='Unnamed: 32',inplace=True)
df.drop(columns='id',inplace=True)
df.columns.values
df
df['diagnosis'].replace('M',1,inplace=True)
df['diagnosis'].replace('B',0,inplace=True)
df
y=df.iloc[:,0].values
y.shape
x=df.iloc[:,1:].values
x.shape
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
np.sqrt(X_train.shape[0])
#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=21)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
accuracy=[]
for i in range (1,31):
    #Import knearest neighbors Classifier model
    from sklearn.neighbors import KNeighborsClassifier
    #Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=i)
    #Train the model using the training sets
    knn.fit(X_train, y_train)
    #Predict the response for test dataset
    y_pred = knn.predict(X_test)
    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    accuracy.append(metrics.accuracy_score(y_test, y_pred))
import matplotlib.pyplot as plt
plt.plot(range(1,31),accuracy)
#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=10)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",round(metrics.accuracy_score(y_test, y_pred)*100,4),"%")
# decision_tree

from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(max_depth=5)
clf.fit(X_train,y_train)
Y_pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,Y_pred)
param_dist={
    "criterion":["gini","entropy"],
    "max_depth":[1,2,3,4,5,None]
}
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(clf,param_grid=param_dist, cv=10, n_jobs=-1)
grid.fit(X_train,y_train)
Y_pred=grid.predict(X_test)
grid.best_estimator_
grid.best_score_
from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(y_test, Y_pred))

