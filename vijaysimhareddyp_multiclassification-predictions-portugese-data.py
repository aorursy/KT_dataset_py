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
df=pd.read_csv(r'/kaggle/input/desafio-worcap-2020/treino.csv')

df
df.head()
df.tail()
df.describe()
df.info()
x=df.iloc[:,:-1].values

y=df.iloc[:,-1].values
x
y
x.shape
y.shape
from sklearn.model_selection import train_test_split

xT,xt,yT,yt=train_test_split(x,y,test_size=0.25,random_state=0)
xT.shape
xt.shape
yT.shape
yt.shape
from sklearn.tree import DecisionTreeClassifier    # importing DecisionTreeClassifier

dt = DecisionTreeClassifier()                    # Storing it in a Variable

dt = dt.fit(xT,yT)                           # Fitting x & y into the variable

y_pred= dt.predict(xT)                # Predicting x and storing it in y_pred

y_pred
yT
from sklearn import metrics

print("Train Accuracy using DecisionTree:",round(metrics.accuracy_score(yT,y_pred)*100,2),"%")
from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(yT, y_pred)

print(confusion)
yt_pred=dt.predict(xt)

yt_pred
yt
from sklearn import metrics

print("Test Accuracy using DecisionTree:",round(metrics.accuracy_score(yt,yt_pred)*100,2),"%")
from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(yt, yt_pred)

print(confusion)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 10,n_jobs=2, random_state= 0)

rfc.fit(x, y)
yR_pred=rfc.predict(xT)
yR_pred
yT
from sklearn import metrics

print("Train Accuracy using RandomForest:",round(metrics.accuracy_score(yT,yR_pred)*100,2),"%")
from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(yT, yR_pred)

print(confusion)
yr_pred=rfc.predict(xt)
yr_pred
yt
from sklearn import metrics

print("Test Accuracy using RandomForest:",round(metrics.accuracy_score(yt,yr_pred)*100,2),"%")
from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(yt, yr_pred)

print(confusion)