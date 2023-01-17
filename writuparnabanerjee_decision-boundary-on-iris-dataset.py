# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
df.head()
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



k = []

k.extend(df['species'].values)

e = le.fit_transform(k)



df['species'] = e
df.head()
X=df.iloc[:,[0,2]]
y = df.iloc[:,4]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.3)
X_train.shape
X_test.shape
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
X_train
a=np.arange(start=X_train[:,0].min()-1, stop=X_train[:,0].max()+1, step=0.01)

b=np.arange(start=X_train[:,1].min()-1,stop=X_train[:,1].max()+1, step=0.01)



XX,YY=np.meshgrid(a,b)
XX.shape
input_array=np.array([XX.ravel(),YY.ravel()]).T

labels=clf.predict(input_array)
labels
plt.contourf(XX,YY,labels.reshape(XX.shape),alpha=0.75)

plt.scatter(X_train[:,0],X_train[:,1],c=y_train)