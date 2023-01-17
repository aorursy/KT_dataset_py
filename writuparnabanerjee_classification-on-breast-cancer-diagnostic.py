# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df.head()
df.info()
df.drop(columns='Unnamed: 32',inplace=True)
df.shape
plt.subplots(figsize=(15,5))

sns.heatmap(df.corr(),linewidth=0.1,linecolor='white')

plt.show()
df.drop(columns='id',inplace=True)
a={'M':int(1),'B':int(0)}

df['diagnosis'].replace(a,inplace=True)
df.head()
X=df.iloc[:,1:].values

y=df.iloc[:,0].values
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X=scaler.fit_transform(X)

X
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=1)
#Using Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf= DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred,y_test)
from sklearn.metrics import classification_report

print(classification_report(y_pred,y_test))
tn,fp,fn,tp = confusion_matrix(y_pred,y_test).ravel()
print('Acurracy:',(tp+tn)/(tp+fn+fp+tn))

P=tp/(tp+fp)

R=tp/(tp+fn)

F=(2*P*R)/(P+R)

print('Precision:',P)

print('Recall:',R)

print('F1 score',F)

#Using KNN
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
accuracy=[]

for i in range (1,26):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    accuracy.append(accuracy_score(y_test,knn.predict(X_test)))
len(accuracy)
plt.plot(range(1,26),accuracy)
knn=KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train,y_train)
y_pred1=knn.predict(X_test)
accuracy_score(y_pred1,y_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred1)
from sklearn.metrics import classification_report

print(classification_report(y_pred1,y_test))
tn1,fp1,fn1,tp1 = confusion_matrix(y_pred1,y_test).ravel()
print('Acurracy:',(tp1+tn1)/(tp1+fn1+fp1+tn1))

P1=tp1/(tp1+fp1)

R1=tp1/(tp1+fn1)

F1=(2*P1*R1)/(P1+R1)

print('Precision:',P1)

print('Recall:',R1)

print('F1 score',F1)

#The Knn model is preferable because the the number of malignant cancers detected as benign by the confusion_matrix 

#is 0. The recall is 100% for this model.