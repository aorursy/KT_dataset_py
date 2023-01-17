import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline
import math
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Load our warnings libraries
import warnings
warnings.filterwarnings('ignore')
# Read  data
df  = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
df.head()
df.shape
df.info()
df.drop("Unnamed: 32",axis=1,inplace=True)
df.isnull().sum()
sns.countplot(df['diagnosis'], palette='RdBu')
df['diagnosis'] = df['diagnosis'].map({'B':1,'M':0})
df.head()
df.drop("id",axis=1,inplace=True)
df.describe()
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit_transform(df['diagnosis'])
df['diagnosis']=le.fit_transform(df['diagnosis'])
df.head()
corr=df.corr()
corr
df.corr()['diagnosis'].sort_values()
X=df.drop(['diagnosis'],axis=1)
y=df['diagnosis']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
model1=lr.fit(X_train,y_train)
prediction1=model1.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,prediction1)
cm
Truepositive=cm[0][0]
Truenegative=cm[1][1]
Falsenegative=cm[1][0]
Falsepositive=cm[0][1]
print('Testing Accuracy:',(Truepositive+Truenegative)/(Truepositive+Truenegative+Falsenegative+Falsepositive))
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
model3=rfc.fit(X_train,y_train)
prediction3=model3.predict(X_test)
confusion_matrix(y_test,prediction3)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction3)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k
model4 = knn.fit(X_train, y_train)
prediction4 = knn.predict(X_test)
confusion_matrix(y_test,prediction4)

scoreList = []
for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn2.fit(X_train, y_train)
    scoreList.append(knn2.score(X_test, y_test))
    
plt.plot(range(1,20), scoreList)
plt.xticks(np.arange(1,20,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 6)  # n_neighbors means k
model4 = knn.fit(X_train, y_train)
prediction4 = knn.predict(X_test)
confusion_matrix(y_test,prediction4)

from sklearn.svm import SVC
svc_model=SVC()
svc_model.fit(X_train,y_train)
y_predict=svc_model.predict(X_test)
y_predict
cm=confusion_matrix(y_test,y_predict)
cm