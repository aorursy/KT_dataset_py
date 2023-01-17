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
df = pd.read_csv("../input/titanic/train.csv")
df.sample(5)
X = df.drop('Survived',axis=1)

y = df['Survived']
X.head()
y.head()
X.drop(['PassengerId','Pclass','Name','Ticket','Cabin'],axis=1,inplace=True)
X.info()
X['Age'] = X.Age.fillna(df.Age.median())
X.info()
X['Embarked'] = X['Embarked'].fillna(X['Embarked'].value_counts().sort_values(ascending=False).index[0])
X.info()
X.isnull().sum()
X.head()
X1 = pd.get_dummies(X[['Sex','Embarked']],drop_first=True)
Train_data = pd.concat([X,X1],axis=1)
Train_data.head()
Train_data.drop(['Sex','Embarked'],axis=1,inplace=True)
Train_data.head()
Train_data['Age'],Train_data['Fare']=Train_data['Age'].astype('int64'),Train_data['Fare'].astype('int64')
Train_data.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train_data=scaler.fit_transform(Train_data)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

accuracy_rate=[]

for i in range(1,25):

    knn = KNeighborsClassifier(n_neighbors=i)

    score = cross_val_score(knn,train_data,y,cv=10)

    accuracy_rate.append(score.mean())
import matplotlib.pyplot as plt

plt.plot(range(1,25),accuracy_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)

plt.title("ACCURACY_RATE vs KNN")

plt.xlabel("K")

plt.ylabel("accuracy_rate")
knn = KNeighborsClassifier(n_neighbors=8)

knn.fit(train_data,y)
knn.score(train_data,y)
test = pd.read_csv("../input/titanic/test.csv")

test.head()
test.drop(['PassengerId','Pclass','Name','Ticket','Cabin'],axis=1,inplace=True)
test.head()
test.info()
test['Age'] = test.Age.fillna(test.Age.median())

test['Embarked'] = test['Embarked'].fillna(test['Embarked'].value_counts().sort_values(ascending=False).index[0])

test1= pd.get_dummies(test[['Sex','Embarked']],drop_first=True)
Test = pd.concat([test,test1],axis=1)
Test.head()
Test.drop(['Sex','Embarked'],axis=1,inplace=True)
Test['Fare'].isna().sum()
Test['Fare']=Test['Fare'].fillna(Test['Fare'].median())
Test['Age'],Test['Fare']=Test['Age'].astype('int64'),Test['Fare'].astype('int64')
test_data=scaler.transform(Test)

test_predict = knn.predict(test_data)


knn.score(test_data,test_predict)
y = pd.DataFrame({'survived':test_predict})
output = pd.concat([Test,y])
output.to_csv('output.csv')