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
file_1='../input/titanic/gender_submission.csv'

file_2= '../input/titanic/train.csv'

file_3='../input/titanic/test.csv'

df_gender=pd.read_csv(file_1)

df_train=pd.read_csv(file_2)

df_test=pd.read_csv(file_3)
df_train.info()
df_train.head()
df_train.drop_duplicates()
my_dict={"Sex":{"male":0,"female":1},"Embarked":{'Q':1,'S':2,'C':3}}

df_train.replace(my_dict,inplace=True)

df_train.head()
train_x=df_train.drop(['Name','Ticket','Cabin','Survived'],axis=1)

train_x['Age']=train_x['Age'].fillna(train_x.Age.mean())

train_x['Embarked']=train_x['Embarked'].fillna(0)

train_x.info()
train_y=df_train['Survived']
df_test.info()
my_dict={"Sex":{"male":0,"female":1},"Embarked":{'Q':1,'S':2,'C':3}}

df_test.replace(my_dict,inplace=True)

test_x=df_test.drop(['Name','Ticket','Cabin'],axis=1)

test_x['Age']=test_x['Age'].fillna(test_x.Age.mean())

test_x['Fare']=test_x['Fare'].fillna(test_x.Fare.mean())
test_x.info()
import matplotlib.pyplot as plt

import seaborn as sns

corr = df_train.corr()

plt.figure(figsize=(14,6))

ax = sns.heatmap(

    corr, annot=True,

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45, 

    horizontalalignment='right'

);
sns.barplot(x="Pclass",y="Survived", data=df_train)
sns.barplot(x="Sex",y="Survived", data=df_train)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

model_log= LogisticRegression()

model_log.fit(train_x,train_y)

pred=model_log.predict(test_x)

test_y=df_gender['Survived']

acc_log=model_log.score(train_x, train_y)

acc_log
from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

model_tree=DecisionTreeClassifier()

model_tree.fit(train_x,train_y)

pred_tree=model_tree.predict(test_x)

acc_tree=model_tree.score(train_x, train_y)

acc_tree
from sklearn.ensemble import RandomForestClassifier

model_forest=RandomForestClassifier(n_estimators=10)

model_forest.fit(train_x,train_y)

pred_forest=model_forest.predict(test_x)

acc_forest=model_forest.score(train_x, train_y)

acc_forest
from sklearn.neighbors import KNeighborsClassifier

K=[]

train=[]

test=[]

for k in range(1,10):

    model= KNeighborsClassifier(n_neighbors=k)

    model.fit(train_x,train_y)

    train_score=model.score(train_x,train_y)

    test_score=model.score(test_x,test_y)

    K.append(k)

    train.append(train_score)

    test.append(test_score)

plt.plot(K,train,color='b')

plt.plot(K,test,color='r')

plt.show()
knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(train_x,train_y)

pred=knn.predict(test_x)

acc_knn=knn.score(train_x, train_y)

acc_knn
from sklearn.naive_bayes import GaussianNB

model_naive= GaussianNB()

model_naive.fit(train_x,train_y)

pred=model_naive.predict(test_x)

acc_naive=model_naive.score(train_x, train_y)

acc_naive
from sklearn import svm

model_svm= svm.SVC()

model_svm.fit(train_x, train_y)

pred=model_svm.predict(test_x)

acc_svm=model_svm.score(train_x, train_y)

acc_svm
models = pd.DataFrame({

    'Model': ['Logistic Regression', 'Decision Tree',

              'Random Forest','KNN', 'Naive Bayes','Support Vector Machines'],

    'Score': [acc_log, acc_tree, acc_forest, 

              acc_knn, acc_naive, acc_svm]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": pred_tree

    })

submission.head()
submission.to_csv('submission.csv', index=False)