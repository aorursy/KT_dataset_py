# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/train.csv')
data.head()
data.describe()
# посмотрим, где есть пропущенные значения
data.isnull().sum()
# посомтрим, сколько людей выжило
sns.countplot('Survived',data=data)
plt.show()
# Теперь посмотрим, сколько выжило женщин/мужчин
data.groupby(['Sex','Survived'])['Survived'].count()
# Далее надо все-таки выяснить, можно ли за деньги купить себе жизнь
# Построим график того, сколько людей из какого класса выжило
sns.countplot('Pclass',hue='Survived',data=data)
plt.show()
# Исходя из верхнего графика видно, что выживших пассажиров первого класса больше, нежели погибших.
# Давайте посмотрим, сколько пассажиров какого класса было и найдем процент выживших среди каждого класса
sns.countplot('Pclass',data=data)
plt.show()
# Посмотрим на процентное соотношение выживших и умерших по каждому классу
data.groupby(['Pclass','Survived'])['Survived'].count()/data.groupby(['Pclass']).count()['Survived']
# Теперь давайте посмотрим, сколько людей в каком порту село и сколько из них выжило
sns.countplot('Embarked',hue='Survived',data=data)
plt.show()
data.groupby(['Embarked','Survived'])['Survived'].count()/data.groupby(['Embarked']).count()['Survived']
f,ax=plt.subplots(1,2,figsize=(20,5))
sns.countplot('Embarked',hue='Sex',data=data,ax=ax[0])
ax[0].set_title(u'Распредление людей по полу')
sns.countplot('Embarked',hue='Pclass',data=data,ax=ax[1])
ax[1].set_title(u'Распределение людей по классу')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()
data.groupby(['Embarked','Survived', 'Pclass'])['Survived'].count()
data.Age.fillna(data.Age.mean(), inplace=True)
data.Age.isnull().any()
data.Embarked.fillna('S', inplace=True)
data.Embarked.isnull().any()
data.drop(['Name','Ticket','Fare','Cabin','PassengerId'],axis=1,inplace=True)
data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
data['Sex'].replace(['male','female'],[0,1],inplace=True)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 
train,test=train_test_split(data,test_size=0.3,random_state=42,stratify=data['Survived'])
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]
X=data[data.columns[1:]]
Y=data['Survived']
dec_tree=DecisionTreeClassifier()
dec_tree.fit(train_X,train_Y)
DT_prediction=dec_tree.predict(test_X)
print(metrics.accuracy_score(DT_prediction,test_Y))
KNN=KNeighborsClassifier(n_neighbors=7) 
KNN.fit(train_X,np.ravel(train_Y))
KNN_prediction=KNN.predict(test_X)
test_Y = np.ravel(test_Y)
print(metrics.accuracy_score(KNN_prediction,test_Y))