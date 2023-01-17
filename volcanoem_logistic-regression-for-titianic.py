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
import numpy as np

import pandas as pd

import seaborn as sns 

import matplotlib.pylab as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import MinMaxScaler#归一化

from sklearn.metrics import classification_report

data = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

data.head()
test.head()
data.isnull().sum()
data = data.drop(labels=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

test = test.drop(labels=['Name', 'Ticket', 'Cabin'], axis=1)

test.head()
data = data.dropna()

test["Age"] = test["Age"].fillna(np.mean(test["Age"]))

test["Fare"] = test["Fare"].fillna(np.mean(test["Fare"]))

minmax = MinMaxScaler()#实例化函数，这个是个类

data["Fare"] = minmax.fit_transform(np.array(data["Fare"]).reshape(-1,1))

data["Age"] = minmax.fit_transform(np.array(data["Age"]).reshape(-1,1))

test["Age"] = minmax.fit_transform(np.array(test["Age"]).reshape(-1,1))

test["Fare"] = minmax.fit_transform(np.array(test["Fare"]).reshape(-1,1))

PassengerId = test.PassengerId

data_dummy = pd.get_dummies(data[['Sex', 'Embarked']])

data_conti = pd.DataFrame(data, columns=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'], index=data.index)

test_dummy = pd.get_dummies(test[['Sex', 'Embarked']])

test_conti = pd.DataFrame(test, columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'], index=test.index)

data_conti.head()
data = data_conti.join(data_dummy)#拼接的意思

test = test_conti.join(test_dummy)

data.head()

X = data.iloc[:, 1:]##取出特征列

y = data.iloc[:, 0]##取出的就是标签列

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)##数据集的划分

stdsc = StandardScaler()#实例化

X_train_conti_std = stdsc.fit_transform(X_train[['Age', 'SibSp', 'Parch', 'Fare']])#拟合

X_test_conti_std = stdsc.fit_transform(X_test[['Age', 'SibSp', 'Parch', 'Fare']])#拟合



# 将ndarray转为dataframe

X_train_conti_std = pd.DataFrame(data=X_train_conti_std, columns=['Age', 'SibSp', 'Parch', 'Fare'], index=X_train.index)

X_test_conti_std = pd.DataFrame(data=X_test_conti_std, columns=['Age', 'SibSp', 'Parch', 'Fare'], index=X_test.index)

data.head()
#基于训练集使用逻辑回归建模

classifier = LogisticRegression(random_state=0)#实例化算法

classifier.fit(X_train, y_train)#模型训练

 

# 将模型应用于测试集并查看混淆矩阵

y_pred = classifier.predict(X_test)#预测

confusion_matrix = confusion_matrix(y_test, y_pred)#打印混淆矩阵，是很多评分函数的标准来源

print(confusion_matrix)#打印混淆矩阵
test.head()
test.isnull().sum()
predict = classifier.predict(test)
predict
submission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predict})
submission.to_csv("submission.csv",index=False)
pd.read_csv("submission.csv")