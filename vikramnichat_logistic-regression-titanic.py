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

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns
import pandas as pd

df = pd.read_csv("../input/titanic_train.csv")
df.head()
df.isnull
sns.heatmap(df.isnull(),yticklabels=False)
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=df)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=df)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=df)
#filling the NA value by mean of age

plt.figure(figsize=(12,7))

sns.boxplot(x='Pclass',y='Age',data=df)
df.fillna(df.mean(),inplace=True)
df.head()
sns.heatmap(df.isnull(),yticklabels=False)
df.drop('Cabin',axis=1,inplace=True)
sns.heatmap(df.isnull(),yticklabels=False)
df.head()
pd.get_dummies(df['Embarked'],drop_first=True).head()
embark = pd.get_dummies(df['Embarked'], drop_first=True)
sex = pd.get_dummies(df['Sex'], drop_first=True)
df = pd.concat([df,sex,embark], axis=1)
df.head()
df.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
df.drop(['PassengerId'],axis=1,inplace=True)
df.head()
#Applying Logistic Regression 

X=df.drop('Survived',axis=1)

Y=df['Survived']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
#importing Logistic regresssion

from sklearn.linear_model import LogisticRegression

model1=LogisticRegression()

model1=model1.fit(X_train,Y_train)
model1.score(X_test,Y_test)