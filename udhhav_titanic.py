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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

train.head()
train.isnull().sum()
sns.heatmap(train.isnull(),yticklabels=False,cbar = False, cmap = 'viridis')
sns.countplot(x='Survived',hue='Sex', data=train,palette='RdBu_r')
sns.countplot(x='Survived',hue='Pclass', data=train,palette='rainbow')
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
plt.figure(figsize=(10,10))

sns.boxplot(x='Pclass',y='Age', data=train,palette='winter')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
train.info()
pd.get_dummies(train['Embarked'],drop_first=True).head()
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
train.head()
test.head()
sex = pd.get_dummies(test['Sex'],drop_first=True)

embark = pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

test.head()
test = pd.concat([test,sex,embark],axis=1)

test.head()
X= train.drop('Survived',axis=1)
X.head()
y= train['Survived']

y.head()
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
from sklearn.ensemble import RandomForestClassifier

features = ["Pclass", "male", "SibSp", "Parch"]

X = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.head()
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predictions

    })



submission.to_csv('submission.csv', index=False)
submission.head()