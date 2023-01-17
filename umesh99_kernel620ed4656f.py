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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

train = train.drop(['Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)

test = test.drop(['Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)

train.head()

for df in [train,test]:

    df['Sex_binary'] = df['Sex'].map({'male':1,'female':0})



train['Age'] = train['Age'].fillna(0)

test['Age'] = test['Age'].fillna(0)



features = ['Pclass','Age','Sex_binary','SibSp','Parch']

target = ['Survived']

train[features].head()
train[target].head().values
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(random_state=1,

                             n_estimators=190,

                            max_depth=10,

                             bootstrap=True,

                            n_jobs=-1,verbose=False)



clf.fit(train[features],train[target])
pridiction = clf.predict(test[features])



pridiction
test.head()
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':pridiction})

submission.head()
filename = 'Titanic Prediction -2.csv'

submission.to_csv(filename, index = False)

print('saved file:',filename)