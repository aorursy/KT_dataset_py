import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/titanic/train.csv')

gs = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.columns
x_train = train.drop(['Survived', 'Name','PassengerId','Embarked','Ticket','Cabin'], axis = 1) 

y_train = train['Survived']

true = gs['Survived']
test.columns
t = test.drop(['Name','PassengerId','Embarked','Ticket','Cabin'], axis = 1) 
x_train=x_train.replace('male',0)

x_train=x_train.replace('female',1)

t=t.replace('male',0)

t=t.replace('female',1)

x_train=x_train.replace(np.nan,0)

t=t.replace(np.nan,0)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train,y_train)
pred = logreg.predict(t)

pred
accuracy_level = logreg.score(t,true)

accuracy_level
print(f'The accuracy level of above predictive model is {accuracy_level}')