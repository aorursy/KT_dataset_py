# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plotting

fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 12

fig_size[1] = 12

plt.rcParams["figure.figsize"] = fig_size

print(plt.rcParams.get('figure.figsize')) #making output images bigger



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

train_data.head()
train_data.describe(include = 'all')
train_data.describe(include=['O'])
train_data.info()

print('_'*40)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

newdf = train_data.select_dtypes(include=numerics)

newdf.hist()
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_data.head()
test_data.describe()
women_survival_data = train_data[train_data.Sex == 'female']["Survived"]

women_survival_rate = sum(women_survival_data) / len(women_survival_data)

print("women survival rate is ", round(women_survival_rate,2))
men_survival_data = train_data[train_data.Sex == "male"]["Survived"]

men_survival_rate = sum(men_survival_data) / len(men_survival_data)

print("men_survival_rate is", round(men_survival_rate,2) )
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[['Sex','Survived']].groupby('Sex', as_index = False).mean().sort_values(by='Survived', ascending=False)
train_data[['Embarked','Survived']].groupby('Embarked', as_index = False).mean().sort_values(by='Survived', ascending=False)
from sklearn.model_selection import train_test_split

y = train_data["Survived"].astype('bool') #change from 64bit int into bool

#df['column_name'] = df['column_name'].astype('bool')



features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]

X = pd.get_dummies(train_data[features]) #changing categorical data into binary features

X_submit = pd.get_dummies(test_data[features])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.describe()
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=25)

model.fit(X_train, y_train)

y_pred = model.predict(X_test).astype('int')

print("model fit complete!")
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=25)

model.fit(X,y)

predictions = model.predict(X_submit).astype('int')

print("model fit complete!")
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")