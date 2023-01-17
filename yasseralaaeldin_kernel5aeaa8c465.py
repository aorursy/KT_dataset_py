# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.describe()
fig , axes = plt.subplots(ncols =2,figsize=(10,10))

train.Age.hist(bins = 8,ax = axes[0])

plt.title('Age')

train.Fare.hist(bins=15,ax = axes[1])

plt.title('Fare')

fig , axes  = plt.subplots(ncols = 3 , nrows=2,figsize = (15,7))

sns.countplot(data = train , x = 'Sex',ax=axes[0][0])

sns.countplot(data = train , x = 'Survived',ax=axes[0][1])

sns.countplot(data = train , x = 'Pclass',ax=axes[0][2])

sns.countplot(data = train , x = 'SibSp',ax=axes[1][0])

sns.countplot(data = train , x = 'Parch',ax=axes[1][1])

sns.countplot(data = train , x = 'Embarked',ax=axes[1][2])

fig , axes = plt.subplots(ncols=2,figsize=(15,10))

(train.groupby(['Pclass','Survived']).count()['PassengerId'] / train.groupby(['Survived']).count()['PassengerId']).plot(kind='bar',colors=['tab:red','tab:blue'],ax = axes[0])

(train.groupby(['Survived','Sex']).count()['PassengerId'] / train.groupby(['Sex']).count()['PassengerId']).plot(kind='bar',colors=['tab:red','tab:blue'],ax = axes[1])
train.groupby(['Survived']).describe()['Age']

#the min is the only one has big difference between the two classes
train.groupby(['Survived']).hist('Age',bins = 20)

plt.title('non survivors')

plt.title('survivors')

plt.xticks(np.arange(0,80,5))

df = train

df['Sex_encoded'] = np.where(train.Sex == 'female',0,1)

df.Fare.fillna(df.Fare.mean(),inplace = True)

df.drop(columns = ['Ticket','Name','Cabin'],inplace = True)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['Embarked'] = le.fit_transform(df['Embarked'].fillna('NA'))
from sklearn.model_selection import train_test_split

y = df.Survived

X = df.drop(columns = ['Survived','Sex','PassengerId'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from xgboost.sklearn import XGBClassifier

xgb =XGBClassifier(n_estimators=1000,learning_rate=0.02

                  )

xgb.fit(X_train,y_train)
from sklearn.metrics import classification_report

y_pred = xgb.predict(X_test)

print(classification_report(y_test, y_pred))
for name, importance in zip(X.columns, xgb.feature_importances_):

    print(name, "=", importance)
features = X.columns

importances = xgb.feature_importances_

indices = np.argsort(importances)



plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
df['Age_Tier'] = df.Age

for index , a in df.Age.iteritems():

    if(a == np.nan):

        df['Age_Tier'][index] = np.nan

    elif((a <= 3.0)):

        df['Age_Tier'][index] = '1 -> 3'

    elif((a > 3.0) and (a <= 16.0)):

        df['Age_Tier'][index] = '4 -> 16'

    elif((a > 16.0) and (a <= 36.0)):

        df['Age_Tier'][index] = '17 -> 36'

    elif((a > 36) & (a <= 52)):

        df['Age_Tier'][index] = '37 -> 52'

    elif((a > 52)):

        df['Age_Tier'][index] = '> 52'

df.Age_Tier.fillna("ef",inplace = True)

df[(df['Age_Tier'] == "ef") & (df['Survived'] == 1)]['Age_Tier'] = '1-3'

df[(df['Age_Tier'] == "ef") & (df['Survived'] == 0)]['Age_Tier'] = '17 -> 36'
df['Age_Tier'] = le.fit_transform(df['Age_Tier'])
from sklearn.model_selection import train_test_split

y = df.Survived

X = df.drop(columns = ['Survived','Sex','PassengerId'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from xgboost.sklearn import XGBClassifier

xgb =XGBClassifier(n_estimators=1000,learning_rate=0.02)

xgb.fit(X_train,y_train)



from sklearn.metrics import classification_report

y_pred = xgb.predict(X_test)

print(classification_report(y_test, y_pred))
features = X.columns

importances = xgb.feature_importances_

indices = np.argsort(importances)



plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
test['Sex_encoded'] = X.Sex_encoded

test['Age_Tier'] = X.Age_Tier

test.Embarked = X.Embarked

predictions = xgb.predict(test[X.columns])
submission = pd.DataFrame({'PassengerId':test.PassengerId ,'Survived':predictions})

submission.to_csv('Titanic prediction.csv',index = False)