import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

%matplotlib inline
df = pd.read_csv('../input/titanic_data.csv')
df.head()
df_null = df.isnull()

sns.heatmap(df_null, yticklabels = False, cbar = False, cmap = 'hot')
df.drop('Cabin', axis = 1, inplace = True)

df_null = df.isnull()

sns.heatmap(df_null, yticklabels = False, cbar = False, cmap = 'hot')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (10,8))

sns.countplot(x = 'Survived', hue = 'Pclass', data = df, ax = ax1)

sns.countplot(x = 'Survived', hue = 'Embarked', data = df, ax = ax2)

sns.countplot(x = 'Survived', hue = 'SibSp', data = df, ax = ax3)

sns.countplot(x = 'Survived', hue = 'Sex', data = df, ax = ax4)

ax3.legend(loc = 1)
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (10,8))

sns.barplot(x = 'Pclass', y = 'Age', data = df, ax = ax1, errwidth = 0)

sns.barplot(x = 'Embarked', y = 'Age', data = df, ax = ax2, errwidth = 0)

sns.barplot(x = 'SibSp', y = 'Age', data = df, ax = ax3, errwidth = 0)

sns.barplot(x = 'Sex', y = 'Age', data = df, ax = ax4, errwidth = 0)

def age_pclass(cols):

    age = cols[0]

    pclass = cols[1]

    if pd.isnull(age):

        if pclass == 1:

            return df['Age'][df['Pclass'] == 1].mean()

        elif pclass == 2:

            return df['Age'][df['Pclass'] == 2].mean()

        else:

            return df['Age'][df['Pclass'] == 3].mean()

    else:

        return age
df['Age'] = df[['Age', 'Pclass']].apply(age_pclass, axis = 1)
df_null = df.isnull()

sns.heatmap(df_null, yticklabels = False, cbar = False, cmap = 'hot')
df_dummy = pd.get_dummies(df.drop(['Name','Ticket'], axis = 1), drop_first = True)
df_dummy.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report
feature = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
X = df_dummy[feature]

y = df_dummy['Survived']
RF = RandomForestClassifier(n_estimators = 200)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

RF.fit(X_train, y_train)
pred = RF.predict(X_test)

print(confusion_matrix(y_test, pred))

print('\n')

print(classification_report(y_test, pred))
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors = 10)
KNN.fit(X_train, y_train)

pred_KNN = KNN.predict(X_test)
print(classification_report(y_test, pred_KNN))
data = {}

for n in range(1,10):

    KNN = KNeighborsClassifier(n_neighbors = n)

    KNN.fit(X_train, y_train)

    pred_KNN = KNN.predict(X_test)

    score = KNN.score(X_test, y_test)

    data[n] = float(score)
data_x = []

data_y = []

for x, y in data.items():

    data_x.append(x)

    data_y.append(y)
sns.pointplot(x = data_x, y = data_y)
KNN2 = KNeighborsClassifier(n_neighbors = 7)

KNN2.fit(X_train, y_train)

pred_KNN2 = KNN2.predict(X_test)

print(classification_report(y_test, pred_KNN2))
from xgboost import XGBClassifier

XG = XGBClassifier()

XG.fit(X_train, y_train)
pred_XG = XG.predict(X_test)
cr = classification_report(y_test, pred_XG)

print(cr)