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
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
df = train

df.head(5)
df.info()
import matplotlib.pyplot as plt

import seaborn as sns





df['Pclass'] = df['Pclass'].astype('category')

df['Sex'] = df['Sex'].astype('category')

df['Embarked'] = df['Embarked'].astype('category')



del df['PassengerId']





df['Age Group'] = np.where(df['Age']<=1, 'Infant',

                           np.where(df['Age']<=5,'Small Child',

                                    np.where(df['Age']<=12, 'Child',

                                    np.where(df['Age']<=19, 'Teen',

                                             np.where(df['Age']<=25,'Young',

                                                     np.where(df['Age']<60,'Adult',

                                                                      'Elder'))))))



df = df[df['Fare'] > 0.0]



#Since most of cabins name are given for Pclass 1 and 2. It means that they were only assigned

df['Cabin'] = df['Cabin'].str[:1]

df['Cabin'].fillna("Unavailable",inplace = True)

df = df.dropna()



#Deleting Ticket becasue it of simple no use.

del df['Ticket']





df['Age'] = df['Age'].astype(int)

df['FamilySize'] = df['SibSp'] + df['Parch']



#Converting Name into title

df['Name'] = df['Name'].str.split(',').str[1]

df['Name'] = df['Name'].str.split('.').str[0]

df['Name'] = df['Name'].str.strip()

df['Name'] = df['Name'].replace(['Col','Major','Mr.','Don','Sir','Capt'], 'Mr')

df['Name'] = df['Name'].replace([' Ms',' Mlle','the Countess'],'Miss')

df['Name'] = df['Name'].replace(['Lady','Mme'],'Mrs')



df['Cabin'] = df['Cabin'].astype('category')

df['Name'] = df['Name'].astype('category')

df['Age Group'] = df['Age Group'].astype('category')
df['Sex'].value_counts()
df['Pclass'].value_counts()
df['Embarked'].value_counts()
df['Cabin'].value_counts()
#Fare Distribution

mydata = df.dropna(subset=['Fare'])

plt.figure(figsize=(10,4))

sns.distplot(mydata['Fare'], kde=False)
sns.set(style="whitegrid")

plt.figure(figsize=(8,5))

ax = sns.boxplot(x = 'Pclass', y = 'Fare', data=df, orient="v")
sns.set(style="whitegrid")

plt.figure(figsize=(8,5))

ax = sns.boxplot(x = 'Sex', y = 'Fare', data=df, orient="v")
#sns.set(style='darkgrid')

plt.figure(figsize=(5,3))

ax = sns.countplot(x='Sex', data=df)


plt.figure(figsize=(5,3))

ax = sns.countplot(x='Pclass', hue = 'Survived', data=df)
plt.figure(figsize=(8,3))

ax = sns.countplot(x='Age Group', hue = 'Survived', data=df)
sns.catplot(y="Survived", x ="Age", hue = 'Sex', row = 'Pclass', kind = 'swarm',orient="h", height=2, aspect=4, data=df)
plt.figure(figsize=(5,3))

ax = sns.countplot(x='Pclass', hue='Survived',  data=df)
sns.catplot(x="Sex", y="Survived", hue="Pclass", kind="bar", data=df)
sns.catplot(x="Sex", y="Survived", hue="Pclass", kind="point", data=df)
sns.catplot(x="Sex", y="Survived", hue="Age Group", kind="bar", data=df)
sns.catplot(x="Sex", y="Survived",

            kind="violin", split=True, data=df);
#cleaning rest of non-numeric variables

categorical_columns = ['Pclass','Name', 'Sex', 'Cabin','Embarked','Age Group']

# transform the categorical columns

df = pd.get_dummies(df, columns=categorical_columns)



import sklearn

from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()

scaler.fit(df)

df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)



from sklearn.feature_selection import SelectKBest, chi2

X = df.loc[:,df.columns!='Survived']

y = df[['Survived']]

selector = SelectKBest(chi2, k=3)

selector.fit(X, y)

X_new = selector.transform(X)

#print(X.columns[selector.get_support(indices=True)])
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)



from sklearn.linear_model import LogisticRegression



logisticRegr = LogisticRegression(random_state=1,solver='liblinear',fit_intercept=True)

#use logistic model to fit training data

logisticRegr.fit(X_train, y_train)

#generate predicted classes for test data

logis_pred = logisticRegr.predict(X_test)

#generate predicted probabilites for test data

logis_pred_prob = logisticRegr.predict_proba(X_test)



from sklearn.metrics import accuracy_score

score = accuracy_score(y_test,logis_pred)

print('Accuracy :',score)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, logis_pred)
from sklearn.tree import DecisionTreeClassifier



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)







DecisionT = DecisionTreeClassifier(random_state=1)

#use logistic model to fit training data

DecisionT.fit(X_train, y_train)

#generate predicted classes for test data

dt_pred = DecisionT.predict(X_test)

#generate predicted probabilites for test data

dt_prob = DecisionT.predict_proba(X_test)



score = accuracy_score(y_test,dt_pred)

print('Accuracy :',score)

confusion_matrix(y_test, dt_pred)
from sklearn import svm

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

clf_svc = svm.SVC(kernel='linear')

clf_svc.fit(X_train,y_train)

svm_pred = clf_svc.predict(X_test)

score = accuracy_score(y_test,svm_pred)

print('Accuracy :',score)

confusion_matrix(y_test, svm_pred)